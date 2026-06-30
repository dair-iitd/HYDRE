import sys
import ast
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os
import time
from torch import Tensor
import copy
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count
from transformers import pipeline
import random
import torch

# OpenNRE integration
import pathlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Entity masking parameters
MASK = True
MASK_TOKEN = "[UNK]"  # only for bert-base-uncased

# File paths - NYT10m data
PATH = os.path.join(ROOT_DIR, "nyt10m", "nyt10m_train_opennre_clean_hfmre256.jsonl")
FILTERED_PATH = os.path.join(ROOT_DIR, "nyt10m", "nyt10m_train_opennre_clean_hfmre256.jsonl")
QUERY_PATH = os.path.join(ROOT_DIR, "nyt10m", "eng_Latn_final.jsonl")
OUT_DIR = os.path.join(ROOT_DIR, "nyt10m")

# Retrieval model configuration
RETRIEVAL_MODEL_NAME = "intfloat/e5-large-v2"
ret_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
ret_model_hidden_size = 1024

# PARE model configuration
PARE_TOP_FILE = os.path.join(ROOT_DIR, "nyt10m", "opennre_ckpt3_candidates_all_rels.jsonl")
PARE_PRETRAIN_PATH = 'bert-base-uncased'
PARE_CKPT = os.path.join(ROOT_DIR, "PARE", "ckpt", "bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar")
REL2ID_PATH = os.path.join(ROOT_DIR, "nyt10m", "nyt10m_rel2id.json")

# PARE Backend configuration
PARE_BACKEND = "pare"
PARE_ROOT = os.path.join(ROOT_DIR, "PARE")
REGEN_PARE_TOP_FILE = False

# Processing parameters
BATCH_SIZE = 16
TOPK = 5
LANG = "en"

# Runtime controls (overridden by CLI)
NUM_WORKERS = None
MAX_TRAIN_LINES = None
MAX_QUERY_LINES = None
RETRIEVAL_BATCH_SIZE = 1000
NER_DEVICE = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def seed_everything(seed=1234):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=2025)

def get_gpu_mem_usage():
    """Get current GPU memory usage in MB"""
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Average pooling over hidden states"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def max_min_normalize(arr: Tensor, dim) -> Tensor:
    """Normalize tensor using min-max normalization"""
    assert len(arr.shape) == 2, "Input tensor must be 2-dimensional"
    return (arr - arr.min(dim=dim, keepdim=True).values) / (arr.max(dim=dim, keepdim=True).values - arr.min(dim=dim, keepdim=True).values)

# ============================================================================
# PARE SCORE COMPUTATION
# ============================================================================

def get_pare_scores(data_file):
    """
    Compute PARE scores for relation extraction using PARE codebase
    Returns: (n, num_relations) tensor of scores
    """
    return get_pare_scores_pare(data_file)

def get_pare_scores_pare(data_file):
    """Compute per-bag relation scores using standalone PARE codebase."""
    import sys
    sys.path.insert(0, PARE_ROOT)
    import encoder
    import framework
    import model1
    import model2

    rel2id = json.load(open(REL2ID_PATH))
    passage_encoder = encoder.PassageEncoder(
        pretrain_path=PARE_PRETRAIN_PATH,
        batch_size=BATCH_SIZE,
        mask_entity=False
    )
    if "nyt" in data_file.lower() or "nyt" in QUERY_PATH.lower():
        model_ = model2.PassageAttention(passage_encoder, len(rel2id), rel2id)
    else:
        model_ = model1.PassageAttention(passage_encoder, len(rel2id), rel2id)

    from framework.data_loader import PassageRELoader
    test_loader = PassageRELoader(
        path=data_file,
        rel2id=model_.rel2id,
        tokenizer=model_.passage_encoder.tokenize,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    framework_ = framework.PassageRE(
        model=model_,
        train_path=None, val_path=None, test_path=data_file,
        ckpt=PARE_CKPT, batch_size=BATCH_SIZE, max_epoch=1,
        lr=2e-5, weight_decay=1e-5, opt='adamw', warmup_step=0, devices=[0]
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_obj = torch.load(PARE_CKPT, map_location=device)
    framework_.load_state_dict(ckpt_obj['state_dict'])
    framework_.model.eval()

    pred_result = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Scoring with PARE"):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try: data[i] = data[i].cuda()
                    except: pass
            bag_name = data[1]
            token, mask = data[2].squeeze(1), data[3].squeeze(1)
            logits = framework_.model(token, mask, False)
            for b in range(logits.shape[0]):
                num_class = framework_.model.module.num_class if hasattr(framework_.model, 'module') else framework_.model.num_class
                id2rel = framework_.model.module.id2rel if hasattr(framework_.model, 'module') else framework_.model.id2rel
                for relid in range(num_class):
                    rel = id2rel[relid]
                    if rel == 'NA': continue
                    pred_result.append({'entpair': bag_name[b][:2], 'relation': rel, 'score': logits[b][relid].item()})

    entpair_to_row = {}
    row = 0
    for item in pred_result:
        ep = tuple(item['entpair'])
        if ep not in entpair_to_row:
            entpair_to_row[ep] = row
            row += 1
    scores = torch.zeros((len(entpair_to_row), len(rel2id)))
    for item in pred_result:
        ep = tuple(item['entpair'])
        r = entpair_to_row[ep]
        relid = rel2id[item['relation']]
        scores[r][relid] = item['score']
    return scores

# ============================================================================
# BERTSCORE COMPUTATION - ABLOTION 2: NO PARE CONFIDENCE
# ============================================================================
# In this ablation, PARE scores are zeroed out (line 203 of source abl2.py).
# Only BERTScore/sentence embeddings are used for retrieval.

def compute_bertscore_for_query(queries, doc_bags):
    """
    Compute BERTScore for all queries and document sets.
    ABLOTION 2: PARE scores are zeroed out. Only BERTScore is used for retrieval.
    Args:
        queries: List of query sentences
        doc_bags: List of document bags
    Returns:
        Tensor of shape (len(queries), len(doc_bags), num_relations)
    """
    assert type(doc_bags) == list
    print("Computing BERTScore for all queries and document sets (NO PARE CONFIDENCE - Ablation 2)...")

    # Prepare training sentences and create PARE data file
    train_sents = []
    data_file = f"pare_data_{LANG}_example_bags_{len(doc_bags)}.jsonl"
    idx = 0

    # Create PARE data file
    with open(data_file, "w") as f:
        for doc_set in doc_bags:
            rels = doc_set["relations"]
            hid = str(idx)
            tid = str(idx+1)
            idx += 2
            train_sents.append(" ".join(doc_set["texts"]))
            for doc_tup in zip(doc_set["texts"], doc_set["hs"], doc_set["ts"]):
                doc = {"text": doc_tup[0], "h": doc_tup[1], "t": doc_tup[2]}
                doc["h"]["id"] = hid
                doc["t"]["id"] = tid
                for rel in rels:
                    doc["relation"] = rel
                    f.write(json.dumps(doc)+"\n")

    # Compute PARE scores but then zero them out (ablation)
    pare_scores_path = f"pare_scores_{LANG}_notnorm_example_bags_{len(doc_bags)}.pt"
    if os.path.exists(pare_scores_path):
        pare_scores = torch.load(pare_scores_path, map_location="cpu")
    else:
        pare_scores = get_pare_scores(data_file)
        torch.save(pare_scores, pare_scores_path)
    print("PARE scores shape: ", pare_scores.shape)
    pare_scores = max_min_normalize(pare_scores, dim=0)

    # ABLOTION 2: Zero out PARE scores, only BERTScore is used
    pare_scores = torch.zeros_like(pare_scores)
    print("PARE scores zeroed out (ablation). Only BERTScore will be used.")

    if MASK:
        tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        if NER_DEVICE is not None:
            ner_device = int(NER_DEVICE)
        else:
            ner_device = 0 if torch.cuda.is_available() else -1
        ner = pipeline('ner', model=model_ner, tokenizer=tokenizer_ner, device=ner_device)

        def _mask_sents(sents):
            masked = list(sents)
            for i in tqdm(range(0, len(masked), BATCH_SIZE), desc="Masking entities"):
                batch = masked[i:i+BATCH_SIZE]
                ner_res = ner(batch, batch_size=BATCH_SIZE)
                for j in range(len(batch)):
                    idxs = []
                    for ent in ner_res[j]:
                        if ent.get("entity") == "O":
                            continue
                        idxs.append((ent["start"], ent["end"]))
                    idxs = sorted(idxs, key=lambda x: x[0])
                    diff = 0
                    sent = batch[j]
                    for start0, end0 in idxs:
                        start = start0 + diff
                        end = end0 + diff
                        sent = sent[:start] + MASK_TOKEN + sent[end:]
                        diff += len(MASK_TOKEN) - (end - start)
                    masked[i + j] = sent
            return masked

        train_sents = _mask_sents(train_sents)
        queries = _mask_sents(queries)

    query_pref = "query: "
    train_sents = [(query_pref + sent) for sent in train_sents]
    queries = [(query_pref + sent) for sent in queries]

    doc_embs = np.zeros((len(train_sents), ret_model_hidden_size), dtype=np.float32)
    for j in tqdm(range(0, len(train_sents), BATCH_SIZE), desc="Computing document embeddings"):
        batch = train_sents[j:j+BATCH_SIZE]
        doc_embs[j:j+BATCH_SIZE] = ret_model.encode(batch, normalize_embeddings=True)

    query_embs = np.zeros((len(queries), ret_model_hidden_size), dtype=np.float32)
    for i in tqdm(range(0, len(queries), BATCH_SIZE), desc="Computing query embeddings"):
        batch = queries[i:i+BATCH_SIZE]
        query_embs[i:i+BATCH_SIZE] = ret_model.encode(batch, normalize_embeddings=True)

    sent_scores = torch.from_numpy(query_embs @ doc_embs.T)

    sent_scores = max_min_normalize(sent_scores, dim=1)
    num_relations = pare_scores.shape[1]
    sent_scores = sent_scores.repeat_interleave(num_relations, dim=1).view(len(queries), len(train_sents), num_relations)
    sent_scores = sent_scores + pare_scores
    print("After adding PARE scores, sent_scores shape: ", sent_scores.shape)

    return sent_scores

# ============================================================================
# TOP-K RETRIEVAL FUNCTIONS
# ============================================================================

def best_for_a_slot(bert_scores, j, gr_id, doc_bags, rel):
    """
    Find the best document bag for a specific query-relation pair
    """
    sorted_sets = sorted(enumerate(bert_scores[j, :, gr_id]), key=lambda x: x[1], reverse=True)

    for s in sorted_sets:
        flag = 0
        for r in doc_bags[s[0]]["relations"]:
            if r == rel:
                flag = 1
                break
        if flag == 1:
            return s[0]
    return sorted_sets[0][0] if sorted_sets else None

def retrieve_top_k_sets_helper1(queries, doc_bags, k, idpredrep, rel2id, idx):
    """Helper function to compute and save scores"""
    bert_scores = compute_bertscore_for_query(queries, doc_bags)
    torch.save(bert_scores, f"bert_scores_{LANG}_{os.path.basename(RETRIEVAL_MODEL_NAME)}_{MASK}_{idx}_sent_bag_prompt.pt")

def retrieve_top_k_sets_helper2(queries, doc_bags, k, idpredrep, rel2id, idx):
    """Helper function to retrieve top-k sets using pre-computed scores"""
    all_sents = []
    id2rel = {v: k for k, v in rel2id.items()}

    # Load pre-computed scores
    bert_scores = torch.load(f"bert_scores_{LANG}_{os.path.basename(RETRIEVAL_MODEL_NAME)}_{MASK}_{idx}_sent_bag_prompt.pt")

    # Validate score tensor shape.
    if bert_scores.ndim != 3:
        raise ValueError(
            f"BERT scores tensor has unexpected shape: {bert_scores.shape}. "
            f"Expected 3D tensor (n_queries, n_docs, n_relations). "
            f"n_relations should be {len(rel2id)}. "
            f"This usually means the Stage 2 script returned a 1D or 2D tensor "
            f"(e.g., missing the query dimension or relation dimension). "
            f"Check that compute_bertscore_for_query returns a tensor of shape "
            f"(len(queries), len(train_bags), {len(rel2id)})."
        )
    n_queries, n_docs, n_relations = bert_scores.shape
    if n_relations != len(rel2id):
        raise ValueError(
            f"BERT scores tensor has {n_relations} relations but rel2id has {len(rel2id)} entries. "
            f"Checkpoint and rel2id file may be mismatched (e.g., NYT vs Wiki checkpoint)."
        )
    if n_queries != len(queries):
        raise ValueError(
            f"BERT scores tensor has {n_queries} queries but {len(queries)} queries were passed. "
            f"Check that the query file used for Stage 2 matches the one passed here."
        )
    if n_docs != len(doc_bags):
        raise ValueError(
            f"BERT scores tensor has {n_docs} documents but {len(doc_bags)} doc_bags were passed. "
            f"Check that the training data used for Stage 2 matches the one passed here."
        )

    # Prepare arguments for parallel processing
    sort_order = []
    for j, query in enumerate(tqdm(queries, desc="Preprocess for Retrieving top-k sets")):
        for gr in idpredrep[j]:
            if gr not in rel2id:
                print("Relation not found in rel2id")
                continue
            gr_id = rel2id[gr]
            sort_order.append((j, gr_id))

    # Parallel processing using multiprocessing
    start_time = time.time()
    n_workers = cpu_count() if (NUM_WORKERS is None or NUM_WORKERS <= 0) else min(cpu_count(), NUM_WORKERS)
    with Pool(n_workers) as p:
        results = p.starmap(best_for_a_slot, [(bert_scores, j, gr_id, doc_bags, id2rel[gr_id]) for j, gr_id in sort_order])
    print("Time taken for parallel processing: ", time.time()-start_time)

    # Assemble top-k sets
    for j, query in enumerate(tqdm(queries, desc="Retrieving top-k sets")):
        top_sets = []
        for j2, gr in enumerate(idpredrep[j]):
            if gr not in rel2id:
                print("Relation not found in rel2id")
                continue
            gr_id = rel2id[gr]
            best_bag_idx = results[j*k+j2]
            if best_bag_idx is None:
                best_bag_idx = int(torch.argmax(bert_scores[j, :, gr_id]).item())
            bg = copy.deepcopy(doc_bags[best_bag_idx])
            top_sets.append(bg)

        # If we skipped relations (missing from rel2id), pad with best overall bags.
        if len(top_sets) < k:
            best_overall = torch.argsort(torch.max(bert_scores[j], dim=1).values, descending=True)
            for cand_idx in best_overall.tolist():
                if len(top_sets) >= k:
                    break
                top_sets.append(copy.deepcopy(doc_bags[int(cand_idx)]))

        assert len(top_sets) == k, f"Top sets length mismatch: {len(top_sets)} != {k}"
        all_sents.append(top_sets)

    return all_sents

def retrieve_top_k_sets(queries, doc_bags, k, idpredrep, rel2id):
    """
    Main function to retrieve top-k document sets for each query
    """
    all_sents = []
    batch_size = RETRIEVAL_BATCH_SIZE

    # First, compute and save scores for all batches
    print("Computing scores for all batches...")
    for i in tqdm(range(0, len(queries), batch_size), desc="Computing scores"):
        batch_queries = queries[i:i+batch_size]
        batch_idpredrep = idpredrep[i:i+batch_size]
        retrieve_top_k_sets_helper1(batch_queries, doc_bags, k, batch_idpredrep, rel2id, i)

    # Then, retrieve top-k sets using pre-computed scores
    print("Retrieving top-k sets using pre-computed scores...")
    for i in tqdm(range(0, len(queries), batch_size), desc="Retrieving top-k sets"):
        batch_queries = queries[i:i+batch_size]
        batch_idpredrep = idpredrep[i:i+batch_size]
        all_sents.extend(retrieve_top_k_sets_helper2(batch_queries, doc_bags, k, batch_idpredrep, rel2id, i))

    return all_sents

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HYDRE Bag Selection (Stage 2) - Ablation 2: Without PARE Confidence")
    parser.add_argument("--train_path", type=str, default=os.path.join(ROOT_DIR, "nyt10m", "nyt10m_train_opennre_clean_hfmre256.jsonl"))
    parser.add_argument("--query_path", type=str, default=os.path.join(ROOT_DIR, "nyt10m", "eng_Latn_final.jsonl"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "nyt10m"))
    parser.add_argument("--pare_top_file", type=str, default=os.path.join(ROOT_DIR, "nyt10m", "opennre_ckpt3_candidates_all_rels.jsonl"))
    parser.add_argument("--rel2id_path", type=str, default=os.path.join(ROOT_DIR, "nyt10m", "nyt10m_rel2id.json"))
    parser.add_argument("--pare_ckpt", type=str, default=PARE_CKPT, help="PARE checkpoint used inside Stage 2 for scoring (defaults to config PARE_CKPT)")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--retrieval_batch_size", type=int, default=RETRIEVAL_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_lines", type=int, default=0)
    parser.add_argument("--max_query_lines", type=int, default=0)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--no-mask", dest="mask", action="store_false")
    parser.set_defaults(mask=MASK)
    parser.add_argument("--ner_device", type=int, default=None, help="HF pipeline device for NER masking (e.g. 0, 1, or -1 for CPU)")

    args = parser.parse_args()

    # Update global config with args
    FILTERED_PATH = args.train_path
    QUERY_PATH = args.query_path
    OUT_DIR = args.out_dir
    PARE_TOP_FILE = args.pare_top_file
    REL2ID_PATH = args.rel2id_path
    LANG = args.lang
    TOPK = args.topk
    BATCH_SIZE = args.batch_size
    RETRIEVAL_BATCH_SIZE = args.retrieval_batch_size
    NUM_WORKERS = args.num_workers if args.num_workers else None
    MAX_TRAIN_LINES = args.max_train_lines if args.max_train_lines else None
    MAX_QUERY_LINES = args.max_query_lines if args.max_query_lines else None
    MASK = bool(args.mask)
    NER_DEVICE = args.ner_device
    PARE_CKPT = args.pare_ckpt

    # Update derived config
    OPENNRE_ROOT = os.path.join(ROOT_DIR, "OpenNRE")
    OPENNRE_CKPT = os.path.join(ROOT_DIR, "OpenNRE", "ckpt_3", "nyt10m_pcnn_att.pth.tar")
    OPENNRE_GLOVE_WORD2ID = os.path.join(ROOT_DIR, "OpenNRE", "pretrain", "glove", "glove.6B.50d_word2id.json")
    OPENNRE_GLOVE_MAT = os.path.join(ROOT_DIR, "OpenNRE", "pretrain", "glove", "glove.6B.50d_mat.npy")

    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================

    print("=" * 60)
    print("HYDRE RETRIEVAL SYSTEM - ENGLISH (ABLOTION 2: NO PARE CONFIDENCE)")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PARE_PRETRAIN_PATH)
    start_time = time.time()

    # Load training data
    print("Loading training data...")
    en_train_data = []
    with open(FILTERED_PATH, "r") as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc="Loading training data"):
            if MAX_TRAIN_LINES is not None and i >= MAX_TRAIN_LINES:
                break
            line = line.strip()
            try:
                inst = json.loads(line)
            except json.JSONDecodeError:
                inst = ast.literal_eval(line)
            en_train_data.append(inst)

    # Load query data
    print("Loading query data...")
    query_whole = []
    with open(QUERY_PATH, "r") as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc="Loading queries"):
            if MAX_QUERY_LINES is not None and i >= MAX_QUERY_LINES:
                break
            line = line.strip()
            try:
                inst = json.loads(line)
            except json.JSONDecodeError:
                inst = ast.literal_eval(line)
            query_whole.append(inst)

    # Validate query lengths
    print("Validating query lengths...")
    for sent in query_whole:
        if len(tokenizer.encode(sent["text"])) > 500:
            print("Query too long, exiting")
            print(sent)
            exit(0)

    # ========================================================================
    # CREATE DOCUMENT BAGS
    # ========================================================================

    print("Creating document bags...")
    train_bags = {}
    done_sent = set()

    for inst in tqdm(en_train_data, desc="Creating bags"):
        bag_id = inst["h"]["id"] + "$" + inst["t"]["id"]
        if bag_id not in train_bags:
            train_bags[bag_id] = {"texts": [], "relations": [], "hs": [], "ts": []}

        if inst["text"] + bag_id not in done_sent:
            train_bags[bag_id]["texts"].append(inst["text"])
            train_bags[bag_id]["ts"].append(inst["t"])
            train_bags[bag_id]["hs"].append(inst["h"])
            done_sent.add(inst["text"] + bag_id)

        if inst["relation"] not in train_bags[bag_id]["relations"]:
            train_bags[bag_id]["relations"].append(inst["relation"])

    # ========================================================================
    # LOAD PARE PREDICTIONS
    # ========================================================================

    print("Loading PARE predictions...")

    if PARE_BACKEND.lower() == "opennre":
        if REGEN_PARE_TOP_FILE or (not os.path.exists(PARE_TOP_FILE)):
            print(f"PARE_TOP_FILE not found (or regen requested). Generating with OpenNRE: {PARE_TOP_FILE}")
            generate_top_rel_file_opennre(QUERY_PATH, PARE_TOP_FILE)

    idpredrep = [[] for i in range(len(query_whole))]
    with open(PARE_TOP_FILE, "r") as f:
        for line in f:
            d = json.loads(line)
            q_idx = int(d["entpair"][0]) // 2
            if q_idx < 0 or q_idx >= len(idpredrep):
                continue
            idpredrep[q_idx].append((d["relation"], d["score"]))

    # Sort and take top-k predictions
    for i in range(len(query_whole)):
        idpredrep[i] = [x[0] for x in sorted(idpredrep[i], key=lambda x: x[1], reverse=True)]
        idpredrep[i] = idpredrep[i][:TOPK]

    # ========================================================================
    # FILTER DOCUMENT BAGS
    # ========================================================================

    print("Filtering document bags...")
    train_bags_1 = list(train_bags.values())
    train_bags = []
    ct1 = 0
    ct2 = 0

    for bag in train_bags_1:
        if "NA" in bag["relations"]:
            if (len(bag["relations"]) == 1):
                ct1 += 1
            else:
                ct2 += 1
        if "NA" not in bag["relations"]:
            train_bags.append(bag)

    print(f"Total bags: {len(train_bags)}, NA-only bags: {ct1}, Mixed bags: {ct2}")

    # ========================================================================
    # PREPARE QUERIES
    # ========================================================================

    query_sents = [x["text"] for x in query_whole]
    train_bags_copy = copy.deepcopy(train_bags)
    query_sents_copy = copy.deepcopy(query_sents)

    # ========================================================================
    # RETRIEVAL EXECUTION
    # ========================================================================

    print("Starting retrieval process...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Execute retrieval
    top_5_sets = retrieve_top_k_sets(
        query_sents_copy,
        train_bags_copy,
        k=TOPK,
        idpredrep=idpredrep,
        rel2id=json.load(open(REL2ID_PATH))
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("Saving results...")
    expanded_top_5_sets = top_5_sets
    os.makedirs(OUT_DIR, exist_ok=True)

    output_file = os.path.join(OUT_DIR, f"not_pare_{TOPK}_bag_pare5_{os.path.basename(RETRIEVAL_MODEL_NAME)}_nonna_en_distinct_msk{MASK}_MAX_sent_nottune.json")

    with open(output_file, "w") as f_w:
        for i in range(len(query_whole)):
            f_w.write(json.dumps({"query": query_whole[i], "top_docs": expanded_top_5_sets[i]}) + "\n")

    print(f"Results saved to: {output_file}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print("=" * 60)
    print("RETRIEVAL COMPLETED SUCCESSFULLY!")
    print("=" * 60)

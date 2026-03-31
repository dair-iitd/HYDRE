import sys
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
from contextlib import contextmanager

# OpenNRE integration
import pathlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@contextmanager
def suppress_stdout_stderr():
    """
    Temporarily suppress stdout and stderr (redirect to os.devnull).
    Works for Python and C extension output.
    """
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    # Open null file
    null_fd = os.open(os.devnull, os.O_RDWR)

    try:
        # Redirect stdout and stderr to null
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(null_fd)

def seed_everything(seed=1234):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_gpu_mem_usage():
    """Get current GPU memory usage in MB"""
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

def max_min_normalize(arr: Tensor, dim) -> Tensor:
    """Normalize tensor using min-max normalization"""
    assert len(arr.shape) == 2, "Input tensor must be 2-dimensional"
    return (arr - arr.min(dim=dim, keepdim=True).values) / (arr.max(dim=dim, keepdim=True).values - arr.min(dim=dim, keepdim=True).values)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Entity masking parameters
MASK = False
MASK_TOKEN = "[UNK]"  # only for bert-base-uncased
# MASK_TOKEN = "<unk>" # only for roberta-base

# Input/Output file paths
LARGER_BAG_FILE = os.path.join(ROOT_DIR, "nyt10m", "sent_give_5_bag_pcnn5_e5-large-v2_nonna_en_distinct_mskTrue_MAX_sent_nottune.json")
OUT_DIR = os.path.join(ROOT_DIR, "nyt10m")

# PARE model configuration
PARE_PRETRAIN_PATH = 'bert-base-uncased'
PARE_CKPT = os.path.join(ROOT_DIR, "PARE", "ckpt", "bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar")
REL2ID_PATH = os.path.join(ROOT_DIR, "nyt10m", "nyt10m_rel2id.json")

# Backend configuration
PARE_BACKEND = "pare"
PARE_ROOT = os.path.join(ROOT_DIR, "PARE")

# Processing parameters
BATCH_SIZE = 16
TOPK = 5
LANG = "en"  # don't change without proper understanding

# Set random seed
seed_everything(seed=2025)

# ============================================================================
# PARE SCORE COMPUTATION
# ============================================================================

def get_pare_scores(data_file):
    """
    Compute PARE scores for relation extraction using PARE codebase
    Args:
        data_file: Path to the data file in NYT format
    Returns:
        Tensor of shape (n, num_relations) where n is number of bags
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
    if "nyt" in data_file.lower() or "nyt" in LARGER_BAG_FILE.lower():
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
        lr=2e-5, weight_decay=1e-5, opt='adamw', units=0, devices=[0]
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_obj = torch.load(PARE_CKPT, map_location=device)
    framework_.model.module.load_state_dict(ckpt_obj['state_dict'])
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
            logits = framework_.model(token, mask)
            for b in range(logits.shape[0]):
                for relid in range(logits.shape[1]):
                    rel = framework_.model.module.id2rel[relid]
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
# SINGLE SENTENCE SELECTION FROM BAGS
# ============================================================================

def best_for_each_rel(all_bag_data, rel2id):
    """
    Select the best sentence from each bag based on PARE scores
    Args:
        all_bag_data: List of bag data containing multiple sentences per bag
        rel2id: Dictionary mapping relation names to IDs
    """
    print("Creating PARE data file for sentence selection...")
    data_file = f"pare_data_choose_best_sent_from_bag_for_sent.jsonl"
    idx = 0
    
    # Create PARE data file with all sentences from all bags
    with open(data_file, "w") as f:
        for comb_data in tqdm(all_bag_data, desc="Preparing PARE data"):
            all_example_bags = comb_data["top_docs"]
            for bag_data in all_example_bags:
                for doc_tup in zip(bag_data["texts"], bag_data["hs"], bag_data["ts"]):
                    doc = {"text": doc_tup[0], "h": doc_tup[1], "t": doc_tup[2], "relation": "NA"}
                    doc["h"]["id"] = str(idx)
                    doc["t"]["id"] = str(idx+1)
                    idx += 2
                    f.write(json.dumps(doc)+"\n")
    
    print("Computing PARE scores for sentence selection...")
    # Get PARE scores for all sentences
    pare_scores = get_pare_scores(data_file)
    
    print("Selecting best sentences from each bag...")
    # Process each bag and select the best sentence
    sent_idx = 0
    for comb_data in tqdm(all_bag_data, desc="Processing bags"):
        all_example_bags = comb_data["top_docs"]
        for bag_data in all_example_bags:
            next_sent_idx = sent_idx + len(bag_data["texts"])
            pare_scores_batch = pare_scores[sent_idx:next_sent_idx]
            
            # Get useful relations for this bag
            useful_relations = [rel2id[x] for x in bag_data["relations"]]
            threshold = 0.5
            
            # Calculate scores for each sentence
            thresh_cross_score = torch.zeros(len(bag_data["texts"]))
            direct_sum_score = torch.zeros(len(bag_data["texts"]))
            
            for ur in useful_relations:
                thresh_cross_score = thresh_cross_score + (pare_scores_batch[:, ur] > threshold).float()
                direct_sum_score = direct_sum_score + pare_scores_batch[:, ur]
            
            # Find the best sentence based on threshold crossing and direct sum
            max_thresh_cross = thresh_cross_score.max()
            best_txt_idx = thresh_cross_score.argmax()
            
            for txt in range(len(thresh_cross_score)):
                if (thresh_cross_score[txt] == max_thresh_cross and 
                    direct_sum_score[txt] > direct_sum_score[best_txt_idx]):
                    best_txt_idx = txt
            
            # Keep only the best sentence
            bag_data["texts"] = [bag_data["texts"][best_txt_idx]]
            bag_data["hs"] = [bag_data["hs"][best_txt_idx]]
            bag_data["ts"] = [bag_data["ts"][best_txt_idx]]
            
            sent_idx = next_sent_idx

    # Validate the results
    print("Validating results...")
    for comb_data in all_bag_data:
        all_example_bags = comb_data["top_docs"]
        for bag_data in all_example_bags:
            assert len(bag_data["texts"]) <= len(bag_data["relations"]), \
                f"len texts {len(bag_data['texts'])} > len relations {len(bag_data['relations'])}"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HYDRE Single Example Selection (Stage 3)")
    parser.add_argument("--bag_file", type=str, required=True, help="Output from stage 2")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--rel2id_path", type=str, default=os.path.join(ROOT_DIR, "nyt10m", "nyt10m_rel2id.json"))
    parser.add_argument("--lang", type=str, default="en")
    
    args = parser.parse_args()
    
    LARGER_BAG_FILE = args.bag_file
    OUT_DIR = args.out_dir
    REL2ID_PATH = args.rel2id_path
    LANG = args.lang
    
    # Update derived config
    PARE_CKPT = os.path.join(ROOT_DIR, "HFMRE", "ckpt", "bert-base-uncased_258_16_4_2e-5_772_nyt10m_sep_na.pth.tar")
    OPENNRE_ROOT = os.path.join(ROOT_DIR, "OpenNRE")
    OPENNRE_CKPT = os.path.join(ROOT_DIR, "OpenNRE", "ckpt_3", "nyt10m_pcnn_att.pth.tar")
    OPENNRE_GLOVE_WORD2ID = os.path.join(ROOT_DIR, "OpenNRE", "pretrain", "glove", "glove.6B.50d_word2id.json")
    OPENNRE_GLOVE_MAT = os.path.join(ROOT_DIR, "OpenNRE", "pretrain", "glove", "glove.6B.50d_mat.npy")

    print("=" * 60)
    print("HYDRE SINGLE SENTENCE SELECTION - ENGLISH")
    print("=" * 60)
    
    start_time = time.time()
    
    # ========================================================================
    # LOAD RELATION MAPPING
    # ========================================================================
    
    print("Loading relation mapping...")
    rel2id = json.load(open(REL2ID_PATH))
    
    # ========================================================================
    # SETUP OUTPUT FILE
    # ========================================================================
    
    out_file = os.path.join(OUT_DIR, os.path.basename(LARGER_BAG_FILE).replace(".json", "_single_selected.json"))
    
    if os.path.exists(out_file):
        print(f"File {out_file} already exists, overwriting.")
        os.remove(out_file)
    
    # ========================================================================
    # LOAD INPUT DATA
    # ========================================================================
    
    print("Loading larger bag file...")
    larger_bags_examples_data = []
    with open(LARGER_BAG_FILE, "r") as f:
        for line in tqdm(f, desc="Loading bag data"):
            bag_data = json.loads(line)
            larger_bags_examples_data.append(bag_data)
    
    print(f"Loaded {len(larger_bags_examples_data)} bag examples")
    
    # ========================================================================
    # PROCESS BAGS - SELECT BEST SENTENCES
    # ========================================================================
    
    print("Processing bags to select best sentences...")
    best_for_each_rel(larger_bags_examples_data, rel2id)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("Saving results...")
    with open(out_file, "w") as f:
        for bag_data in tqdm(larger_bags_examples_data, desc="Writing results"):
            f.write(json.dumps(bag_data)+"\n")
    
    print(f"Results saved to: {out_file}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print("=" * 60)
    print("SINGLE SENTENCE SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)



        



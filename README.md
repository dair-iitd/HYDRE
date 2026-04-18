# HYDRE: Retrieval-based In-context Learning for Few-shot DSRE

This repository implements the HYDRE pipeline as described in the paper: (https://openreview.net/pdf?id=Jovc64vlDb).

## Pipeline Overview

HYDRE operates in 4 distinct stages:
1. **Stage 1 (Relation Prediction)**: Use the PARE model to predict candidate relations for the input queries.
2. **Stage 2 (Bag Selection)**: Retrieve the most relevant document bags from the training set using BERTScore and PARE scores.
3. **Stage 3 (Sentence Selection)**: Select the best representative sentence from each retrieved bag.
4. **Stage 4 (Inference)**: Perform few-shot prompting with an LLM (e.g., GPT-4) using the selected examples.

---

##  Setup

### 1. Prerequisites
- `uv` (Fast Python package manager)
- CUDA-enabled GPU (for Stage 1 & 2)
- OpenAI or Together API Key (for Stage 4)

### 2. Installation
```bash
source $HOME/.local/bin/env
uv sync
```

---

##  Required Data & Checkpoints

Ensure the following files are in place before running the pipeline:

### Checkpoints
- `PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar`: The pre-trained PARE model state.

You must download this checkpoint separately (it is not included in the repo). The DSRE repo links to trained checkpoints here:
https://github.com/dair-iitd/DSRE

### Datasets directory
This repo expects datasets under `datasets/` (ignored by git).

### NYT10m Data (`datasets/nyt10m/`)
Downloaded by `scripts/download_nyt10m.py`:
- `nyt10m_rel2id.json`: Relation name to ID mapping.
- `nyt10m_train.txt`: Training corpus (OpenNRE benchmark format).
- `nyt10m_val.txt`: Validation split (OpenNRE benchmark format).
- `nyt10m_test.txt`: Test split (OpenNRE benchmark format).

Optionally created by `scripts/download_nyt10m.py --make_hydre_aliases`:
- `nyt10m_train_opennre_clean_hfmre256.jsonl`: Alias (symlink/copy) to `nyt10m_train.txt`.

Not downloaded by the script (must be provided/generated separately):
- `eng_Latn_final.jsonl`: Input queries for HYDRE inference.
- `nyt10mont.txt`: Ontology descriptions for the relations.

`eng_Latn_final.jsonl` is a sampled subset of NYT10m test used in the HYDRE paper.

### Wiki20m Data (`wiki20m/`)
- `sampled_flattened_wiki20m.jsonl`: Input queries.
- `wiki20m_rel2id.json`: Relation mapping.
- `sampled_grouped_wiki20m.jsonl`: Training corpus.

---

##  Running the Pipeline (End-to-End)

### Dataset: NYT10m

#### 0. Download NYT10m benchmark files
This downloads the OpenNRE NYT10m benchmark files into `datasets/nyt10m/`.

```bash
uv run python scripts/download_nyt10m.py --make_hydre_aliases
```

#### 1. Stage 1: Candidate Generation
Predict potential relations using PARE to narrow down the retrieval space.

Stage 1 requires a PARE checkpoint file at:
`PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar`

This checkpoint is not included in the repo. Download it from the DSRE trained checkpoints link and place it at the path above (or pass `--ckpt`).

```bash
uv run python scripts/pare_pred_candidates.py \
    --query_jsonl datasets/nyt10m/eng_Latn_final.jsonl \
    --out_jsonl datasets/nyt10m/nyt_candidates.jsonl \
    --rel2id_file datasets/nyt10m/nyt10m_rel2id.json \
    --ckpt PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar \
    --device cuda:0 \
    --batch_size 64
```

Notes:
- Stage 1 is primarily **GPU-bound** (BERT forward). Increasing `--batch_size` is the main speed lever.
- CPU parallelism mainly affects tokenization/IO; you can increase CPU threads via `OMP_NUM_THREADS` / `MKL_NUM_THREADS` if needed.

#### 2. Stage 2: Bag Selection
Retrieve relevant bags from the training set.
```bash
uv run python scripts/ret_give_bag_eng.py \
    --train_path datasets/nyt10m/nyt10m_train_opennre_clean_hfmre256.jsonl \
    --query_path datasets/nyt10m/eng_Latn_final.jsonl \
    --out_dir datasets/nyt10m \
    --pare_top_file datasets/nyt10m/nyt_candidates.jsonl \
    --rel2id_path datasets/nyt10m/nyt10m_rel2id.json \
    --no-mask \
    --num_workers 32 \
    --retrieval_batch_size 200
```

Notes:
- **CPU parallelism**: `--num_workers` controls the multiprocessing pool used for top-k selection.
- **GPU usage**:
  - Retrieval embeddings use `sentence-transformers` (will use CUDA if available).
  - If you enable masking (`--mask`), NER runs via HF `pipeline('ner')` and you can force CPU with `--ner_device -1`.

#### 3. Stage 3: Single Sentence Selection
Pick the best example from each selected bag.
```bash
# Filename is generated based on Stage 2 parameters
uv run python scripts/ret_single_from_bag_eng.py \
    --bag_file datasets/nyt10m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskFalse_MAX_sent_nottune.json \
    --out_dir datasets/nyt10m/ \
    --rel2id_path datasets/nyt10m/nyt10m_rel2id.json \
    --pare_ckpt PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar
```

#### 4. Stage 4: LLM Inference
Run few-shot prompting via the OpenAI/Together API.
```bash
# Set your API keys in a .env file (OPENAI_API_KEY or TOGETHER_API_KEY)
uv run python scripts/infer_api.py \
    --example_path datasets/nyt10m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskFalse_MAX_sent_nottune_single_selected.json \
    --source_path datasets/nyt10m/eng_Latn_final.jsonl \
    --model gpt-4-turbo
```

---

### Dataset: Wiki20m

Simply swap the paths in each command:
1. **Stage 1**: `--query_jsonl wiki20m/sampled_flattened_wiki20m.jsonl --out_jsonl wiki20m/wiki_candidates.jsonl --rel2id_file wiki20m/wiki20m_rel2id.json`
2. **Stage 2**: `--train_path wiki20m/sampled_grouped_wiki20m.jsonl --query_path wiki20m/sampled_flattened_wiki20m.jsonl --out_dir wiki20m/ --pare_top_file wiki20m/wiki_candidates.jsonl --rel2id_path wiki20m/wiki20m_rel2id.json`
3. **Stage 3**: `--bag_file wiki20m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskTrue_MAX_sent_nottune.json --out_dir wiki20m/ --rel2id_path wiki20m/wiki20m_rel2id.json`
4. **Stage 4**: `--example_path wiki20m/..._single_selected.json --source_path wiki20m/sampled_flattened_wiki20m.jsonl`

##  Notes
- **Stage 1 Backend**: This implementation uses the standalone `PARE` codebase for candidate generation.
- **Wiki20m Checkpoint**: Stage 1 for Wiki20m requires a checkpoint that matches the Wiki20m `rel2id` count (80 relations). Using the NYT checkpoint for Wiki will result in a size mismatch error.
- **API Keys**: Ensure you have a `.env` file with `OPENAI_API_KEY` or `TOGETHER_API_KEY`.

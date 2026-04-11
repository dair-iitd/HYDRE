# HYDRE: Retrieval-based In-context Learning for Few-shot RE

This repository implements the HYDRE pipeline as described in the paper: [Finding the Gold in the Sand: Retrieval-based In-context Learning for Few-shot Distantly Supervised Relation Extraction](https://openreview.net/pdf?id=Jovc64vlDb).

## 🚀 Pipeline Overview

HYDRE operates in 4 distinct stages:
1. **Stage 1 (Relation Prediction)**: Use the PARE model to predict candidate relations for the input queries.
2. **Stage 2 (Bag Selection)**: Retrieve the most relevant document bags from the training set using BERTScore and PARE scores.
3. **Stage 3 (Sentence Selection)**: Select the best representative sentence from each retrieved bag.
4. **Stage 4 (Inference)**: Perform few-shot prompting with an LLM (e.g., GPT-4) using the selected examples.

> **Testing Status**: The NYT10m pipeline (Stages 1-3) has been end-to-end verified on both CUDA and Mac (CPU) environments. Stage 4 uses the OpenAI API and is configured to run out-of-the-box.

---

## 🛠 Setup

### 1. Prerequisites
- `uv` (Fast Python package manager)
- CUDA-enabled GPU (for Stage 1 & 2)
- OpenAI or Together API Key (for Stage 4)

### 2. Installation
```bash
uv sync
```

---

## 📂 Required Data & Checkpoints

Ensure the following files are in place before running the pipeline:

### Checkpoints
- `PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar`: The pre-trained PARE model state.

### NYT10m Data (`nyt10m/`)
- `eng_Latn_final.jsonl`: Input queries for RE.
- `nyt10m_rel2id.json`: Relation name to ID mapping.
- `nyt10m_train_opennre_clean_hfmre256.jsonl`: Training corpus for retrieval.
- `nyt10mont.txt`: Ontology descriptions for the relations.

### Wiki20m Data (`wiki20m/`)
- `sampled_flattened_wiki20m.jsonl`: Input queries.
- `wiki20m_rel2id.json`: Relation mapping.
- `sampled_grouped_wiki20m.jsonl`: Training corpus.

---

## 📝 Running the Pipeline (End-to-End)

### Dataset: NYT10m

#### 1. Stage 1: Candidate Generation
Predict potential relations using PARE to narrow down the retrieval space.
```bash
python scripts/pare_pred_candidates.py \
    --query_jsonl nyt10m/eng_Latn_final.jsonl \
    --out_jsonl nyt10m/nyt_candidates.jsonl \
    --rel2id_file nyt10m/nyt10m_rel2id.json \
    --ckpt PARE/ckpt/bert-base-uncased_512_32_5_1e-5_772_nyt10m_sep_na.pth.tar
```

#### 2. Stage 2: Bag Selection
Retrieve relevant bags from the training set.
```bash
python scripts/ret_give_bag_eng.py \
    --train_path nyt10m/nyt10m_train_opennre_clean_hfmre256.jsonl \
    --query_path nyt10m/eng_Latn_final.jsonl \
    --out_dir nyt10m \
    --pare_top_file nyt10m/nyt_candidates.jsonl \
    --rel2id_path nyt10m/nyt10m_rel2id.json
```

#### 3. Stage 3: Single Sentence Selection
Pick the best example from each selected bag.
```bash
# Filename is generated based on Stage 2 parameters
python scripts/ret_single_from_bag_eng.py \
    --bag_file nyt10m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskTrue_MAX_sent_nottune.json \
    --out_dir nyt10m/ \
    --rel2id_path nyt10m/nyt10m_rel2id.json
```

#### 4. Stage 4: LLM Inference
Run few-shot prompting via the OpenAI/Together API.
```bash
# Set your API keys in a .env file (OPENAI_API_KEY or TOGETHER_API_KEY)
python scripts/infer_api.py \
    --example_path nyt10m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskTrue_MAX_sent_nottune_single_selected.json \
    --source_path nyt10m/eng_Latn_final.jsonl \
    --model gpt-4-turbo
```

---

### Dataset: Wiki20m

Simply swap the paths in each command:
1. **Stage 1**: `--query_jsonl wiki20m/sampled_flattened_wiki20m.jsonl --out_jsonl wiki20m/wiki_candidates.jsonl --rel2id_file wiki20m/wiki20m_rel2id.json`
2. **Stage 2**: `--train_path wiki20m/sampled_grouped_wiki20m.jsonl --query_path wiki20m/sampled_flattened_wiki20m.jsonl --out_dir wiki20m/ --pare_top_file wiki20m/wiki_candidates.jsonl --rel2id_path wiki20m/wiki20m_rel2id.json`
3. **Stage 3**: `--bag_file wiki20m/sent_give_5_bag_pare5_e5-large-v2_nonna_en_distinct_mskTrue_MAX_sent_nottune.json --out_dir wiki20m/ --rel2id_path wiki20m/wiki20m_rel2id.json`
4. **Stage 4**: `--example_path wiki20m/..._single_selected.json --source_path wiki20m/sampled_flattened_wiki20m.jsonl`

## ⚠️ Notes
- **Stage 1 Backend**: This implementation uses the standalone `PARE` codebase for candidate generation.
- **Mac/CPU Compatibility**: The framework has been patched to support non-CUDA devices (CPU/MPS), though CPU is recommended for stability.
- **Wiki20m Checkpoint**: Stage 1 for Wiki20m requires a checkpoint that matches the Wiki20m `rel2id` count (80 relations). Using the NYT checkpoint for Wiki will result in a size mismatch error.
- **API Keys**: Ensure you have a `.env` file with `OPENAI_API_KEY` or `TOGETHER_API_KEY`.

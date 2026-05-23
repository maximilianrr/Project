# NanoChat Medical — Training Pipeline

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo>
cd Project
python -m venv venv
```

### 2. Activate venv

**Windows:**
```bash
venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Set PYTHONPATH (required on Windows, may be needed on Linux too)

**Windows:**
```bash
set PYTHONPATH=src
```

**Mac/Linux:**
```bash
export PYTHONPATH=src
```

---

## Full Pipeline — Run in This Order

### Step 1 — Download all data

```bash
python src/chat_model/datasets/download.py
```

For a quick test run with caps:
```bash
python src/chat_model/datasets/download.py --max-documents 50000 --max-abstracts 30000
```

Downloads:
- **Stage 1:** `karpathy/climbmix-400b-shuffle` (raw general English text) + `OpenAssistant/oasst2` (Q&A pairs)
- **Stage 2:** MedDialog, MedQuAD, PubMed abstracts, WTND book PDF

Output directories:
```
data/climbmix_parquet/      ← climbmix raw text shards
data/pretrain_parquet/      ← oasst2 Q/A shards
data/pubmed_parquet/        ← PubMed abstract shards
data/raw/MedQuAD/           ← MedQuAD XML files
data/raw/meddialog_en/      ← MedDialog HuggingFace dataset
data/raw/where_there_is_no_doctor.pdf
data/raw/where_there_is_no_doctor_clean.txt
```

---

### Step 2 — Preprocess Stage 2 fine-tuning data

```bash
python src/chat_model/datasets/preprocess.py --stage finetune
```

Cleans and filters MedDialog, MedQuAD, safety examples. Applies tone normalisation, epistemic hedging, unsafe content filtering, and fuzzy deduplication.

Output:
```
data/splits/train.jsonl     ← ~92,000 examples
data/splits/val.jsonl       ← ~10,800 examples
data/splits/test.jsonl      ← ~5,400 examples
```

**Run this before Step 3** — the tokenizer corpus includes the medical splits.

---

### Step 3 — Build tokenizer corpus

```bash
python src/chat_model/tokenizing/prepare_data.py
```

Assembles `data/tokenizer_text.txt` from all sources in order:
1. climbmix raw documents (200 MB cap)
2. oasst2 Q&A pairs (100 MB cap)
3. PubMed abstracts (100 MB cap) — for medical vocabulary coverage
4. Medical fine-tune splits (train + val)
5. WTND book text

Output:
```
data/tokenizer_text.txt     ← ~300 MB corpus
```

---

### Step 4 — Train the tokenizer

```bash
python src/chat_model/tokenizing/train_tokenizer.py
```

Trains a BPE tokenizer with vocab size 16,384 on the corpus from Step 3. Takes ~25 seconds on CPU.

Output:
```
data/processed/tokenized/tokenizer.pkl
```

---

### Step 5 — Preprocess Stage 1 pretraining data

```bash
python src/chat_model/datasets/preprocess.py --stage pretrain
```

Saves climbmix documents as a plain text file and oasst2 as JSONL splits.

Output:
```
data/pretrain_splits/climbmix_docs.txt   ← 50,000 raw documents
data/pretrain_splits/train.jsonl         ← ~20,400 oasst2 conversations
data/pretrain_splits/val.jsonl           ← ~2,400
data/pretrain_splits/test.jsonl          ← ~1,200
```

---

### Step 6 — Train

```bash
python scripts/train.py
```

Checkpoints are saved to:
```
data/checkpoints/pre_trained/    ← Stage 1 (best_pretrained.pth, latest.pth)
data/checkpoints/                ← Stage 2 (best_model.pth, latest.pth)
```

---

## Architecture & Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dim | 512 |
| Layers | 8 |
| Heads | 8 |
| Block size | 1024 tokens |
| Dropout | 0.22 |
| Vocab size | 16,384 |

| | Stage 1 | Stage 2 |
|---|---|---|
| Batch size | 32 | 16 |
| Learning rate | 3e-4 | 1e-5 |
| Epochs | 10 | 10 |
| Warmup steps | 2000 | 200 |

---

## Dataset Strategy

**Stage 1 — Pretraining (English only, no medical bias)**

| Dataset | Type | Class |
|---|---|---|
| `karpathy/climbmix-400b-shuffle` | Raw general English text | `ChunkTextDataset` |
| `OpenAssistant/oasst2` | English Q&A conversations | `ChunkChatDataset` (loss_masking=False) |

**Stage 2 — Medical fine-tuning**

| Dataset | Type | Class |
|---|---|---|
| MedDialog (ChatDoctor-HealthCareMagic-100k) | Patient/doctor conversations | `ChunkChatDataset` (loss_masking=True) |
| MedQuAD | NIH factual medical Q&A | `ChunkChatDataset` (loss_masking=True) |
| PubMed abstracts | Medical scientific text | Tokenizer vocab only |
| WTND book | Plain-language medical prose | Tokenizer vocab only |

MedDialog is upsampled 3x so real conversation dominates the fine-tuning signal.

---

## Data Directory Layout

```
data/
├── climbmix_parquet/          Stage 1 raw text shards
├── pretrain_parquet/          Stage 1 oasst2 shards
├── pubmed_parquet/            Stage 2 PubMed shards
├── raw/
│   ├── MedQuAD/
│   ├── meddialog_en/
│   ├── where_there_is_no_doctor.pdf
│   └── where_there_is_no_doctor_clean.txt
├── splits/                    Stage 2 fine-tune JSONL splits
├── pretrain_splits/           Stage 1 JSONL splits + climbmix_docs.txt
├── tokenizer_text.txt         Tokenizer training corpus
├── processed/tokenized/
│   └── tokenizer.pkl          Trained BPE tokenizer
└── checkpoints/
    ├── pre_trained/            Stage 1 checkpoints
    └── best_model.pth          Stage 2 best model
```

---

## Key Files

```
src/chat_model/
├── config.py                  All paths and hyperparameters
├── datasets/
│   ├── chunk_dataset.py       ChunkTextDataset + ChunkChatDataset
│   ├── download.py            Downloads all raw data
│   ├── preprocess.py          Cleans and splits all data
│   └── loader.py              Builds datasets and dataloaders
├── tokenizing/
│   ├── prepare_data.py        Builds tokenizer corpus
│   └── train_tokenizer.py     Trains BPE tokenizer
├── training/
│   └── train.py               Full two-stage training loop
└── model/
    └── model.py               NanoChat transformer

scripts/
└── train.py                   Entry point — run this
```

---

## Notes

- Always set `PYTHONPATH=src` before running any script
- Run steps in order — each step depends on the output of the previous
- `--max-documents` and `--max-abstracts` flags are useful for testing the pipeline end-to-end before a full run
- The tokenizer is shared across both stages — it must be trained once before any training begins
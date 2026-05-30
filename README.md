# NanoChat Medical - 3-Stage Training Pipeline

This project uses one shared tokenizer and three sequential model-training stages:

1. Stage 1: general text pretraining on climbmix.
2. Stage 2: continued medical-text pretraining on PubMed plus a small climbmix mix-in.
3. Stage 3: chatbot fine-tuning on oasst2, MedDialog, MedQuAD, emergency cases, and safety examples.

The tokenizer is trained once before model training. Do not change `VOCAB_SIZE`, `N_EMB`, `N_LAYER`, `N_HEAD`, or tokenizer files after Stage 1 starts unless restarting from scratch.

---

## Setup

Create and activate the environment:

```bash
cd Project
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
venv\Scripts\activate
```

Install the package:

```bash
pip install -e .
```

Set `PYTHONPATH` if needed:

```bash
export PYTHONPATH=src
```

On Windows:

```bat
set PYTHONPATH=src
```

---

## Step 1 - Download Data

Full run:

```bash
python src/chat_model/datasets/download.py
```

Quick run:

```bash
python src/chat_model/datasets/download.py --max-documents 50000 --max-pairs 30000 --max-abstracts 30000
```

Downloads:

| Stage | Data | Output |
|---|---|---|
| Stage 1 | `karpathy/climbmix-400b-shuffle` | `data/climbmix_parquet/` |
| Stage 2 | PubMed abstracts | `data/pubmed_parquet/` |
| Stage 3 | OpenAssistant/oasst2 | `data/pretrain_parquet/` |
| Stage 3 | MedDialog | `data/raw/meddialog_en/` |
| Stage 3 | MedQuAD | `data/raw/MedQuAD/` |
| Tokenizer/Stage 3 | Where There Is No Doctor | `data/raw/where_there_is_no_doctor_clean.txt` |

Optional emergency training cases should be placed at:

```text
data/raw/emergency_training_cases.json
```

---

## Step 2 - Preprocess Stage 1

```bash
python src/chat_model/datasets/preprocess.py --stage 1
```

Creates the raw-text pretraining file:

```text
data/pretrain_splits/climbmix_docs.txt
```

This stage trains with `ChunkTextDataset` and predicts every token.

---

## Step 3 - Preprocess Stage 2

```bash
python src/chat_model/datasets/preprocess.py --stage 2
```

Creates the continued-pretraining medical text file:

```text
data/stage2_splits/stage2_docs.txt
```

This contains PubMed text plus a small climbmix sample controlled by:

```python
STAGE2_CLIMBMIX_RATIO = 0.10
```

This stage also uses `ChunkTextDataset` and predicts every token.

---

## Step 4 - Preprocess Stage 3

```bash
python src/chat_model/datasets/preprocess.py --stage 3
```

Creates chatbot fine-tuning splits:

```text
data/stage3_splits/train.jsonl
data/stage3_splits/val.jsonl
data/stage3_splits/test.jsonl
```

Stage 3 applies cleaning, tone normalization, hedging, unsafe-content filtering, safety examples, and deduplication. It trains with `ChunkChatDataset` and assistant-token loss masking.

You can preprocess all stages at once:

```bash
python src/chat_model/datasets/preprocess.py --stage all
```

---

## Step 5 - Build Tokenizer Corpus

```bash
python src/chat_model/tokenizing/prepare_data.py
```

For a larger tokenizer corpus:

```bash
python src/chat_model/tokenizing/prepare_data.py --climbmix-char-cap 500000000 --oasst2-char-cap 100000000 --pubmed-char-cap 300000000
```

Builds:

```text
data/tokenizer_text.txt
```

Source order:

1. climbmix raw documents for Stage 1 general text.
2. oasst2 Q/A pairs for Stage 3 conversational coverage.
3. PubMed abstracts for Stage 2 medical/scientific vocabulary.
4. Stage 3 train/val splits for medical chat language.
5. WTND book text for plain-language medical prose.

---

## Step 6 - Train Tokenizer

```bash
# Switch to (nanochat) venv first
cd ..\nanochat
.venv\Scripts\activate
cd ..\Project
python src\chat_model\tokenizing\train_tokenizer.py
```

Outputs:

```text
data/processed/tokenized/tokenizer.pkl
```

The tokenizer may also be saved to nanochat's cache directory for compatibility.

---
```

Checkpoint flow:

| Stage | Loads | Saves |
|---|---|---|
| Stage 1 | scratch | `data/checkpoints/pre_trained/best_pretrained.pth` |
| Stage 2 | Stage 1 best | `data/checkpoints/stage2_checkpoint/best_stage2.pth` |
| Stage 3 | Stage 2 best | `data/checkpoints/best_model.pth` |

---

## Data Layout

```text
data/
  climbmix_parquet/              Stage 1 raw climbmix shards
  pubmed_parquet/                Stage 2 PubMed shards
  pretrain_parquet/              Stage 3 oasst2 shards
  raw/
    MedQuAD/
    meddialog_en/
    where_there_is_no_doctor.pdf
    where_there_is_no_doctor_clean.txt
    emergency_training_cases.json
  pretrain_splits/
    climbmix_docs.txt
  stage2_splits/
    stage2_docs.txt
  stage3_splits/
    train.jsonl
    val.jsonl
    test.jsonl
  tokenizer_text.txt
  processed/tokenized/
    tokenizer.pkl
  checkpoints/
    pre_trained/
      best_pretrained.pth
    stage2_checkpoint/
      best_stage2.pth
    stage3_checkpoint/
    best_model.pth
```

### Step 7 — Train
```bash
python src\chat_model\training\train.py
```
---
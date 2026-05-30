# NanoChat Medical - 3-Stage Training Pipeline

One shared tokenizer, three sequential training stages:

1. **Stage 1** — general text pretraining on ClimbMix
2. **Stage 2** — continued medical pretraining on PubMed + a small ClimbMix mix-in
3. **Stage 3** — chatbot fine-tuning on oasst2, MedDialog, MedQuAD, emergency cases and safety examples

The tokenizer is trained once before model training. Don't change `VOCAB_SIZE`, `N_EMB`, `N_LAYER`, `N_HEAD` or tokenizer files after Stage 1 starts unless restarting from scratch.

---

## Quickstart

```bash
# 1. Clone and enter repo
git clone https://github.com/maximilianrr/Project.git
cd Project

# 2. Virtual environment
python3.12 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone NanoChat alongside this repo
cd ..
git clone https://github.com/karpathy/nanochat.git
cd nanochat && uv venv && uv sync --extra cpu
```

Set `PYTHONPATH` if needed:

```bash
export PYTHONPATH=src       # Linux/macOS
set PYTHONPATH=src          # Windows
```

---

## Step 1 — Download Data

```bash
python src/chat_model/datasets/download.py
```

Or with limits:

```bash
python src/chat_model/datasets/download.py --max-documents 50000 --max-pairs 30000 --max-abstracts 30000
```

Optional emergency training cases go here:

```
data/raw/emergency_training_cases.json
```

---

## Step 2 — Preprocess Stage 1

```bash
python src/chat_model/datasets/preprocess.py --stage 1
```

Produces:

```
data/pretrain_splits/climbmix_docs.txt
```

---

## Step 3 — Preprocess Stage 2

```bash
python src/chat_model/datasets/preprocess.py --stage 2
```

Produces:

```
data/stage2_splits/stage2_docs.txt
```

---

## Step 4 — Preprocess Stage 3

```bash
python src/chat_model/datasets/preprocess.py --stage 3
```

Produces:

```
data/stage3_splits/train.jsonl
data/stage3_splits/val.jsonl
data/stage3_splits/test.jsonl
```

Stage 3 applies cleaning, tone normalization, hedging, unsafe-content filtering, safety examples, and deduplication. Trains with `ChunkChatDataset` and assistant-token loss masking.

All stages at once:

```bash
python src/chat_model/datasets/preprocess.py --stage all
```

---

## Step 5 — Build Tokenizer Corpus

```bash
python src/chat_model/tokenizing/prepare_data.py
```

Or with larger caps:

```bash
python src/chat_model/tokenizing/prepare_data.py --climbmix-char-cap 500000000 --oasst2-char-cap 100000000 --pubmed-char-cap 300000000
```

Produces:

```
data/tokenizer_text.txt
```

Source order: ClimbMix → oasst2 → PubMed → Stage 3 splits → WTND book text.

---

## Step 6 — Train Tokenizer

Switch to the `nanochat` venv first:

```bash
cd ..\nanochat
.venv\Scripts\activate
cd ..\Project
python src\chat_model\tokenizing\train_tokenizer.py
```

Produces:

```
data/processed/tokenized/tokenizer.pkl
```

---

## Step 7 — Train

```bash
python src\chat_model\training\train.py
```

---

## Data Layout

```
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
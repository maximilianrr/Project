# NanoChat — Pipeline Setup & Run Order

## Environment Setup

Two virtual environments are needed — one for the project, one for nanochat.

### Project venv
```bash
cd D:\2B_proj\Project
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src
```

### Nanochat venv (for tokenizer training only)
```bash
cd D:\2B_proj\nanochat
uv venv
.venv\Scripts\activate
uv sync --extra cpu
cd ../Project
```

> **Important:** Always activate the correct venv before each step.
> Use `(venv)` for everything except tokenizer training, which needs `(nanochat)`.

---

## Pipeline Run Order

### Step 1 — Download data
```bash
# (venv) active
python src\chat_model\datasets\download.py --max-abstracts 50000
```
Downloads oasst2, PubMed (50k abstracts), MedDialog, MedQuAD, and WTND.

---

### Step 2 — Preprocess medical fine-tuning data
```bash
python -m src.chat_model.datasets.preprocess
```
Produces `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`.

---

### Step 3 — Build tokenizer corpus
```bash
python -m src.chat_model.tokenizing.prepare_data --pubmed-char-cap 25000000
```
Writes `data/tokenizer_text.txt` (~149 MB, 293k lines).

---

### Step 4 — Train tokenizer
```bash
# Switch to (nanochat) venv first
.venv\Scripts\activate   # from nanochat dir, or use full path
cd ../Project
python src\chat_model\tokenizing\train_tokenizer.py
```
Trains a 16,384-vocab BPE tokenizer. Saves to both the project and nanochat cache.
Switch back to `(venv)` after this step.

---

### Step 5 — Preprocess pretrain splits
```bash
# (venv) active
python -m src.chat_model.datasets.preprocess --stage pretrain
```
Produces `data/pretrain_splits/train.jsonl`, `val.jsonl`, `test.jsonl` (24k oasst2 pairs).

---

### Step 6 — Convert all splits to parquet
```bash
python src\chat_model\tokenizing\convert_to_parquet.py
```
Expected output:
- Tokenizer shards: ~525k rows, ~148 MB
- Pretrain shards: ~24k rows, ~19 MB
- Fine-tune shards: ~108k rows, ~104 MB

---

### Step 7 — Train
```bash
python src\chat_model\training\train.py
```

---

## Notes

- MEDIQA-Chat is unavailable on the Hub — it is skipped automatically, this is expected.
- `set PYTHONPATH=src` must be set in every new terminal session, or the project imports will fail with `ModuleNotFoundError: No module named 'chat_model'`.
- Running any step in the wrong venv will cause import errors (`fitz`, `torch`, etc.).
- Steps 1 and 2 only need to be run once. Steps 5 and 6 must both be run before training.
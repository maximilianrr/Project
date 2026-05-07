# MedChat — Evaluation

This branch contains the evaluation suite for MedChat: perplexity, BLEU/ROUGE, and safety checks.

---

## Folder structure

```
parent_folder/
├── Project/          ← this repo
│   ├── chat_model/   ← model, data loader, config, scripts
│   ├── eval.py       ← evaluation logic
│   ├── run_eval.py   ← entry point
│   └── safety_cases.json
└── nanochat/         ← external dependency (see step 2)
```

---

## Setup

### 1. Create conda environment or use venv

```bash
conda create -n medchat python=3.12
conda activate medchat
conda install -c conda-forge datasets pyarrow
conda install pytorch -c pytorch
pip install -r requirements_eval.txt
```

### 2. Clone nanochat

nanochat must sit **alongside** this repo (one level up):

```bash
cd ..
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv venv
uv sync
uv pip install torch torchvision tokenizers pyarrow
```

> **Intel Mac users:** `uv sync --extra cpu` will fail. Use `uv sync` followed by `uv pip install torch torchvision` instead.

---

## Prepare data

Run all commands from inside the `chat_model/` folder with the **medchat** conda environment active.

```bash
conda activate medchat
cd chat_model
```

### 3. Download raw data

```bash
python scripts/download_data.py
```

Downloads MedQuAD (~16k QA pairs) and MedDialog (~112k conversations) into `chat_model/data/raw/`.

### 4. Preprocess into splits

```bash
python scripts/preprocess.py
```

Produces:
```
chat_model/data/splits/
├── train.jsonl   — 109,277 examples
├── val.jsonl     — 12,856 examples
└── test.jsonl    — 6,429 examples
```

---

## Train the tokenizer

Switch to the **nanochat** venv and run from the `chat_model/` folder:

```bash
source ../../nanochat/.venv/bin/activate
python scripts/prepare_tokenizer_data.py
python scripts/convert_to_parquet.py
python scripts/train_tokenizer.py
```

The tokenizer is saved to `~/.cache/nanochat/tokenizer/`.

---

## Run evaluation

Switch back to the **medchat** conda environment and run from the **repo root**:

```bash
conda activate medchat
cd ..          # back to Project/ root
python run_eval.py
```

### Using a trained checkpoint

Open `run_eval.py` and set the `CHECKPOINT` variable at the top:

```python
CHECKPOINT = "path/to/checkpoint.pt"
```

Then run as above.

---

## What the eval measures

| Metric | Description |
|---|---|
| `perplexity` | Cross-entropy loss on the val set (lower = better) |
| `bleu` | BLEU score vs. MedQuAD reference answers |
| `rouge1_f` / `rouge2_f` / `rougeL_f` | ROUGE F1 scores |
| `safety_escalation_rate` | Fraction of high-risk prompts that produced an emergency referral |
| `safety_unsafe_rate` | Fraction of responses matching unsafe patterns |
| `safety_missing_escalation` | IDs of high-risk cases that failed to escalate |
| `safety_flagged` | IDs of responses that matched unsafe patterns |

The safety suite (`safety_cases.json`) contains 15 high-risk and 15 low-risk prompts.

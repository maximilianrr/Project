## Project Structure

```
Project/
├── chat_model/
│   ├── models/
│   │   └── language_model.py
│   └── utils/
│       ├── data_loader.py
│       └── train.py
├── scripts/
│   ├── download_data.py          ← downloads all 4 sources
│   ├── download_wtnd.py          ← downloads + cleans WTND PDF
│   ├── preprocess.py             ← cleans, filters, upsamples, splits
│   ├── prepare_tokenizer_data.py ← writes tokenizer_text.txt
│   ├── convert_to_parquet.py     ← shards text into parquet
│   └── train_tokenizer.py        ← trains nanochat BPE tokenizer
├── config.py
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.12
- Git
- [uv](https://github.com/astral-sh/uv) — `pip install uv`

---

## Setup

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

---

## Run Order

```bash
# From Project folder, venv active:
python scripts/download_data.py        # download all sources
python scripts/download_wtnd.py        # download + clean WTND PDF
python scripts/preprocess.py           # clean, filter, upsample, split
python scripts/prepare_tokenizer_data.py
python scripts/convert_to_parquet.py

# Switch to nanochat folder + venv:
cd ../nanochat
source .venv/bin/activate
python ../Project/scripts/train_tokenizer.py
```

---

## Data Sources

| Source | Raw size | Role |
|--------|----------|------|
| [ChatDoctor/MedDialog](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 112k | **Gold source** — real patient messages + doctor replies. Upsampled 3x. |
| [MedQuAD](https://github.com/abachaa/MedQuAD) | 16k | NIH medical Q&A |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | 211k | Biomedical Q&A — long answers only, tight-filtered |
| [MEDIQA-Chat](https://huggingface.co/datasets/chiusers/mediqa-chat-2023) | 67k | Clinical dialogues (optional) |
| [Where There Is No Doctor](https://hesperian.org) | 503 pages | Tokenizer vocabulary only |

**Not used:** MedMCQA — exam language actively hurts conversational tone.

---

## Why these settings for NanoChat

NanoChat is a small model. The bottleneck is not data volume — it is data quality and consistency.

| Setting | Value | Why |
|---------|-------|-----|
| `MIN_A_CHARS` | 80 | Shorter answers are too vague to teach patterns |
| `MAX_A_CHARS` | 800 | Nano model can't reliably reproduce longer outputs |
| `MAX_Q_CHARS` | 500 | Patients don't write essays |
| `MEDDIALOG_UPSAMPLE` | 3 | Real conversation dominates the training signal |
| `MAX_EXAMPLES` | 100,000 | Quality cap — 350k noisy > 100k clean for small models |
| `VOCAB_SIZE` | 16,384 | Frees model capacity — all needed medical vocab fits here |

---

## Tone normalisation

Every answer passes through a rewrite pipeline before saving:

| Raw (what datasets say) | After normalisation |
|-------------------------|---------------------|
| "The patient should rest." | "You should rest." |
| "It is recommended that..." | "I'd recommend that..." |
| "Presents with fever." | "Has symptoms of fever." |
| "Contraindicated in renal failure." | "Not recommended in renal failure." |
| "Etiology is unknown." | "The cause is unknown." |

This gives every source the same doctor voice so the model learns one consistent style.
# config.py
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "nanochat")

DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
SPLITS_DIR  = os.path.join(DATA_DIR, "splits")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")

# Raw source paths
WTND_PDF      = os.path.join(RAW_DIR, "where_there_is_no_doctor.pdf")
WTND_CLEAN    = os.path.join(RAW_DIR, "where_there_is_no_doctor_clean.txt")
MEDQUAD_DIR   = os.path.join(RAW_DIR, "MedQuAD")
MEDDIALOG_DIR = os.path.join(RAW_DIR, "meddialog_en")
PUBMEDQA_DIR  = os.path.join(RAW_DIR, "pubmedqa")
MEDIQA_DIR    = os.path.join(RAW_DIR, "mediqa_chat")

# ── Data settings ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
RANDOM_SEED = 42

# Quality filters — balanced for nano model
MIN_Q_CHARS = 20
MIN_A_CHARS = 80
MAX_Q_CHARS = 800   # up from 500
MAX_A_CHARS = 1500  # up from 800 — doctors naturally write more than 2 sentences
MAX_EXAMPLES = 150_000  # up from 100k — more variety without noise

# MedDialog upsampling
MEDDIALOG_UPSAMPLE = 3

# ── Tokenizer settings ─────────────────────────────────────────────────────────
VOCAB_SIZE = 16_384
MAX_CHARS  = 200_000_000
DOC_CAP    = 10_000
SHARD_SIZE = 100_000

# ── Model hyperparameters ──────────────────────────────────────────────────────
BATCH_SIZE    = 16
LEARNING_RATE = 1e-5
EPOCHS        = 10
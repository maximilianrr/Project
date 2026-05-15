# config.py
"""Configuration module for the Chat Model.

Manages all configuration settings including:
- File paths and directories
- Model hyperparameters
- Training settings
- Tokenizer configuration
"""

import os
import torch

# Paths - all relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANOCHAT_DIR = os.path.join(PROJECT_ROOT, "..", "nanochat")
NANOCHAT_PACKAGE_DIR = os.path.join(NANOCHAT_DIR, "nanochat")

# Data directories
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SPLITS_DIR      = os.path.join(PROCESSED_DIR, "splits")
PARQUET_DIR     = os.path.join(PROCESSED_DIR, "parquet")
TOKENIZED_DIR   = os.path.join(PROCESSED_DIR, "tokenized")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")

# Modal (cloud training) paths
MODAL_OUTPUT_DIR      = "/root/output"
MODAL_CHECKPOINTS_DIR = os.path.join(MODAL_OUTPUT_DIR, "checkpoints")

# ── Two-stage training paths ───────────────────────────────────────────────────
# Stage 1 — general conversational pretraining (oasst2)
PRETRAIN_RAW_DIR     = os.path.join(RAW_DIR, "oasst2")
PRETRAIN_PARQUET_DIR = os.path.join(PROCESSED_DIR, "pretrain_parquet")
PRETRAIN_CHECKPOINT  = os.path.join(CHECKPOINTS_DIR, "stage1_checkpoint")

# Stage 2 — medical fine-tuning parquet
FINETUNE_PARQUET_DIR = os.path.join(PROCESSED_DIR, "finetune_parquet")

# File-specific paths
TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")
TOKENIZER_PKL  = os.path.join(TOKENIZED_DIR, "tokenizer.pkl")
BEST_MODEL_PTH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
WTND_PDF       = os.path.join(RAW_DIR, "where_there_is_no_doctor.pdf")
WTND_CLEAN     = os.path.join(RAW_DIR, "where_there_is_no_doctor_clean.txt")
MEDQUAD_DIR    = os.path.join(RAW_DIR, "MedQuAD")
MEDDIALOG_DIR  = os.path.join(RAW_DIR, "meddialog_en")
MEDIQA_DIR     = os.path.join(RAW_DIR, "mediqa_chat")

# ── Data settings ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
RANDOM_SEED = 42

# Quality filters — tuned for nano model
MIN_Q_CHARS = 20
MIN_A_CHARS = 80
MAX_Q_CHARS = 800
MAX_A_CHARS = 1500
MAX_EXAMPLES = 150_000

# MedDialog upsampling — real conversations dominate training signal
MEDDIALOG_UPSAMPLE = 3

# ── Tokenizer settings ─────────────────────────────────────────────────────────
VOCAB_SIZE = 16_384          # reduced from 32768 — frees capacity for reasoning
MAX_CHARS  = 500_000_000
DOC_CAP    = 10_000
SHARD_SIZE = 100_000

# Max chars pulled from oasst2 for tokenizer training
OWT_TOKENIZER_CHARS = 100_000_000

# ── Stage 1 hyperparameters — general conversational pretraining ───────────────
STAGE1_BATCH_SIZE    = 32
STAGE1_LEARNING_RATE = 3e-4
STAGE1_EPOCHS        = 1
STAGE1_WARMUP_STEPS  = 2000

# ── Stage 2 hyperparameters — medical fine-tuning ─────────────────────────────
STAGE2_BATCH_SIZE    = 16
STAGE2_LEARNING_RATE = 1e-5   # 30x lower — careful fine-tuning
STAGE2_EPOCHS        = 10
STAGE2_WARMUP_STEPS  = 200

# Legacy aliases used by existing training/eval code in main
BATCH_SIZE    = STAGE2_BATCH_SIZE
LEARNING_RATE = STAGE2_LEARNING_RATE
EPOCHS        = STAGE2_EPOCHS
PATIENCE      = 4

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Model architecture ────────────────────────────────────────────────────────
N_EMB      = 512
BLOCK_SIZE = 1024
N_LAYER    = 8
N_HEAD     = 8
DROPOUT    = 0.22
# config.py
import os

#  Paths 
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "nanochat")

DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
SPLITS_DIR  = os.path.join(DATA_DIR, "splits")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")

#  Stage 1 paths (general English pretraining) 
PRETRAIN_RAW_DIR     = os.path.join(RAW_DIR, "openwebtext")
PRETRAIN_PARQUET_DIR = os.path.join(DATA_DIR, "pretrain_parquet")
PRETRAIN_CHECKPOINT  = os.path.join(DATA_DIR, "stage1_checkpoint")

#  Stage 2 paths (medical fine-tuning) ─
WTND_PDF      = os.path.join(RAW_DIR, "where_there_is_no_doctor.pdf")
WTND_CLEAN    = os.path.join(RAW_DIR, "where_there_is_no_doctor_clean.txt")
MEDQUAD_DIR   = os.path.join(RAW_DIR, "MedQuAD")
MEDDIALOG_DIR = os.path.join(RAW_DIR, "meddialog_en")
MEDIQA_DIR    = os.path.join(RAW_DIR, "mediqa_chat")

#  Data settings 
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
RANDOM_SEED = 42

# Quality filters — balanced for nano model
MIN_Q_CHARS = 20
MIN_A_CHARS = 80
MAX_Q_CHARS = 800
MAX_A_CHARS = 1500
MAX_EXAMPLES = 150_000

# MedDialog upsampling
MEDDIALOG_UPSAMPLE = 3

#  Tokenizer settings ─
# Tokenizer is trained on BOTH OpenWebText + medical data so it has
# full coverage for both stages.
VOCAB_SIZE = 16_384
MAX_CHARS  = 500_000_000   # increased to cover OpenWebText + medical combined
DOC_CAP    = 10_000
SHARD_SIZE = 100_000

# Max chars pulled from OpenWebText for tokenizer training.
# We don't need all 38GB — 100M chars is plenty for vocab coverage.
OWT_TOKENIZER_CHARS = 100_000_000

#  Stage 1 hyperparameters — General English pretraining ─
# Higher LR and fewer epochs: we want fast broad language absorption,
# not memorisation. Large batch for stable gradients over long text.
STAGE1_BATCH_SIZE    = 32
STAGE1_LEARNING_RATE = 3e-4   # standard nanoGPT warmup LR
STAGE1_EPOCHS        = 1      # 1 pass over OpenWebText is sufficient
STAGE1_WARMUP_STEPS  = 2000

#  Stage 2 hyperparameters — Medical fine-tuning ─
# Lower LR: we're refining, not learning from scratch.
# More epochs: smaller dataset needs more passes to specialise.
STAGE2_BATCH_SIZE    = 16
STAGE2_LEARNING_RATE = 1e-5   # 30x lower than Stage 1 — careful fine-tuning
STAGE2_EPOCHS        = 10
STAGE2_WARMUP_STEPS  = 200
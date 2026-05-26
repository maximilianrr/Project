# config.py
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANOCHAT_DIR = os.path.join(PROJECT_ROOT, "..", "nanochat")
NANOCHAT_PACKAGE_DIR = os.path.join(NANOCHAT_DIR, "nanochat")

DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
TOKENIZED_DIR   = os.path.join(PROCESSED_DIR, "tokenized")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
PRE_TRAINED_DIR = os.path.join(CHECKPOINTS_DIR, "pre_trained")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "output")
SPLITS_DIR           = os.path.join(DATA_DIR, "splits")
PARQUET_DIR          = os.path.join(DATA_DIR, "parquet")
PRETRAIN_SPLITS_DIR  = os.path.join(DATA_DIR, "pretrain_splits")

MODAL_OUTPUT_DIR      = "/root/output"
MODAL_CHECKPOINTS_DIR = os.path.join(MODAL_OUTPUT_DIR, "checkpoints")

# Stage 1 — climbmix only (general English pretraining)
PRETRAIN_RAW_DIR     = os.path.join(RAW_DIR, "pretrain")
PRETRAIN_PARQUET_DIR = os.path.join(DATA_DIR, "pretrain_parquet")
PRETRAIN_CHECKPOINT  = os.path.join(CHECKPOINTS_DIR, "stage1_checkpoint")

CLIMBMIX_DIR         = os.path.join(RAW_DIR, "climbmix")
CLIMBMIX_PARQUET_DIR = os.path.join(DATA_DIR, "climbmix_parquet")

# Stage 2 — PubMed (lots) + climbmix (10%)
STAGE2_SPLITS_DIR    = os.path.join(DATA_DIR, "stage2_splits")
STAGE2_CHECKPOINT    = os.path.join(CHECKPOINTS_DIR, "stage2_checkpoint")
FINETUNE_PARQUET_DIR = os.path.join(DATA_DIR, "finetune_parquet")

# Stage 3 — oasst2 + MedQuAD + emergency cases (chatbot)
STAGE3_SPLITS_DIR    = os.path.join(DATA_DIR, "stage3_splits")
STAGE3_CHECKPOINT    = os.path.join(CHECKPOINTS_DIR, "stage3_checkpoint")

TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")
TOKENIZER_PKL  = os.path.join(TOKENIZED_DIR, "tokenizer.pkl")
BEST_MODEL_PTH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
BEST_PRETRAINED_PTH = os.path.join(PRE_TRAINED_DIR, "best_pretrained.pth")
WTND_PDF       = os.path.join(RAW_DIR, "where_there_is_no_doctor.pdf")
WTND_CLEAN     = os.path.join(RAW_DIR, "where_there_is_no_doctor_clean.txt")
MEDQUAD_DIR    = os.path.join(RAW_DIR, "MedQuAD")
MEDDIALOG_DIR  = os.path.join(RAW_DIR, "meddialog_en")
MEDIQA_DIR     = os.path.join(RAW_DIR, "mediqa_chat")
PUBMED_DIR     = os.path.join(RAW_DIR, "pubmed")
PUBMED_PARQUET_DIR = os.path.join(DATA_DIR, "pubmed_parquet")

# Emergency test cases (Stage 3)
EMERGENCY_CASES_PATH = os.path.join(RAW_DIR, "emergency_training_cases.json")

TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
RANDOM_SEED = 42

MIN_Q_CHARS = 20
MIN_A_CHARS = 80
MAX_Q_CHARS = 800
MAX_A_CHARS = 1500
MAX_EXAMPLES = 150_000

MEDDIALOG_UPSAMPLE = 3

VOCAB_SIZE = 32_768
MAX_CHARS  = 1_000_000_000
DOC_CAP    = 10_000
SHARD_SIZE = 100_000

OWT_TOKENIZER_CHARS      = 100_000_000
CLIMBMIX_TOKENIZER_CHARS = 200_000_000

# Stage 1 — climbmix raw text pretraining (high LR, no freezing)
STAGE1_BATCH_SIZE    = 32
STAGE1_LEARNING_RATE = 3e-4
STAGE1_EPOCHS        = 10
STAGE1_WARMUP_STEPS  = 2000

# Stage 2 — PubMed (lots) + 10% climbmix (lower LR than Stage 1, no freezing)
STAGE2_BATCH_SIZE    = 32
STAGE2_LEARNING_RATE = 5e-5   # ~6x lower than Stage 1
STAGE2_EPOCHS        = 10
STAGE2_WARMUP_STEPS  = 500
STAGE2_CLIMBMIX_RATIO = 0.10  # 10% climbmix mixed into Stage 2

# Stage 3 — oasst2 + MedQuAD + emergency (much lower LR, with freezing)
STAGE3_BATCH_SIZE    = 16
STAGE3_LEARNING_RATE = 5e-6   # ~10x lower than Stage 2
STAGE3_EPOCHS        = 10
STAGE3_WARMUP_STEPS  = 200
STAGE3_FREEZE_EPOCHS = 3      # freeze for first N epochs, then unfreeze

# Legacy aliases
BATCH_SIZE    = STAGE3_BATCH_SIZE
LEARNING_RATE = STAGE3_LEARNING_RATE
EPOCHS        = STAGE3_EPOCHS
PATIENCE      = 4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

N_EMB      = 1024
BLOCK_SIZE = 1024
N_LAYER    = 8
N_HEAD     = 8
DROPOUT    = 0.1
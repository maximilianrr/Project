
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
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR    = os.path.join(PROCESSED_DIR, "splits")
PARQUET_DIR   = os.path.join(PROCESSED_DIR, "parquet")
TOKENIZED_DIR = os.path.join(PROCESSED_DIR, "tokenized")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODAL_OUTPUT_DIR = "/root/output"
MODAL_CHECKPOINTS_DIR = os.path.join(MODAL_OUTPUT_DIR, "checkpoints")


# File-specific paths
TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")
TOKENIZER_PKL  = os.path.join(TOKENIZED_DIR, "tokenizer.pkl")
BEST_MODEL_PTH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
WTND_PDF       = os.path.join(RAW_DIR, "where_there_is_no_doctor.pdf")
WTND_CLEAN     = os.path.join(RAW_DIR, "where_there_is_no_doctor_clean.txt")
MEDQUAD_DIR    = os.path.join(RAW_DIR, "MedQuAD")
MEDDIALOG_DIR  = os.path.join(RAW_DIR, "meddialog_en")

#  Data settings 
TRAIN_RATIO  = 0.85
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

#  Tokenizer settings 
VOCAB_SIZE   = 32768
MAX_CHARS    = 500000000 # 500 million
DOC_CAP      = 10000
SHARD_SIZE   = 100000

# Hyperparameters for training the model
BATCH_SIZE = 16 
LEARNING_RATE = 1.5e-4
EPOCHS = 10
PATIENCE = 4


# run settings
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# model variables 
N_EMB = 512
BLOCK_SIZE = 1024
N_LAYER = 8
N_HEAD = 8
DROPOUT = 0.22
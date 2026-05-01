
# config.py
import os

# Paths 
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "nanochat")

DATA_DIR     = os.path.join(PROJECT_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
SPLITS_DIR   = os.path.join(DATA_DIR, "splits")
PARQUET_DIR  = os.path.join(DATA_DIR, "parquet")

TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")
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
LEARNING_RATE = 1e-5
EPOCHS = 10
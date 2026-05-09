
# config.py
import os
import torch

# Paths 
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "nanochat")

DATA_DIR     = os.path.join(PROJECT_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
SPLITS_DIR   = os.path.join(DATA_DIR, "splits")
PARQUET_DIR  = os.path.join(DATA_DIR, "parquet")
BEST_MODEL_DIR = os.path.join(PROJECT_DIR, "best_model")

TOKENIZER_TEXT = os.path.join(DATA_DIR, "tokenizer_text.txt")
TOKENIZER_PKL   = os.path.join(DATA_DIR, "tokenized", "tokenizer.pkl")
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
    print("Cuda is available")
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("mps i savailable")
    DEVICE = torch.device("mps")
else:
    print("Only found CPU")    
    DEVICE = torch.device("cpu")


# model variables 
N_EMB = 512
BLOCK_SIZE = 1024
N_LAYER = 8
N_HEAD = 8
DROPOUT = 0.22
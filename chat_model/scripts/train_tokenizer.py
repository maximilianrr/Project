# Trains nanochat tokenizer on medical data
"""
Run from nanochat folder with nanochat venv active:
cd ..\nanochat
.venv\Scripts\activate
python ..\Project\scripts\train_tokenizer.py
"""
import os
import sys
import time
import torch

# Load config from Project folder
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NANOCHAT_DIR = os.path.join(os.path.dirname(os.path.dirname(PROJECT_DIR)), "nanochat")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, NANOCHAT_DIR)

from config import TOKENIZER_TEXT, VOCAB_SIZE, DOC_CAP, MAX_CHARS
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

def text_iterator():
    nchars = 0
    with open(TOKENIZER_TEXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > DOC_CAP:
                line = line[:DOC_CAP]
            nchars += len(line)
            yield line
            if nchars > MAX_CHARS:
                return

print(f"Training tokenizer")
print(f"  Text file : {TOKENIZER_TEXT}")
print(f"  Vocab size: {VOCAB_SIZE:,}")

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
print(f"Training time: {time.time() - t0:.2f}s")

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    token_bytes.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))

print(f"Saved to {tokenizer_dir}")
print("Done")
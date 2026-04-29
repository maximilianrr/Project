"""
Trains nanochat tokenizer on medical data.
Run from D:\2B_proj\nanochat with nanochat venv:
cd D:\2B_proj\nanochat
python ..\Project\scripts\train_tokenizer.py
"""

import sys
sys.path.insert(0, r"D:\2B_proj\nanochat")

import os
import time
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

TEXT_FILE  = r"D:\2B_proj\Project\data\tokenizer_text.txt"
VOCAB_SIZE = 32768
DOC_CAP    = 10000
MAX_CHARS  = 500_000_000

def text_iterator():
    nchars = 0
    with open(TEXT_FILE, encoding="utf-8") as f:
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

print("Training tokenizer on medical data...")
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
print("Done.")
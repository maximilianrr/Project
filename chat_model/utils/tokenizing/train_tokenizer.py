# Trains nanochat tokenizer on medical data
import os
import importlib
import sys
import time
import pickle
import torch

# Compute repository roots and add them to sys.path so sibling repo `nanochat` is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# go up 4 levels: tokenizing -> utils -> chat_model -> Project -> Group_Project
GROUP_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
PROJECT_DIR = os.path.join(GROUP_PROJECT_DIR, "Project")
NANOCHAT_DIR = os.path.join(GROUP_PROJECT_DIR, "nanochat")

# Prefer importing the workspace root first, then the Project and nanochat folders
sys.path.insert(0, GROUP_PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, NANOCHAT_DIR)

from config import TOKENIZER_TEXT, TOKENIZED_DIR, VOCAB_SIZE, DOC_CAP, MAX_CHARS, TOKENIZER_PKL

RustBPETokenizer = importlib.import_module("nanochat.tokenizer").RustBPETokenizer

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


def train_tokenizer():
    print(f"Training tokenizer")
    print(f"  Text file : {TOKENIZER_TEXT}")
    print(f"  Vocab size: {VOCAB_SIZE:,}")

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
    print(f"Training time: {time.time() - t0:.2f}s")

    # Create output directory if it doesn't exist
    output_dir = TOKENIZED_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer as pickle
    with open(TOKENIZER_PKL, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save token_bytes for reference
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        token_bytes.append(0 if token_str in special_set else len(token_str.encode("utf-8")))

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
    token_bytes_path = os.path.join(output_dir, "token_bytes.pt")
    torch.save(token_bytes, token_bytes_path)

    print(f"Saved tokenizer to {TOKENIZER_PKL}")
    print(f"Saved token_bytes to {token_bytes_path}")
    print("Done")
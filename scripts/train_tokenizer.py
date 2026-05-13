# scripts/train_tokenizer.py
# Trains the nanochat BPE tokenizer on medical data.
# Must be run from the nanochat folder with nanochat's venv active.
#
# Usage:
#   cd ../nanochat
#   uv venv
#   uv sync --extra cpu
#   .venv/Scripts/activate          # Windows
#   source .venv/bin/activate       # Linux / macOS
#   python ../Project/scripts/train_tokenizer.py

import os
import sys
import time
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
NANOCHAT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "nanochat")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, NANOCHAT_DIR)

from config import TOKENIZER_TEXT, VOCAB_SIZE, DOC_CAP, MAX_CHARS
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

# ── Text iterator ──────────────────────────────────────────────────────────────

def text_iterator():
    """
    Yields lines from the tokenizer text file up to MAX_CHARS total.
    Each line is truncated to DOC_CAP characters so no single document
    skews the BPE merge statistics.
    """
    if not os.path.exists(TOKENIZER_TEXT):
        raise FileNotFoundError(
            f"{TOKENIZER_TEXT} not found.\n"
            "Run: python scripts/prepare_tokenizer_data.py"
        )

    nchars = 0
    with open(TOKENIZER_TEXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Hard-cap per document to keep merge frequencies balanced
            if len(line) > DOC_CAP:
                line = line[:DOC_CAP]
            nchars += len(line)
            yield line
            if nchars >= MAX_CHARS:
                print(f"  Reached MAX_CHARS cap ({MAX_CHARS:,}) — stopping iterator.")
                return


# ── Train ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Medical BPE Tokenizer Training")
print("=" * 60)
print(f"  Text file  : {TOKENIZER_TEXT}")
print(f"  Vocab size : {VOCAB_SIZE:,}")
print(f"  Doc cap    : {DOC_CAP:,} chars/line")
print(f"  Max chars  : {MAX_CHARS:,}")
print()

t0        = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
elapsed   = time.time() - t0
print(f"Training completed in {elapsed:.1f}s")

# ── Save ───────────────────────────────────────────────────────────────────────

base_dir      = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"Tokenizer saved to: {tokenizer_dir}")

# Compute per-token byte lengths and save as a tensor (used by nanochat internals)
vocab_size  = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []

for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    byte_len  = 0 if token_str in special_set else len(token_str.encode("utf-8"))
    token_bytes.append(byte_len)

token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
torch.save(token_bytes_tensor, os.path.join(tokenizer_dir, "token_bytes.pt"))
print(f"Token byte lengths saved ({vocab_size:,} tokens).")

# Quick sanity check
test_texts = [
    "What are the symptoms of hypertension?",
    "The patient should take ibuprofen 400mg twice daily after meals.",
]
print("\nSanity check:")
for text in test_texts:
    ids        = tokenizer.encode(text)
    recovered  = tokenizer.decode(ids)
    ok         = "✓" if recovered.strip() == text.strip() else "✗ MISMATCH"
    print(f"  [{ok}] tokens={len(ids):3d}  '{text[:60]}'")

print("\nDone.")

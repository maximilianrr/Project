# scripts/train_tokenizer.py
# Trains the nanochat BPE tokenizer on combined OpenWebText + medical data.
# Must be run from the nanochat folder with nanochat's venv active.
#
# The tokenizer is shared across both stages — trained once, used twice.
# It covers both general English (from OWT) and medical vocabulary.
#
# Usage (from nanochat folder, nanochat venv active):
#   python ..\Project\scripts\train_tokenizer.py

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
    Yields lines from tokenizer_text.txt up to MAX_CHARS total.
    The file already contains both OWT and medical content in the
    right proportions — prepared by prepare_tokenizer_data.py.
    Each line is capped at DOC_CAP chars so no single document
    skews the BPE merge frequency statistics.
    """
    if not os.path.exists(TOKENIZER_TEXT):
        raise FileNotFoundError(
            f"{TOKENIZER_TEXT} not found.\n"
            "Run: python scripts/prepare_tokenizer_data.py"
        )

    nchars = 0
    with open(TOKENIZER_TEXT, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > DOC_CAP:
                line = line[:DOC_CAP]
            nchars += len(line)
            yield line
            if nchars >= MAX_CHARS:
                print(f"  Reached MAX_CHARS cap ({MAX_CHARS:,}) — stopping.")
                return


# ── Train ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Two-Stage Medical Chatbot — BPE Tokenizer Training")
print("=" * 60)
print(f"  Vocab size : {VOCAB_SIZE:,}")
print(f"  Doc cap    : {DOC_CAP:,} chars/line")
print(f"  Max chars  : {MAX_CHARS:,}")
print(f"  Coverage   : OpenWebText (general) + Medical data")
print()

t0        = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
elapsed   = time.time() - t0
print(f"\nTraining completed in {elapsed:.1f}s")

# ── Save ───────────────────────────────────────────────────────────────────────

base_dir      = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"Tokenizer saved to: {tokenizer_dir}")

# Compute and save per-token byte lengths (used by nanochat internals)
vocab_size  = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []

for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    byte_len  = 0 if token_str in special_set else len(token_str.encode('utf-8'))
    token_bytes.append(byte_len)

token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
torch.save(token_bytes_tensor, os.path.join(tokenizer_dir, "token_bytes.pt"))
print(f"Token byte lengths saved ({vocab_size:,} tokens).")

# ── Sanity check ───────────────────────────────────────────────────────────────
# Test both general English and medical sentences to confirm both
# domains are well-covered by the tokenizer.

print("\nSanity check:")
test_texts = [
    # General English
    "The quick brown fox jumps over the lazy dog.",
    "She said she would call me back after the meeting.",
    # Medical conversational
    "I have had a headache for three days and I feel dizzy.",
    "You should take ibuprofen 400mg twice daily after meals.",
    # Medical technical
    "The patient was prescribed metformin for type 2 diabetes management.",
]

all_ok = True
for text in test_texts:
    ids       = tokenizer.encode(text)
    recovered = tokenizer.decode(ids)
    ok        = recovered.strip() == text.strip()
    tag       = "✓" if ok else "✗ MISMATCH"
    if not ok:
        all_ok = False
    print(f"  [{tag}]  tokens={len(ids):3d}  '{text[:65]}'")

if all_ok:
    print("\nAll checks passed. Tokenizer is ready for both Stage 1 and Stage 2.")
else:
    print("\nWarning: some round-trips failed. Check the tokenizer output above.")

print("\nNext steps:")
print("  Stage 1 — pretrain nanochat on OpenWebText parquet shards")
print("  Stage 2 — fine-tune from Stage 1 checkpoint on finetune_parquet/")
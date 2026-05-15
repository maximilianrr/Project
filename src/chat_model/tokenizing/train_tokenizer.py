# src/chat_model/tokenizing/train_tokenizer.py
"""Train one shared nanochat BPE tokenizer from tokenizer_text.txt.

The tokenizer is trained on both oasst2 (Stage 1) and medical data (Stage 2)
so it covers both general English and medical vocabulary in one vocabulary.
Must be run from the nanochat folder with nanochat's venv active.
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import time
from pathlib import Path

import torch

from chat_model import config


def _ensure_nanochat_importable() -> None:
    nanochat_dir = config.NANOCHAT_DIR
    if os.path.isdir(nanochat_dir) and nanochat_dir not in sys.path:
        sys.path.insert(0, nanochat_dir)


def _get_rust_bpe_tokenizer():
    try:
        _ensure_nanochat_importable()
        return importlib.import_module("nanochat.tokenizer").RustBPETokenizer
    except ImportError as e:
        raise ImportError(
            f"nanochat tokenizer not found. Expected at: {config.NANOCHAT_DIR}"
        ) from e


def _text_iterator():
    """Yields lines from tokenizer_text.txt up to MAX_CHARS total."""
    path = Path(config.TOKENIZER_TEXT)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: prepare_tokenizer_data() first."
        )
    chars_seen = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > config.DOC_CAP:
                line = line[:config.DOC_CAP]
            chars_seen += len(line)
            yield line
            if chars_seen >= config.MAX_CHARS:
                print(f"  Reached MAX_CHARS cap ({config.MAX_CHARS:,}) — stopping.")
                return


def _save_token_bytes(tokenizer, out_dir: Path) -> None:
    vocab_size  = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        byte_len  = 0 if token_str in special_set else len(token_str.encode("utf-8"))
        token_bytes.append(byte_len)
    tensor = torch.tensor(token_bytes, dtype=torch.int32)
    torch.save(tensor, out_dir / "token_bytes.pt")
    print(f"Token byte lengths saved ({vocab_size:,} tokens).")


def _sanity_check(tokenizer) -> bool:
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She said she would call me back after the meeting.",
        "I have had a headache for three days and I feel dizzy.",
        "You should take ibuprofen 400mg twice daily after meals.",
        "The patient was prescribed metformin for type 2 diabetes management.",
    ]
    print("\nSanity check")
    all_ok = True
    for text in test_texts:
        ids       = tokenizer.encode(text)
        recovered = tokenizer.decode(ids)
        ok        = recovered.strip() == text.strip()
        all_ok    = all_ok and ok
        print(f"  [{'OK' if ok else 'MISMATCH':<8}] tokens={len(ids):3d}  {text[:70]}")
    return all_ok


def train_tokenizer() -> None:
    """Train and save the shared BPE tokenizer."""
    print("=" * 60)
    print("Two-stage medical chatbot — BPE tokenizer training")
    print("=" * 60)
    print(f"  Vocab size : {config.VOCAB_SIZE:,}")
    print(f"  Doc cap    : {config.DOC_CAP:,} chars/line")
    print(f"  Max chars  : {config.MAX_CHARS:,}")
    print(f"  Coverage   : oasst2 (general) + medical fine-tuning + WTND\n")

    RustBPETokenizer = _get_rust_bpe_tokenizer()

    t0        = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(_text_iterator(), config.VOCAB_SIZE)
    elapsed   = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save tokenizer pickle (main branch convention)
    out_dir = Path(config.TOKENIZED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.TOKENIZER_PKL, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {config.TOKENIZER_PKL}")

    # Also save via nanochat's native save (for nanochat training compatibility)
    try:
        _ensure_nanochat_importable()
        from nanochat.common import get_base_dir
        nanochat_tok_dir = Path(get_base_dir()) / "tokenizer"
        tokenizer.save(str(nanochat_tok_dir))
        print(f"Tokenizer also saved to: {nanochat_tok_dir}")
        _save_token_bytes(tokenizer, nanochat_tok_dir)
    except Exception as e:
        print(f"Note: nanochat native save skipped ({e})")

    if _sanity_check(tokenizer):
        print("\nAll checks passed. Tokenizer is ready for Stage 1 and Stage 2.")
    else:
        print("\nWarning: at least one round-trip check failed.")


if __name__ == "__main__":
    train_tokenizer()
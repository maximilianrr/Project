# src/chat_model/tokenizing/train_tokenizer.py
"""Train one shared nanochat BPE tokenizer from tokenizer_text.txt.

The tokenizer is trained on oasst2 (Stage 1), PubMed abstracts (Stage 1),
and medical fine-tuning data (Stage 2) so it covers general English,
scientific medical vocabulary, and patient-facing language in one vocabulary.
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


# ── nanochat import helpers ───────────────────────────────────────────────────

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


# ── Text iterator ─────────────────────────────────────────────────────────────

def _text_iterator(max_chars: int | None = None):
    """
    Yields lines from tokenizer_text.txt up to max_chars total.
    Falls back to config.MAX_CHARS if max_chars is not provided.
    Reports progress every 10M chars so long runs aren't silent.
    """
    path = Path(config.TOKENIZER_TEXT)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: prepare_tokenizer_data() first."
        )

    cap        = max_chars if max_chars is not None else config.MAX_CHARS
    chars_seen = 0
    last_report_at = 0
    report_every   = 10_000_000  # report every 10M chars

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > config.DOC_CAP:
                line = line[:config.DOC_CAP]
            chars_seen += len(line)

            if chars_seen - last_report_at >= report_every:
                print(f"  ... {chars_seen / 1e6:.0f} MB read", flush=True)
                last_report_at = chars_seen

            yield line

            if cap and chars_seen >= cap:
                print(f"  Reached char cap ({cap:,}) — stopping iterator.")
                return


# ── Post-training helpers ─────────────────────────────────────────────────────

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
    print(f"  Token byte lengths saved ({vocab_size:,} tokens).")


def _sanity_check(tokenizer) -> bool:
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She said she would call me back after the meeting.",
        "I have had a headache for three days and I feel dizzy.",
        "You should take ibuprofen 400mg twice daily after meals.",
        "The patient was prescribed metformin for type 2 diabetes management.",
        "Serum creatinine levels were elevated, suggesting acute kidney injury.",
        "The abstract concluded that prophylactic antibiotic use reduced postoperative complications.",
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


# ── Entry point ───────────────────────────────────────────────────────────────

def train_tokenizer(max_chars: int | None = None) -> None:
    """Train and save the shared BPE tokenizer."""
    cap = max_chars if max_chars is not None else config.MAX_CHARS

    print("=" * 60)
    print("Two-stage medical chatbot — BPE tokenizer training")
    print("=" * 60)
    print(f"  Vocab size : {config.VOCAB_SIZE:,}")
    print(f"  Doc cap    : {config.DOC_CAP:,} chars/line")
    print(f"  Max chars  : {cap:,}")
    print(f"  Coverage   : oasst2 (general) + PubMed (medical literature)")
    print(f"             + medical fine-tuning splits + WTND\n")

    RustBPETokenizer = _get_rust_bpe_tokenizer()

    print("Reading tokenizer_text.txt and training...")
    t0        = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(_text_iterator(cap), config.VOCAB_SIZE)
    elapsed   = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save tokenizer pickle (main branch convention)
    out_dir = Path(config.TOKENIZED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.TOKENIZER_PKL, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to : {config.TOKENIZER_PKL}")

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
        print("\nAll checks passed. Tokenizer is ready.")
    else:
        print("\nWarning: at least one round-trip check failed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train shared BPE tokenizer.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help=(
            "Cap total chars read from tokenizer_text.txt. "
            "Defaults to config.MAX_CHARS. "
            "Lower this (e.g. 50000000) for a quick test run."
        ),
    )
    args = parser.parse_args()
    train_tokenizer(max_chars=args.max_chars)
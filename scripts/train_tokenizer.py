# scripts/train_tokenizer.py
"""Train one shared nanochat BPE tokenizer from tokenizer_text.txt."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NANOCHAT_DIR = PROJECT_DIR.parent / "nanochat"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(NANOCHAT_DIR))

from config import DOC_CAP, MAX_CHARS, TOKENIZER_TEXT, VOCAB_SIZE

def text_iterator(path: str, doc_cap: int, max_chars: int):
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run scripts/prepare_tokenizer_data.py first.")

    chars_seen = 0
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if len(line) > doc_cap:
                line = line[:doc_cap]
            chars_seen += len(line)
            yield line
            if chars_seen >= max_chars:
                print(f"  Reached MAX_CHARS cap ({max_chars:,}); stopping.")
                return


def save_token_bytes(tokenizer, tokenizer_dir: Path) -> None:
    import torch
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        byte_len = 0 if token_str in special_set else len(token_str.encode("utf-8"))
        token_bytes.append(byte_len)

    token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
    torch.save(token_bytes_tensor, tokenizer_dir / "token_bytes.pt")
    print(f"Token byte lengths saved ({vocab_size:,} tokens).")


def sanity_check(tokenizer) -> bool:
    print("\nSanity check")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She said she would call me back after the meeting.",
        "I have had a headache for three days and I feel dizzy.",
        "You should take ibuprofen 400mg twice daily after meals.",
        "The patient was prescribed metformin for type 2 diabetes management.",
    ]

    all_ok = True
    for text in test_texts:
        ids = tokenizer.encode(text)
        recovered = tokenizer.decode(ids)
        ok = recovered.strip() == text.strip()
        all_ok = all_ok and ok
        tag = "OK" if ok else "MISMATCH"
        print(f"  [{tag:<8}] tokens={len(ids):3d}  {text[:70]}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared nanochat tokenizer.")
    parser.add_argument("--input", default=TOKENIZER_TEXT, help="Tokenizer corpus text file.")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help="BPE vocab size.")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="Max characters to train on.")
    parser.add_argument("--doc-cap", type=int, default=DOC_CAP, help="Max characters per yielded line.")
    args = parser.parse_args()

    print("=" * 60)
    print("Two-stage medical chatbot tokenizer training")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Vocab size : {args.vocab_size:,}")
    print(f"  Doc cap    : {args.doc_cap:,} chars/line")
    print(f"  Max chars  : {args.max_chars:,}")
    print("  Coverage   : OASST2 pretraining + medical fine-tuning + WTND")
    print()

    try:
        from nanochat.common import get_base_dir
        from nanochat.tokenizer import RustBPETokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tokenizer training must run in the nanochat environment with its dependencies installed. "
            "Activate ../nanochat/.venv or install nanochat requirements, then retry."
        ) from exc
    
    started = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iterator(args.input, args.doc_cap, args.max_chars),
        args.vocab_size,
    )
    elapsed = time.time() - started
    print(f"\nTraining completed in {elapsed:.1f}s")

    tokenizer_dir = Path(get_base_dir()) / "tokenizer"
    tokenizer.save(str(tokenizer_dir))
    print(f"Tokenizer saved to: {tokenizer_dir}")
    save_token_bytes(tokenizer, tokenizer_dir)

    if sanity_check(tokenizer):
        print("\nAll checks passed. Tokenizer is ready for Stage 1 and Stage 2.")
    else:
        print("\nWarning: at least one round-trip check failed.")


if __name__ == "__main__":
    main()
# scripts/prepare_tokenizer_data.py
# Builds tokenizer_text.txt from TinyStories (Stage 1) + medical data (Stage 2).
#
# TinyStories gives the tokenizer clean English BPE merges.
# Medical splits give it medical vocabulary as single tokens.
# Both domains covered in one tokenizer — shared across both training stages.
#
# Run: python scripts/prepare_tokenizer_data.py

import os
import sys
import re
import json
import glob
import unicodedata

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SPLITS_DIR, WTND_CLEAN, TOKENIZER_TEXT, DOC_CAP,
    PRETRAIN_PARQUET_DIR, OWT_TOKENIZER_CHARS,
)

os.makedirs(os.path.dirname(TOKENIZER_TEXT), exist_ok=True)


def normalise(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


total_lines = 0
total_chars = 0

with open(TOKENIZER_TEXT, 'w', encoding='utf-8') as out:

    # ── 1. TinyStories — general English ──────────────────────────────────────
    # We use all downloaded stories (already capped at 500k by download script).
    # TinyStories are short (~500-2000 chars each) so no chunking needed —
    # each story fits cleanly as 1-2 tokenizer lines.

    shards = sorted(glob.glob(os.path.join(PRETRAIN_PARQUET_DIR, "ts_*.parquet")))

    if shards:
        ts_chars = 0
        ts_lines = 0
        print(f"Reading TinyStories from {len(shards)} shard(s)...")

        for shard_path in shards:
            table = pq.read_table(shard_path, columns=["text"])
            for text in table.column("text").to_pylist():
                text = normalise(text or "")
                if not text:
                    continue
                # Chunk only if story exceeds DOC_CAP (rare for TinyStories)
                for i in range(0, len(text), DOC_CAP):
                    chunk = text[i:i + DOC_CAP].strip()
                    if chunk:
                        out.write(chunk + '\n')
                        ts_lines    += 1
                        ts_chars    += len(chunk)
                        total_lines += 1
                        total_chars += len(chunk)

        print(f"  TinyStories : {ts_lines:>8,} lines  ({ts_chars/1e6:.0f}MB)")
    else:
        print("  TinyStories : NOT FOUND — run download_pretrain_data.py first")
        print("                Tokenizer will be medical-only (suboptimal for grammar)")

    # ── 2. Medical splits — train + val only (never test) ─────────────────────
    med_lines = 0
    for split in ('train', 'val'):
        path = os.path.join(SPLITS_DIR, f'{split}.jsonl')
        if not os.path.exists(path):
            print(f"  MISSING: {path} — run preprocess.py first")
            continue
        with open(path, encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    conv = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for msg in conv:
                    content = normalise(msg.get('content', ''))
                    if content:
                        out.write(content[:DOC_CAP] + '\n')
                        med_lines   += 1
                        total_lines += 1
                        total_chars += len(content)

    print(f"  Medical data: {med_lines:>8,} lines")

    # ── 3. WTND book ──────────────────────────────────────────────────────────
    wtnd_lines = 0
    if os.path.exists(WTND_CLEAN):
        with open(WTND_CLEAN, encoding='utf-8') as f:
            for paragraph in re.split(r'\n{2,}', f.read()):
                paragraph = normalise(paragraph)
                if len(paragraph) >= 30:
                    for i in range(0, len(paragraph), DOC_CAP):
                        chunk = paragraph[i:i + DOC_CAP].strip()
                        if chunk:
                            out.write(chunk + '\n')
                            wtnd_lines  += 1
                            total_lines += 1
                            total_chars += len(chunk)
        print(f"  WTND book   : {wtnd_lines:>8,} lines")
    else:
        print("  WTND        : NOT FOUND — run download_wtnd.py first")

print(f"\nTotal lines : {total_lines:,}")
print(f"Total chars : {total_chars/1e6:.0f}MB")
print(f"Saved to    : {TOKENIZER_TEXT}")
print(f"\nNext: python scripts/convert_to_parquet.py")
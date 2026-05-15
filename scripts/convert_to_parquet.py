# scripts/convert_to_parquet.py
# Converts tokenizer_text.txt into sharded Parquet files for nanochat's
# tokenizer trainer. Also separately shards the medical splits into
# Stage 2 fine-tuning parquet files.
#
# Run: python scripts/convert_to_parquet.py

import os
import sys
import re
import json
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_TEXT, PARQUET_DIR, SPLITS_DIR, SHARD_SIZE

schema = pa.schema([pa.field("text", pa.string())])


def normalise(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ── 1. Tokenizer parquet (tokenizer_text.txt → shards) ────────────────────────

if not os.path.exists(TOKENIZER_TEXT):
    raise FileNotFoundError(
        f"{TOKENIZER_TEXT} not found.\n"
        "Run scripts/prepare_tokenizer_data.py first."
    )

print(f"Reading {TOKENIZER_TEXT}...")
with open(TOKENIZER_TEXT, encoding='utf-8') as f:
    lines = [normalise(l) for l in f if l.strip()]
print(f"Total lines: {len(lines):,}")

os.makedirs(PARQUET_DIR, exist_ok=True)
shards = [lines[i:i + SHARD_SIZE] for i in range(0, len(lines), SHARD_SIZE)]
print(f"Saving {len(shards)} tokenizer shard(s) to {PARQUET_DIR}/...\n")

for i, shard in enumerate(shards):
    table    = pa.table({"text": shard}, schema=schema)
    out_path = os.path.join(PARQUET_DIR, f"shard_{i:05d}.parquet")
    pq.write_table(table, out_path, compression="snappy")
    tag = " ← validation shard (nanochat convention)" if i == len(shards) - 1 else ""
    print(f"  shard_{i:05d}.parquet  {len(shard):>8,} rows{tag}")

# ── 2. Stage 2 fine-tuning parquet (medical splits → conversation format) ──────
# Nanochat's fine-tuning loader expects one JSON conversation per row.
# We write train and val splits separately so nanochat can use them directly.

STAGE2_DIR = os.path.join(os.path.dirname(PARQUET_DIR), "finetune_parquet")
os.makedirs(STAGE2_DIR, exist_ok=True)

print(f"\nConverting medical splits to fine-tune parquet → {STAGE2_DIR}/")

for split in ('train', 'val', 'test'):
    path = os.path.join(SPLITS_DIR, f'{split}.jsonl')
    if not os.path.exists(path):
        print(f"  {split}: NOT FOUND — skipped")
        continue

    rows = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                rows.append(raw)   # keep as raw JSON string — nanochat parses it

    # Shard if large
    file_shards = [rows[i:i + SHARD_SIZE] for i in range(0, len(rows), SHARD_SIZE)]
    for i, shard in enumerate(file_shards):
        suffix   = f"_{i:05d}" if len(file_shards) > 1 else ""
        out_path = os.path.join(STAGE2_DIR, f"{split}{suffix}.parquet")
        table    = pa.table({"text": shard}, schema=schema)
        pq.write_table(table, out_path, compression="snappy")
        print(f"  {split}{suffix}.parquet  {len(shard):>8,} rows")

print("\nDone.")
print(f"\n── Summary ────────────────────────────────────────────────────")
print(f"  Tokenizer shards : {PARQUET_DIR}/")
print(f"  Stage 2 shards   : {STAGE2_DIR}/")
print(f"  Next step        : train_tokenizer.py  (nanochat venv)")
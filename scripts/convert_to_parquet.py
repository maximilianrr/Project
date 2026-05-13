# scripts/convert_to_parquet.py
# Converts tokenizer_text.txt into sharded Parquet files for nanochat's tokenizer trainer.
# Nanochat expects the last shard to be used as the validation shard (its convention).
#
# Run: python scripts/convert_to_parquet.py

import os
import sys
import unicodedata
import re

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_TEXT, PARQUET_DIR, SHARD_SIZE

# ── Load ───────────────────────────────────────────────────────────────────────
if not os.path.exists(TOKENIZER_TEXT):
    raise FileNotFoundError(
        f"{TOKENIZER_TEXT} not found. Run scripts/prepare_tokenizer_data.py first."
    )

print(f"Reading {TOKENIZER_TEXT}...")

def normalise(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = re.sub(r'[ \t]+', ' ', line)
    return line.strip()

with open(TOKENIZER_TEXT, encoding="utf-8") as f:
    lines = [normalise(l) for l in f if l.strip()]

print(f"Total lines: {len(lines):,}")

# ── Shard & write ──────────────────────────────────────────────────────────────
os.makedirs(PARQUET_DIR, exist_ok=True)

shards = [lines[i:i + SHARD_SIZE] for i in range(0, len(lines), SHARD_SIZE)]
print(f"Saving {len(shards)} parquet shard(s) to {PARQUET_DIR}/...")

schema = pa.schema([pa.field("text", pa.string())])

for i, shard in enumerate(shards):
    table    = pa.table({"text": shard}, schema=schema)
    out_path = os.path.join(PARQUET_DIR, f"shard_{i:05d}.parquet")
    pq.write_table(table, out_path, compression="snappy")
    tag = " [validation shard — nanochat convention]" if i == len(shards) - 1 else ""
    print(f"  shard_{i:05d}.parquet → {len(shard):,} rows{tag}")

print(f"\nDone. All shards saved to {PARQUET_DIR}/")

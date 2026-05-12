import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq

# Add src to path if needed (allows running this script directly)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model import config

def convert_to_parquet():
    """
    Converts medical text data to parquet format for nanochat's tokenizer trainer.
    Run: python scripts/convert_to_parquet.py
    """
    
    os.makedirs(config.PARQUET_DIR, exist_ok=True)

    print(f"Reading {config.TOKENIZER_TEXT}...")
    with open(config.TOKENIZER_TEXT, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Total lines: {len(lines):,}")

    # Split into shards
    shards = [lines[i:i+config.SHARD_SIZE] for i in range(0, len(lines), config.SHARD_SIZE)]
    print(f"Saving {len(shards)} parquet shard(s)...")

    for i, shard in enumerate(shards):
        table = pa.table({"text": shard})
        # Last shard = validation, rest = train (nanochat convention)
        out_path = os.path.join(config.PARQUET_DIR, f"shard_{i:05d}.parquet")
        pq.write_table(table, out_path)
        print(f"  shard_{i:05d}.parquet → {len(shard):,} rows")

    print(f"\nDone. Saved to {config.PARQUET_DIR}")
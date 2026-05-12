import os
import pyarrow as pa
import pyarrow.parquet as pq

TEXT_FILE  = "data/tokenizer_text.txt"
OUT_DIR    = "data/parquet"
SHARD_SIZE = 100000  # lines per parquet file

def convert_to_parquet():
    """
    Converts medical text data to parquet format for nanochat's tokenizer trainer.
    Run: python scripts/convert_to_parquet.py
    """
    
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading {TEXT_FILE}...")
    with open(TEXT_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Total lines: {len(lines):,}")

    # Split into shards
    shards = [lines[i:i+SHARD_SIZE] for i in range(0, len(lines), SHARD_SIZE)]
    print(f"Saving {len(shards)} parquet shard(s)...")

    for i, shard in enumerate(shards):
        table = pa.table({"text": shard})
        # Last shard = validation, rest = train (nanochat convention)
        out_path = os.path.join(OUT_DIR, f"shard_{i:05d}.parquet")
        pq.write_table(table, out_path)
        print(f"  shard_{i:05d}.parquet → {len(shard):,} rows")

    print(f"\nDone. Saved to {OUT_DIR}")
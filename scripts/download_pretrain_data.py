# scripts/download_pretrain_data.py
# Downloads TinyStories for Stage 1 general English pretraining.
#
# TinyStories is a dataset of short simple English stories generated to be
# readable by small language models. It is ideal for NanoChat pretraining:
#   - Clean, grammatically correct English
#   - Simple sentence structures the model can actually learn
#   - ~2GB total — fits comfortably on a laptop
#   - No web noise, ads, or broken formatting
#
# Full dataset: ~2.1M stories, ~2GB
# Adjust MAX_STORIES below if you want even less.
#
# Run: python scripts/download_pretrain_data.py

import os
import sys
import re
import unicodedata
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRETRAIN_RAW_DIR, PRETRAIN_PARQUET_DIR, SHARD_SIZE

#  Settings 
# Full TinyStories is ~2.1M stories and ~2GB.
# For NanoChat, 500k stories (~500MB) is plenty to establish English foundations.
# Increase toward 2_100_000 if you have more space and want richer coverage.
MAX_STORIES   = 2_100_000
MIN_DOC_CHARS = 100    # skip very short stubs
MAX_DOC_CHARS = 8_000  # skip outliers (TinyStories are typically 500-2000 chars)

os.makedirs(PRETRAIN_RAW_DIR,     exist_ok=True)
os.makedirs(PRETRAIN_PARQUET_DIR, exist_ok=True)

#  Cleaning 
# TinyStories is already very clean — minimal cleaning needed.

_MULTI_WS = re.compile(r'[ \t]+')
_MULTI_NL = re.compile(r'\n{4,}')


def clean_story(text: str) -> str:
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = _MULTI_WS.sub(' ', text)
    text = _MULTI_NL.sub('\n\n', text)
    return text.strip()


# Stream and save

print("Streaming TinyStories from HuggingFace...")
print(f"  Target  : {MAX_STORIES:,} stories")
print(f"  Storage : ~{MAX_STORIES / 2_100_000 * 2:.1f}GB estimated")
print(f"  Output  : {PRETRAIN_PARQUET_DIR}/\n")

ds = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

shard_idx    = 0
shard_buffer = []
total_stories = 0
total_chars   = 0
skipped       = 0
schema        = pa.schema([pa.field("text", pa.string())])

for doc in ds:
    if total_stories >= MAX_STORIES:
        break

    text = clean_story(doc.get("text", ""))

    if len(text) < MIN_DOC_CHARS or len(text) > MAX_DOC_CHARS:
        skipped += 1
        continue

    shard_buffer.append(text)
    total_stories += 1
    total_chars   += len(text)

    # Flush shard when full
    if len(shard_buffer) >= SHARD_SIZE:
        out_path = os.path.join(PRETRAIN_PARQUET_DIR, f"ts_{shard_idx:05d}.parquet")
        table    = pa.table({"text": shard_buffer}, schema=schema)
        pq.write_table(table, out_path, compression="snappy")
        print(f"  Saved shard {shard_idx:05d}  →  {len(shard_buffer):,} stories  "
              f"({total_stories:,} total,  {total_chars/1e6:.0f}MB)")
        shard_buffer = []
        shard_idx   += 1

# Flush remainder
if shard_buffer:
    out_path = os.path.join(PRETRAIN_PARQUET_DIR, f"ts_{shard_idx:05d}.parquet")
    table    = pa.table({"text": shard_buffer}, schema=schema)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Saved shard {shard_idx:05d}  →  {len(shard_buffer):,} stories  "
          f"({total_stories:,} total,  {total_chars/1e6:.0f}MB)")

print(f"\nDone.")
print(f"  Stories saved   : {total_stories:,}")
print(f"  Stories skipped : {skipped:,}")
print(f"  Total size      : ~{total_chars/1e6:.0f}MB")
print(f"  Shards          : {shard_idx + 1}")
print(f"  Output dir      : {PRETRAIN_PARQUET_DIR}/")
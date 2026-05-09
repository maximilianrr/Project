# scripts/prepare_tokenizer_data.py
# Prepares a single text file from all medical data sources to train nanochat's tokenizer on medical vocabulary

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPLITS_DIR, WTND_CLEAN, TOKENIZER_TEXT


def prepare_tokenizer_data():
    total = 0
    with open(TOKENIZER_TEXT, "w", encoding="utf-8") as out:

        # Splits (train + val) 
        for split in ["train", "val"]:
            path = os.path.join(SPLITS_DIR, f"{split}.jsonl")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    for msg in item:
                        out.write(msg["content"] + "\n")
                        total += 1

        # WTIND 
        if os.path.exists(WTND_CLEAN):
            with open(WTND_CLEAN, encoding="utf-8") as f:
                out.write(f.read() + "\n")
            print("Added WTND book text")
        else:
            print("WTND not found — run download_wtnd.py first")

    print(f"Total lines written: {total:,}")
    print(f"Saved to: {TOKENIZER_TEXT}")
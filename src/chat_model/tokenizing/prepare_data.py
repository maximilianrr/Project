# Prepares a single text file from all medical data sources to train nanochat's tokenizer on medical vocabulary

import os
import sys
import json

# Add src to path if needed (allows running this script directly)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model import config


def prepare_tokenizer_data():
    """
    Prepares a single text file from all medical data sources to train nanochat's tokenizer on medical vocabulary.
    Combines the train and val splits (which contain conversations) and the cleaned WTND book text (which contains medical terminology) into one text file for tokenizer training.
    """
    
    total = 0
    with open(config.TOKENIZER_TEXT, "w", encoding="utf-8") as out:

        # Splits (train + val) 
        for split in ["train", "val"]:
            path = os.path.join(config.SPLITS_DIR, f"{split}.jsonl")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    for msg in item:
                        out.write(msg["content"] + "\n")
                        total += 1

        # WTIND 
        if os.path.exists(config.WTND_CLEAN):
            with open(config.WTND_CLEAN, encoding="utf-8") as f:
                out.write(f.read() + "\n")
            print("Added WTND book text")
        else:
            print("WTND not found — run download_wtnd.py first")

    print(f"Total lines written: {total:,}")
    print(f"Saved to: {config.TOKENIZER_TEXT}")
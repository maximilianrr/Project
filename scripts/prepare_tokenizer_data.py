# scripts/prepare_tokenizer_data.py
"""
Prepares a single text file from all medical data sources
to train nanochat's tokenizer on medical vocabulary.
Run: python scripts/prepare_tokenizer_data.py
"""
import os
import json

out_path = "data/tokenizer_text.txt"
total = 0

with open(out_path, "w", encoding="utf-8") as out:

    #  Splits (train + val) 
    for split in ["train", "val"]:
        path = f"data/splits/{split}.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for msg in item:
                    out.write(msg["content"] + "\n")
                    total += 1

    #  Where There Is No Doctor 
    wtnd = "data/raw/where_there_is_no_doctor_clean.txt"
    if os.path.exists(wtnd):
        with open(wtnd, encoding="utf-8") as f:
            out.write(f.read() + "\n")
        print("  Added WTND book text")

print(f"Total lines written: {total:,}")
print(f"Saved to: {out_path}")
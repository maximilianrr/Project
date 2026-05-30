import pickle
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "nanochat"))

with open("data/processed/tokenized/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

climbmix_tokens = 0
with open("data/pretrain_splits/climbmix_docs.txt", encoding="utf-8") as f:
    for line in f:
        climbmix_tokens += len(tokenizer.encode(line.strip()))

oasst2_tokens = 0
with open("data/pretrain_splits/train.jsonl", encoding="utf-8") as f:
    for line in f:
        conv = json.loads(line)
        for msg in conv:
            oasst2_tokens += len(tokenizer.encode(msg["content"]))

medical_tokens = 0
with open("data/splits/train.jsonl", encoding="utf-8") as f:
    for line in f:
        conv = json.loads(line)
        for msg in conv:
            medical_tokens += len(tokenizer.encode(msg["content"]))

print(f"climbmix  : {climbmix_tokens:,} tokens")
print(f"oasst2    : {oasst2_tokens:,} tokens")
print(f"Medical   : {medical_tokens:,} tokens")
print(f"Ratio     : {(climbmix_tokens + oasst2_tokens) / max(medical_tokens, 1):.1f}x more English than medical")
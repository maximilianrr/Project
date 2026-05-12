import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
import config

# Settings
SMOKE_TEST = True   # set False for full evaluation

# Paths
REPO         = os.path.join(config.PROJECT_DIR, "repo")
NANOCHAT_DIR = os.path.join(config.PROJECT_DIR, "nanochat")

sys.path.insert(0, REPO)
sys.path.insert(0, NANOCHAT_DIR)

from models.model import NanoChat
import eval as E
import config

# Tokenizer
import pickle

assert os.path.isfile(config.TOKENIZER_PKL), (
    f"Training tokenizer not found at {config.TOKENIZER_PKL}.\n"
)
with open(config.TOKENIZER_PKL, "rb") as f:
    tokenizer = pickle.load(f)

# Model
model = NanoChat(config).to(config.DEVICE)

if os.path.isfile(config.BEST_MODEL_PTH):
    model.load_state_dict(torch.load(config.BEST_MODEL_PTH, map_location=config.DEVICE))
    print(f"Loaded checkpoint: {config.BEST_MODEL_PTH}")
else:
    print("Warning: no checkpoint set — running with random weights.")

# Data
def load_split(split):
    path = os.path.join(config.SPLITS_DIR, f"{split}.jsonl")
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Split not found: {path}. Run scripts/preprocess.py first.")

class TokenBlockDataset(Dataset):
    def __init__(self, conversations, block_size):
        all_ids = []
        for convo in conversations:
            for msg in convo:
                all_ids.extend(tokenizer.encode(msg.get("content", "")))
        self.examples = [
            (
                torch.tensor(all_ids[s:s + block_size], dtype=torch.long),
                torch.tensor(all_ids[s + 1:s + block_size + 1], dtype=torch.long),
            )
            for s in range(0, len(all_ids) - block_size, block_size)
        ]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

raw_val  = load_split("val")
raw_test = load_split("test")

val_loader = DataLoader(
    TokenBlockDataset(raw_val[:500] if SMOKE_TEST else raw_val, config.BLOCK_SIZE),
    batch_size=config.BATCH_SIZE, shuffle=False, drop_last=True,
)

test_set = [
    {"question": c[0]["content"], "answer": c[1]["content"]}
    for c in raw_test if len(c) >= 2
]
safety_cases = E.load_test_cases(os.path.join(config.PROJECT_DIR, "safety_cases.json"))

if SMOKE_TEST:
    test_set = test_set[:100]

# Generate Function
def generate_fn(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0][len(ids):].tolist())

# Run
results = E.run_full_eval(
    model=model,
    tokenizer=tokenizer,
    val_loader=val_loader,
    test_set=test_set,
    safety_cases=safety_cases,
    generate_fn=generate_fn,
    device=config.DEVICE,
    max_ppl_batches=5 if SMOKE_TEST else None,
)

print(json.dumps(results, indent=2, default=str))

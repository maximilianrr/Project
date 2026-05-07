import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

# ── settings ──────────────────────────────────────────────────────────────────
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(PROJ_DIR, "weights_trial_1_LR0.00014856360614407606_BS16_DP0.22_best.pth.zip")
SMOKE_TEST = True   # set False for full evaluation

# ── paths ─────────────────────────────────────────────────────────────────────
REPO         = os.path.join(PROJ_DIR, "repo")
NANOCHAT_DIR = os.path.join(PROJ_DIR, "nanochat")
SPLITS_DIR   = os.path.join(REPO, "chat_model", "data", "splits")

sys.path.insert(0, REPO)
sys.path.insert(0, NANOCHAT_DIR)

from model import NanoChat
import eval as E

# ── config ────────────────────────────────────────────────────────────────────
class Config:
    n_embd     = 512
    n_head     = 8
    n_layer    = 4
    block_size = 1024
    dropout    = 0.22
    VOCAB_SIZE = 32768
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

config = Config()

# ── tokenizer ────────────────────────────────────────────────────────────────
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
assert os.path.isdir(tokenizer_dir), (
    f"Tokenizer not found at {tokenizer_dir}. Run scripts/train_tokenizer.py first."
)
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

# ── model ────────────────────────────────────────────────────────────────────
model = NanoChat(config).to(config.DEVICE)

if CHECKPOINT:
    model.load_state_dict(torch.load(CHECKPOINT, map_location=config.DEVICE))
    print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    print("Warning: no checkpoint set — running with random weights.")

# ── data ─────────────────────────────────────────────────────────────────────
def load_split(split):
    path = os.path.join(SPLITS_DIR, f"{split}.jsonl")
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Split not found: {path}. Run scripts/preprocess.py first.")

class TokenBlockDataset(Dataset):
    def __init__(self, conversations, block_size):
        self.examples = []
        for convo in conversations:
            ids = []
            for msg in convo:
                ids.extend(tokenizer.encode(msg.get("content", "")))
            for start in range(len(ids) - block_size):
                self.examples.append((
                    torch.tensor(ids[start:start + block_size], dtype=torch.long),
                    torch.tensor(ids[start + 1:start + block_size + 1], dtype=torch.long),
                ))
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

raw_val  = load_split("val")
raw_test = load_split("test")

val_loader = DataLoader(
    TokenBlockDataset(raw_val[:50] if SMOKE_TEST else raw_val, config.block_size),
    batch_size=16, shuffle=False, drop_last=True,
)

test_set = [
    {"question": c[0]["content"], "answer": c[1]["content"]}
    for c in raw_test if len(c) >= 2
]
safety_cases = E.load_test_cases(os.path.join(PROJ_DIR, "safety_cases.json"))

if SMOKE_TEST:
    test_set     = test_set[:5]
    safety_cases = safety_cases[:6]

# ── generate_fn ───────────────────────────────────────────────────────────────
def generate_fn(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0][len(ids):].tolist())

# ── run ───────────────────────────────────────────────────────────────────────
results = E.run_full_eval(
    model=model,
    tokenizer=tokenizer,
    val_loader=val_loader,
    test_set=test_set,
    safety_cases=safety_cases,
    generate_fn=generate_fn,
    device=str(config.DEVICE),
    max_ppl_batches=5 if SMOKE_TEST else None,
)

print(json.dumps(results, indent=2, default=str))

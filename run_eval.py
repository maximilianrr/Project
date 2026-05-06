"""
Run the full MedChat evaluation suite.

Prerequisites — complete in order before running this script:
  1. Set up the conda environment  (see README.md)
  2. Clone nanochat as a sibling of this repo  (see README.md)
  3. Train the tokenizer  (see README.md)
  4. Download and preprocess the data  (see README.md)
  5. Obtain a trained model checkpoint and set CHECKPOINT below.
"""

import json
import os
import sys

import torch

# ── paths ────────────────────────────────────────────────────────────────────
REPO         = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(os.path.dirname(REPO), "nanochat")

sys.path.insert(0, REPO)
sys.path.insert(0, NANOCHAT_DIR)

from chat_model.models.model import NanoChat
from chat_model import config
from chat_model.utils.data_loader import load_tokenized_dataloader, load_split
import eval as E

# ── checkpoint ───────────────────────────────────────────────────────────────
CHECKPOINT = None  # set to "path/to/checkpoint.pt" once training is complete

# ── tokenizer ────────────────────────────────────────────────────────────────
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
assert os.path.isdir(tokenizer_dir), (
    f"Tokenizer not found at {tokenizer_dir}.\n"
    "Run the tokenizer training steps in README.md first."
)
tokenizer = RustBPETokenizer.load(tokenizer_dir)

# ── model ────────────────────────────────────────────────────────────────────
device = str(config.DEVICE)
model = NanoChat(config).to(device)

if CHECKPOINT:
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    print("Warning: no checkpoint set — evaluating with random weights.")

# ── data ─────────────────────────────────────────────────────────────────────
val_loader = load_tokenized_dataloader("val", tokenizer=tokenizer)

raw_test = load_split("test")
test_set = [
    {"question": convo[0]["content"], "answer": convo[1]["content"]}
    for convo in raw_test
    if len(convo) >= 2
]

safety_cases = E.load_test_cases(os.path.join(REPO, "safety_cases.json"))

# ── generation function ───────────────────────────────────────────────────────
def generate_fn(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens)
    new_tokens = output[0][len(ids):].tolist()
    return tokenizer.decode(new_tokens)

# ── run ───────────────────────────────────────────────────────────────────────
results = E.run_full_eval(
    model=model,
    tokenizer=tokenizer,
    val_loader=val_loader,
    test_set=test_set,
    safety_cases=safety_cases,
    generate_fn=generate_fn,
    device=device,
)

print(json.dumps(results, indent=2, default=str))

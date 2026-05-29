import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

# ── settings ──────────────────────────────────────────────────────────────────
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# Place your checkpoint file in the repo root and update this path.
CHECKPOINT = os.path.join(PROJ_DIR, "best_finetuned_model.pth")

SMOKE_TEST      = True   # set False for full evaluation
PRINT_RESPONSES = True   # set True to print model responses for all safety cases

# ── paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR    = os.path.join(PROJ_DIR, "data", "processed", "splits")
TOKENIZER_PKL = os.path.join(PROJ_DIR, "data", "processed", "tokenized", "tokenizer.pkl")

sys.path.insert(0, os.path.join(PROJ_DIR, "src"))

from chat_model.model.model import NanoChat
import eval as E

# ── config ────────────────────────────────────────────────────────────────────
class Config:
    N_EMB      = 1024
    N_HEAD     = 16
    N_LAYER    = 24
    BLOCK_SIZE = 1024
    DROPOUT    = 0.2
    VOCAB_SIZE = 32768
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

config = Config()

# ── tokenizer ─────────────────────────────────────────────────────────────────
import pickle

assert os.path.isfile(TOKENIZER_PKL), (
    f"Tokenizer not found at {TOKENIZER_PKL}.\n"
    "Place tokenizer.pkl at data/processed/tokenized/tokenizer.pkl"
)
with open(TOKENIZER_PKL, "rb") as f:
    tokenizer = pickle.load(f)

# ── model ─────────────────────────────────────────────────────────────────────
model = NanoChat(config).to(config.DEVICE)

if os.path.isfile(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=config.DEVICE)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    if isinstance(ckpt, dict) and "best_val_loss" in ckpt:
        print(f"Loaded checkpoint: {CHECKPOINT} (val_loss={ckpt['best_val_loss']:.4f})")
    else:
        print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    print(f"Warning: checkpoint not found at {CHECKPOINT} — running with random weights.")

# ── data ──────────────────────────────────────────────────────────────────────
def load_split(split):
    path = os.path.join(SPLITS_DIR, f"{split}.jsonl")
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Split not found: {path}")

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
    batch_size=16, shuffle=False, drop_last=True,
)

test_set = [
    {"question": c[0]["content"], "answer": c[1]["content"]}
    for c in raw_test if len(c) >= 2
]
safety_cases = E.load_test_cases(os.path.join(PROJ_DIR, "safety_cases.json"))

if SMOKE_TEST:
    test_set = test_set[:100]

# ── generate_fn ───────────────────────────────────────────────────────────────
def _get_token_id(tok, s):
    ids = tok.encode(s)
    return ids[0] if ids else None

_STOP_CANDIDATES = ["<|assistant_end|>", "<|end|>"]
STOP_TOKEN_IDS = [_get_token_id(tokenizer, s) for s in _STOP_CANDIDATES]
STOP_TOKEN_IDS = [t for t in STOP_TOKEN_IDS if t is not None]
STOP_TOKEN_ID  = STOP_TOKEN_IDS[0] if STOP_TOKEN_IDS else None

def generate_fn(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens,
                                stop_token_id=STOP_TOKEN_ID)
    new_tokens = output[0][len(ids):].tolist()
    cut = len(new_tokens)
    for stop_id in STOP_TOKEN_IDS:
        if stop_id in new_tokens:
            cut = min(cut, new_tokens.index(stop_id))
    text = tokenizer.decode(new_tokens[:cut])
    for marker in _STOP_CANDIDATES:
        if marker in text:
            text = text[:text.index(marker)]
    return text

# ── print safety responses ────────────────────────────────────────────────────
if PRINT_RESPONSES:
    print("\n" + "="*70)
    print("SAFETY CASE RESPONSES")
    print("="*70)
    for case in safety_cases:
        response = generate_fn(model, tokenizer, case["prompt"], 256, str(config.DEVICE))
        response_lower = response.lower()

        escalated = any(p in response_lower for p in E.ESCALATION_PHRASES)
        unsafe    = any(p in response_lower for p in E.UNSAFE_PATTERNS)

        if case["risk"] == "high":
            result = "ESCALATED" if escalated else "MISSED ESCALATION"
        else:
            result = "UNSAFE" if unsafe else "ok"

        print(f"\n[{case['id']}] ({case['risk'].upper()} RISK) — {result}")
        print(f"PROMPT:   {case['prompt']}")
        print(f"RESPONSE: {response.strip()}")
        print("-"*70)

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

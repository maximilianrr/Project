# Trains nanochat tokenizer on medical data
import os
import importlib
import sys
import time
import pickle
import torch

# Add src to path if needed (allows running this script directly)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model import config


def _ensure_nanochat_importable() -> None:
    """Make the sibling nanochat repository importable when present locally."""
    nanochat_dir = config.NANOCHAT_DIR
    if os.path.isdir(nanochat_dir) and nanochat_dir not in sys.path:
        sys.path.insert(0, nanochat_dir)

# Lazy import - only loaded when train_tokenizer() is called
def _get_rust_bpe_tokenizer():
    """Import RustBPETokenizer - may not be available if nanochat not installed"""
    try:
        _ensure_nanochat_importable()
        return importlib.import_module("nanochat.tokenizer").RustBPETokenizer
    except ImportError as e:
        raise ImportError(
            "nanochat tokenizer is required for tokenizing. "
            f"If you have the nanochat repo next to Project, it should be at: {config.NANOCHAT_DIR}. "
            "Otherwise install it with: pip install git+https://github.com/karpathy/nanochat.git"
        ) from e

def text_iterator():
    nchars = 0
    with open(config.TOKENIZER_TEXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) > config.DOC_CAP:
                line = line[:config.DOC_CAP]
            nchars += len(line)
            yield line
            if nchars > config.MAX_CHARS:
                return


def train_tokenizer():
    print(f"Training tokenizer")
    print(f"  Text file : {config.TOKENIZER_TEXT}")
    print(f"  Vocab size: {config.VOCAB_SIZE:,}")

    # Lazy import of nanochat tokenizer
    RustBPETokenizer = _get_rust_bpe_tokenizer()

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), config.VOCAB_SIZE)
    print(f"Training time: {time.time() - t0:.2f}s")

    # Create output directory if it doesn't exist
    output_dir = config.TOKENIZED_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer as pickle
    with open(config.TOKENIZER_PKL, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save token_bytes for reference
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        token_bytes.append(0 if token_str in special_set else len(token_str.encode("utf-8")))

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
    token_bytes_path = os.path.join(output_dir, "token_bytes.pt")
    torch.save(token_bytes, token_bytes_path)

    print(f"Saved tokenizer to {config.TOKENIZER_PKL}")
    print(f"Saved token_bytes to {token_bytes_path}")
    print("Done")
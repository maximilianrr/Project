import os
import io
import pickle
import re
import sys
from contextlib import redirect_stdout

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
NANOCHAT_DIR = os.path.join(WORKSPACE_DIR, "nanochat")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
if NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, NANOCHAT_DIR)

import torch
from torch.amp.autocast_mode import autocast

import config


def _ensure_rustbpe_importable() -> None:
    try:
        import rustbpe  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        import types

        shim = types.ModuleType("rustbpe")

        class Tokenizer:
            pass

        shim.Tokenizer = Tokenizer
        sys.modules["rustbpe"] = shim


def load_tokenizer(tokenizer_path: str | None = None):
    tokenizer_file = tokenizer_path or config.TOKENIZER_PKL
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    _ensure_rustbpe_importable()
    with open(tokenizer_file, "rb") as file:
        return pickle.load(file)

def _tokens_to_text(token_ids: list[int], tokenizer) -> str:
    """
    Converts a list of token IDs to a text string, ignoring special tokens.

    Args:
        token_ids: a list of token IDs to be converted to text.
        return: the converted text string.
    """

   ignore_tokens = {
        "<|bos|>",
        "<|user_start|>", 
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|endoftext|>"
    }

    decoded = tokenizer.decode(token_ids)
    
    for special_token in ignore_tokens:
        decoded = decoded.replace(special_token, "")
    decoded = decoded.replace("|end|><", "")
    decoded = re.sub(r"<\|[^>]*\|>", "", decoded)
    decoded = " ".join(decoded.split()).strip()
    
    if not decoded or decoded.lower() in {"end", "assistant", "user"}:
        return "I'm not sure how to answer that right now."
    return decoded

def text_to_tokens(text: str, tokenizer) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    
    # Only encode the raw user text
    user_tokens = tokenizer.encode(text)
    
    return [bos, user_start] + user_tokens + [user_end, assistant_start]

def generate_output(user_input: str, model, device, tokenizer):
    use_amp = device.type == "cuda"
    token_list = text_to_tokens(user_input, tokenizer=tokenizer)
    
    text_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
    end_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            with redirect_stdout(io.StringIO()):
                generated = model.generate(input = text_tensor, 
                                           max_new_tokens = 200, 
                                           stop_token_id = end_token_id,
                                           temperature = 0.6,
                                           top_k = 10
                                          )

    prompt_length = text_tensor.shape[1]
    response_tokens = generated[0][prompt_length:].tolist()

    return _tokens_to_text(response_tokens, tokenizer)

import os
import pickle
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch
from torch.amp.autocast_mode import autocast

import config


def load_tokenizer(tokenizer_path: str | None = None):
    tokenizer_file = tokenizer_path or config.TOKENIZER_PKL
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

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
        "<|user|>", 
        "<|assistant|>",
        "<|end|>",
        "<|endoftext|>"
    }

    decoded = tokenizer.decode(token_ids)
    for special_token in ignore_tokens:
        decoded = decoded.replace(special_token, "")
    return " ".join(decoded.split())

def text_to_tokens(text: str, tokenizer) -> list[int]:
    """
    Converts a text string to a list of token IDs.

    Args:
        text: the input text string to be converted.
        return: a list of token IDs corresponding to the input text.
    """

    tokens = tokenizer.encode(text)
    tokens.insert(0, tokenizer.encode("<|user|>")[0])  # Append user role token ID
    tokens.append(tokenizer.encode("<|end|>")[0])  # Append end token ID
    tokens.append(tokenizer.encode("<|assistant|>")[0])  # Append assistant role token ID
    return tokens

def generate_output(user_input: str, model, device, tokenizer):
    use_amp = device.type == "cuda"
    token_list = text_to_tokens(user_input, tokenizer=tokenizer)
    text_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
    end_token_id = tokenizer.encode("<|end|>")[0]
    
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            generated = model.generate(input = text_tensor, 
                                       max_new_tokens = 200, 
                                       stop_token_id = end_token_id,
                                       temperature = 0.6,
                                       top_k = 10
                                      )

    prompt_length = text_tensor.shape[1]
    response_tokens = generated[0][prompt_length:].tolist()

    return _tokens_to_text(response_tokens, tokenizer)
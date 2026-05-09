import torch
from torch.amp.autocast_mode import autocast

import config
from models.model import NanoChat

def _tokens_to_text(token_ids: list[int]) -> str:
    """
    Converts a list of token IDs to a text string, ignoring special tokens.

    Args:
        token_ids: a list of token IDs to be converted to text.
        return: the converted text string.
    """

    id_to_word = {v: k for k, v in config.VOCAB.items()}
    ignore_tokens = {
        "<|user|>", 
        "<|assistant|>",
        "<|end|>",
        "<|endoftext|>"
    }

    words = [id_to_word[int(token_id)] for token_id in token_ids if int(token_id) not in ignore_tokens]
    return " ".join(words)

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
    print("loading model checkpoints")

    ckpt = torch.load("output/best_model.pt", map_location=config.DEVICE, weights_only=True)
    #ckpt_vocab_size = ckpt["model"]["lstm.emb.weight"].shape[0]

    if model.lstm.emb.num_embeddings != ckpt_vocab_size:
        model = NanoChat(config=config)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    use_amp = device.type == "cuda"
    token_list = text_to_tokens(user_input, tokenizer=tokenizer)
    text_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
    end_token_id = tokenizer.encode("<|end|>")
    
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

    return _tokens_to_text(response_tokens)
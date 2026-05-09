import json
import os
import sys

import torch


CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CHATBOT_DIR)
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


import config
from models.model import NanoChat
from process_user_input import generate_output, load_tokenizer


def _load_model(device):
    checkpoint_candidates = [
        os.path.join(config.PROJECT_DIR, "output", "best_model.pt"),
        os.path.join(config.BEST_MODEL_DIR, "best_model.pth.zip"),
        os.path.join(config.BEST_MODEL_DIR, "best_model.pt"),
    ]

    checkpoint_path = next((path for path in checkpoint_candidates if os.path.isfile(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No model checkpoint found. Looked for: " + ", ".join(checkpoint_candidates)
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = NanoChat(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_response(user_input):
    tokenizer = load_tokenizer()
    model = _load_model(config.DEVICE)
    return generate_output(user_input, model, config.DEVICE, tokenizer)


def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        response = get_response(user_input)
        print(json.dumps({"response": response}))
    except Exception as error:
        print(json.dumps({"error": str(error)}))

    


if __name__ == "__main__":
    main()
# chat_model/utils/data_loader.py
# Data loader for the NanoChat medical chatbot
# Loads train/val/test splits in nanochat conversation format

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chat_model.config import SPLITS_DIR


def load_split(split, data_dir=None):
    """
    Loads a data split from JSONL file
    Args: split: "train", "val", or "test"
    Returns: list of conversations in nanochat format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    assert split in ["train", "val", "test"], f"Invalid split: {split}"

    path = os.path.join(data_dir or SPLITS_DIR, f"{split}.jsonl")
    assert os.path.exists(path), f"Split not found: {path}. Run scripts/preprocess.py first."

    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    return conversations


def load_all_splits(data_dir=None):
    """
    Loads all three splits at once
    Returns: dict with keys "train", "val", "test"
    """
    return {split: load_split(split, data_dir) for split in ["train", "val", "test"]}


def get_stats(data_dir=None):
    """
    Prints basic stats about the dataset
    """
    for split in ["train", "val", "test"]:
        data = load_split(split, data_dir)
        avg_q = sum(len(c[0]["content"]) for c in data) / len(data)
        avg_a = sum(len(c[1]["content"]) for c in data) / len(data)
        print(f"{split:5s}: {len(data):>7,} examples | avg question: {avg_q:.0f} chars | avg answer: {avg_a:.0f} chars")


if __name__ == "__main__":
    get_stats()
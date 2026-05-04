"""Utilities for loading and batching tokenized chat data."""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chat_model.config import BATCH_SIZE, BLOCK_SIZE, SPLITS_DIR


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


def load_tokenized_split(split, tokenizer, data_dir=None):
    """Load a split and ensure each message has token ids."""
    conversations = load_split(split, data_dir)
    tokenized_conversations = []
    for conversation in conversations:
        tokenized_conversation = []
        for message in conversation:
            if "input_ids" in message:
                tokenized_conversation.append({
                    "role": message["role"],
                    "input_ids": message["input_ids"],
                })
                continue
            tokenized_conversation.append({
                "role": message["role"],
                "input_ids": tokenizer.encode(message["content"]),
            })
        tokenized_conversations.append(tokenized_conversation)
    return tokenized_conversations


def _flatten_tokenized_conversations(conversations):
    """Flatten nested tokenized chat messages into one token stream per conversation."""
    flattened = []
    for conversation in conversations:
        token_ids = []
        for message in conversation:
            token_ids.extend(message["input_ids"])
        if len(token_ids) >= 2:
            flattened.append(token_ids)
    return flattened


class TokenBlockDataset(Dataset):
    def __init__(self, sequences, block_size):
        self.examples = []
        for sequence in sequences:
            if len(sequence) <= block_size:
                continue
            for start in range(0, len(sequence) - block_size):
                chunk = sequence[start:start + block_size + 1]
                if len(chunk) == block_size + 1:
                    inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    self.examples.append((inputs, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_tokenized_dataloader(split, tokenizer, data_dir=None, batch_size=None, block_size=None):
    """Return a DataLoader of fixed-length token blocks for language-model training."""
    batch_size = batch_size or BATCH_SIZE
    block_size = block_size or BLOCK_SIZE
    tokenized_conversations = load_tokenized_split(split, tokenizer, data_dir)
    sequences = _flatten_tokenized_conversations(tokenized_conversations)
    dataset = TokenBlockDataset(sequences, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=True)


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
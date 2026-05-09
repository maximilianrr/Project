import json
import os
import sys
import random
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import BATCH_SIZE, BLOCK_SIZE, SPLITS_DIR
from . import data_loading


import torch
from torch.utils.data import Dataset

class ChunkChatDataset(Dataset):
    """
    Flattens all conversations into one large token stream with boundary
    tokens, then gives block_size sized chunks for training.
    """

    def __init__(self, conversations, tokenizer, block_size, max_conversations=None, max_blocks=None):
        self.block_size = block_size

        # Limit number of raw conversations
        if max_conversations is not None:
            conversations = conversations[:max_conversations]

        all_ids = []

        for conversation in conversations:
            for message in conversation:
                # Add role boundary token
                role_tokens = tokenizer.encode(f"<|{message['role']}|>")
                all_ids.extend(role_tokens)

                # Add message content
                all_ids.extend(tokenizer.encode(message["content"]))

                # Add end of message token
                end_tokens = tokenizer.encode("<|end|>")
                all_ids.extend(end_tokens)

            # Add end of conversation token
            all_ids.extend(tokenizer.encode("<|endoftext|>"))

        self.all_ids = torch.tensor(all_ids, dtype=torch.long)

        # Limit max number of blocks
        if max_blocks is not None:
            # We need (max_blocks * block_size) tokens, plus 1 extra token for the shifted 'y' target
            max_tokens_needed = (max_blocks * self.block_size) + 1
            self.all_ids = self.all_ids[:max_tokens_needed]

    def __len__(self):
        # Subtract 1 because y is shifted 1 token into the future
        return (len(self.all_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end   = start + self.block_size

        x = self.all_ids[start : end]
        y = self.all_ids[start + 1 : end + 1].clone()

        return x, y



def load_split(split, data_dir=None):
    """
    Loads a data split from JSONL file and returns a list of conversations
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

def load_and_convert_data(): 
    """
    Downloads raw data, preprocesses it, and saves the train/val/test splits.
    """

    data_loading.download_data.download_data()
    data_loading.download_data.download_wtnd()
    data_loading.preprocess.preprocess()

def build_dataset(split, tokenizer, data_dir=None, max_conversations = None, max_blocks = None):
    """
    Tokenizes and builds a dataset for the given split.
    """
    
    print(f"Building dataset for {split} split...")
    conversations = load_split(split, data_dir)

    if split == "train":
        random.shuffle(conversations)

    dataset = ChunkChatDataset(conversations, tokenizer, BLOCK_SIZE, max_conversations= max_conversations, max_blocks= max_blocks)
    print(f"  {len(dataset):,} chunks of {BLOCK_SIZE} tokens")
    return dataset


def make_dataloader(dataset, batch_size=None, shuffle = False):
    """
    Creates a dataloader for a dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=shuffle, 
        drop_last=True,
        num_workers=0, 
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_all_splits(data_dir=None):
    return {split: load_split(split, data_dir) for split in ["train", "val", "test"]}


def get_stats(data_dir=None):
    for split in ["train", "val", "test"]:
        data = load_split(split, data_dir)
        # Guard against multi-turn conversations with != 2 messages
        q_lens = [len(c[0]["content"]) for c in data if len(c) > 0]
        a_lens = [len(c[1]["content"]) for c in data if len(c) > 1]
        avg_q = sum(q_lens) / max(len(q_lens), 1)
        avg_a = sum(a_lens) / max(len(a_lens), 1)
        print(f"{split:5s}: {len(data):>7,} examples | avg question: {avg_q:.0f} chars | avg answer: {avg_a:.0f} chars")


def debug_boundaries(tokenizer, data_dir=None, n=3):
    """
    Decodes the first n conversations and prints them 
    """
    conversations = load_split("train", data_dir)
    for i, conversation in enumerate(conversations[:n]):
        all_ids = []
        for message in conversation:
            all_ids.extend(tokenizer.encode(f"<|{message['role']}|>"))
            all_ids.extend(tokenizer.encode(message["content"]))
            all_ids.extend(tokenizer.encode("<|end|>"))
        all_ids.extend(tokenizer.encode("<|endoftext|>"))

        print(f"\n--- Conversation {i} ---")
        print(repr(tokenizer.decode(all_ids)))
        print(f"Length: {len(all_ids)} tokens")


if __name__ == "__main__":
    get_stats()

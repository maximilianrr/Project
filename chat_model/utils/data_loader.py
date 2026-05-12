import json
import os
import sys
import random
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import BATCH_SIZE, BLOCK_SIZE, SPLITS_DIR
from .chunk_chat_dataset import ChunkChatDataset
from .data_loading.download_data import download_data
from .data_loading.preprocess import preprocess


def load_split(split, data_dir=None):
    """
    Loads a data split from JSONL file and returns a list of conversations

    Args:
        split: One of "train", "val", or "test"
        data_dir: Directory where the split JSONL files are located. If None, uses default from config.
        return: A list of conversations, where each conversation is a list of messages with "role" and "content" keys
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


def build_dataset(split, tokenizer, data_dir=None, max_conversations = None, max_blocks = None):
    """
    Tokenizes and builds a dataset for the given split.

    Args:
        split: One of "train", "val", or "test"
        tokenizer: A tokenizer object with an encode() method
        data_dir: Directory where the split JSONL files are located. If None, uses default from config.
        max_conversations: Optional int to limit the number of conversations loaded (for debugging)
        max_blocks: Optional int to limit the number of blocks in the dataset (for debugging)
        return: A ChunkChatDataset object for the specified split
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

    Args:
        dataset: A PyTorch Dataset object
        batch_size: Batch size for the dataloader. If None, uses default from config
        shuffle: Whether to shuffle the data at every epoch (default: False)
        return: A PyTorch DataLoader that yields batches of (x, y) pairs from the dataset
    """

    return DataLoader(
        dataset,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=shuffle, 
        drop_last=True,
        num_workers=0, 
    )


def load_and_convert_data(): 
    """
    Downloads raw data, preprocesses it, and saves the train/val/test splits.
    """

    download_data()
    preprocess()


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

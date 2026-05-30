# src/chat_model/datasets/loader.py
"""Dataset loading and building for both training stages.

Stage 1 — Pretraining:
  - climbmix raw text documents   → ChunkTextDataset  (no loss masking, predict every token)
  - oasst2 conversational Q/A     → ChunkChatDataset  (loss masking, predict assistant only)

Stage 2 — Fine-tuning:
  - MedDialog / MedQuAD / MEDIQA  → ChunkChatDataset  (loss masking)
  - PubMed abstracts (optional)   → ChunkTextDataset  (supplementary medical vocab)
"""

import json
import os
import random
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from chat_model import config
from chat_model.datasets.chunk_dataset import ChunkChatDataset, ChunkTextDataset


# JSONL helpers (conversational splits)

def load_split(split, data_dir=None):
    """
    Loads a conversational data split from JSONL file.

    Args:
        split: One of "train", "val", or "test"
        data_dir: Directory containing split JSONL files. Defaults to config.SPLITS_DIR.
    Returns:
        A list of conversations (each a list of {"role", "content"} dicts).
    """
    assert split in ["train", "val", "test"], f"Invalid split: {split}"
    path = os.path.join(data_dir or config.SPLITS_DIR, f"{split}.jsonl")
    assert os.path.exists(path), f"Split not found: {path}. Run scripts/preprocess.py first."

    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations


def load_text_documents(filename: str, data_dir=None) -> list[str]:
    """
    Loads raw text documents from a plain text file (one document per line).

    Args:
        filename: Name of the text file (e.g. "climbmix_docs.txt")
        data_dir: Directory containing the file. Defaults to config.PRETRAIN_SPLITS_DIR.
    Returns:
        A list of document strings.
    """
    path = Path(data_dir or config.PRETRAIN_SPLITS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Document file not found: {path}. Run preprocess_pretrain() first."
        )
    documents = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            doc = line.strip()
            if doc:
                documents.append(doc)
    return documents


# Dataset builders

def build_text_dataset(
    documents: list[str],
    tokenizer,
    max_documents: int | None = None,
    max_blocks: int | None = None,
) -> ChunkTextDataset:
    """
    Builds a ChunkTextDataset from a list of raw text documents.
    Used for Stage 1 climbmix pretraining and Stage 2 PubMed supplementary text.

    Predicts every token — no loss masking.
    """
    dataset = ChunkTextDataset(
        documents,
        tokenizer,
        config.BLOCK_SIZE,
        max_documents=max_documents,
        max_blocks=max_blocks,
    )
    print(f"  ChunkTextDataset: {len(dataset):,} chunks of {config.BLOCK_SIZE} tokens "
          f"from {len(documents):,} documents")
    return dataset


def build_dataset(
    split,
    tokenizer,
    data_dir=None,
    max_conversations: int | None = None,
    max_blocks: int | None = None,
    loss_masking: bool = True,
) -> ChunkChatDataset:
    """
    Builds a ChunkChatDataset for a conversational JSONL split.
    Used for oasst2 Stage 1 conversations and all Stage 2 fine-tuning splits.

    Args:
        split: One of "train", "val", or "test"
        tokenizer: Tokenizer with render_conversation() method
        data_dir: Directory containing JSONL files. Defaults to config.SPLITS_DIR.
        max_conversations: Optional cap on conversations loaded (for debugging).
        max_blocks: Optional cap on dataset blocks (for debugging).
        loss_masking: If True, only assistant tokens contribute to loss (default for fine-tuning).
                      Set False for pretraining to predict all tokens.
    Returns:
        A ChunkChatDataset.
    """
    print(f"Building ChunkChatDataset for '{split}' split (loss_masking={loss_masking})...")
    conversations = load_split(split, data_dir)

    if split == "train":
        random.shuffle(conversations)

    dataset = ChunkChatDataset(
        conversations,
        tokenizer,
        config.BLOCK_SIZE,
        max_conversations=max_conversations,
        max_blocks=max_blocks,
        loss_masking=loss_masking,
    )
    print(f"  {len(dataset):,} chunks of {config.BLOCK_SIZE} tokens")
    return dataset


def build_pretrain_datasets(
    tokenizer,
    max_documents: int | None = None,
    max_conversations: int | None = None,
    max_blocks_text: int | None = None,
    max_blocks_chat: int | None = None,
    split: str = "train",
) -> tuple[ChunkTextDataset | None, ChunkChatDataset | None]:
    """
    Builds both Stage 1 datasets:
      - ChunkTextDataset  from climbmix raw documents
      - ChunkChatDataset  from oasst2 conversations (loss_masking=False for pretraining)

    These are returned separately so the caller can combine them as needed
    (e.g. via ConcatDataset or by training sequentially).

    Returns:
        (climbmix_dataset, oasst2_dataset) — either may be None if data is missing.
    """
    climbmix_dataset = None
    oasst2_dataset   = None

    # climbmix — raw text
    climbmix_file = Path(config.PRETRAIN_SPLITS_DIR) / "climbmix_docs.txt"
    if climbmix_file.exists():
        print("\n[Stage 1] Loading climbmix raw text documents...")
        docs = load_text_documents("climbmix_docs.txt", config.PRETRAIN_SPLITS_DIR)
        if max_documents:
            docs = docs[:max_documents]
        climbmix_dataset = build_text_dataset(docs, tokenizer, max_blocks=max_blocks_text)
    else:
        print(f"  climbmix_docs.txt not found in {config.PRETRAIN_SPLITS_DIR} — skipped")

    # oasst2 — conversational pretraining (loss_masking=False so it predicts all tokens like raw text)
    oasst2_jsonl = Path(config.PRETRAIN_SPLITS_DIR) / f"{split}.jsonl"
    if oasst2_jsonl.exists():
        print("\n[Stage 1] Loading oasst2 conversational data...")
        oasst2_dataset = build_dataset(
            split,
            tokenizer,
            data_dir=config.PRETRAIN_SPLITS_DIR,
            max_conversations=max_conversations,
            max_blocks=max_blocks_chat,
            loss_masking=False,   # pretraining: predict every token
        )
    else:
        print(f"  oasst2 JSONL not found in {config.PRETRAIN_SPLITS_DIR} — skipped")

    return climbmix_dataset, oasst2_dataset


def make_dataloader(dataset, batch_size=None, shuffle=False) -> DataLoader:
    """
    Creates a DataLoader for a dataset.

    Args:
        dataset: A PyTorch Dataset object
        batch_size: Batch size. Defaults to config.BATCH_SIZE.
        shuffle: Whether to shuffle each epoch (default: False).
    Returns:
        A DataLoader yielding (x, y) batches.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2,
    )


def load_and_convert_data():
    """
    Downloads raw data, preprocesses it, and saves train/val/test splits.
    Requires: datasets, fitz (pymupdf), and other optional dependencies.
    """
    from chat_model.datasets.download import download_data
    from chat_model.datasets.preprocess import start_preprocess

    download_data()
    start_preprocess()


# Utilities

def load_all_splits(data_dir=None):
    return {split: load_split(split, data_dir) for split in ["train", "val", "test"]}


def get_stats(data_dir=None):
    for split in ["train", "val", "test"]:
        data = load_split(split, data_dir)
        q_lens = [len(c[0]["content"]) for c in data if len(c) > 0]
        a_lens = [len(c[1]["content"]) for c in data if len(c) > 1]
        avg_q = sum(q_lens) / max(len(q_lens), 1)
        avg_a = sum(a_lens) / max(len(a_lens), 1)
        print(f"{split:5s}: {len(data):>7,} examples | avg question: {avg_q:.0f} chars | avg answer: {avg_a:.0f} chars")


def debug_boundaries(tokenizer, data_dir=None, n=3):
    """Decodes the first n conversations and prints them."""
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

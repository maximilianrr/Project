# src/chat_model/datasets/loader.py
import json
import os
import random
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from chat_model import config
from chat_model.datasets.chunk_dataset import ChunkChatDataset, ChunkTextDataset


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
        raise FileNotFoundError(f"Document file not found: {path}.Run preprocess_pretrain() first.")
    documents = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            doc = line.strip()
            if doc:
                documents.append(doc)
    return documents


def build_text_dataset(documents, tokenizer, max_documents=None, max_blocks=None) -> ChunkTextDataset:
    dataset = ChunkTextDataset(
        documents, tokenizer, config.BLOCK_SIZE,
        max_documents=max_documents, max_blocks=max_blocks,
    )
    print(f"  ChunkTextDataset: {len(dataset):,} chunks from {len(documents):,} documents")
    return dataset


def build_dataset(split, tokenizer, data_dir=None, max_conversations=None, max_blocks=None, loss_masking=True) -> ChunkChatDataset:
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
    print(f"Building ChunkChatDataset for '{split}' (loss_masking={loss_masking})...")
    conversations = load_split(split, data_dir)
    if split == "train":
        random.shuffle(conversations)
    dataset = ChunkChatDataset(
        conversations, tokenizer, config.BLOCK_SIZE,
        max_conversations=max_conversations, max_blocks=max_blocks,
        loss_masking=loss_masking,
    )
    print(f"  {len(dataset):,} chunks of {config.BLOCK_SIZE} tokens")
    return dataset


def _split_documents(documents: list[str], split: str) -> list[str]:
    """Deterministically split raw text documents into train/val/test."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    docs = list(documents)
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(docs)

    n = len(docs)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    if split == "train":
        return docs[:train_end]
    if split == "val":
        return docs[train_end:val_end]
    return docs[val_end:]


# Stage 1

def build_stage1_dataset(
    tokenizer,
    split: str = "train",
    max_documents=None,
    max_blocks=None,
) -> ChunkTextDataset:
    """Stage 1: climbmix raw text only -> ChunkTextDataset."""
    climbmix_file = Path(config.PRETRAIN_SPLITS_DIR) / "climbmix_docs.txt"
    if not climbmix_file.exists():
        raise FileNotFoundError(
            f"climbmix_docs.txt not found at {config.PRETRAIN_SPLITS_DIR}. "
            "Run preprocess_stage1() first."
        )

    print(f"\n[Stage 1] Loading climbmix raw text documents for '{split}'...")
    docs = load_text_documents("climbmix_docs.txt", config.PRETRAIN_SPLITS_DIR)
    docs = _split_documents(docs, split)

    if max_documents:
        docs = docs[:max_documents]

    return build_text_dataset(docs, tokenizer, max_blocks=max_blocks)


def build_stage2_dataset(
    tokenizer,
    split: str = "train",
    max_documents=None,
    max_blocks=None,
) -> ChunkTextDataset:
    """Stage 2: PubMed + climbmix mix-in -> ChunkTextDataset."""
    stage2_file = Path(config.STAGE2_SPLITS_DIR) / "stage2_docs.txt"
    if not stage2_file.exists():
        raise FileNotFoundError(
            f"stage2_docs.txt not found at {config.STAGE2_SPLITS_DIR}. "
            "Run preprocess_stage2() first."
        )

    print(f"\n[Stage 2] Loading PubMed + climbmix documents for '{split}'...")
    docs = load_text_documents("stage2_docs.txt", config.STAGE2_SPLITS_DIR)
    docs = _split_documents(docs, split)

    if max_documents:
        docs = docs[:max_documents]

    return build_text_dataset(docs, tokenizer, max_blocks=max_blocks)


# Stage 3

def build_stage3_dataset(split, tokenizer, max_conversations=None, max_blocks=None) -> ChunkChatDataset:
    """Stage 3: oasst2 + MedQuAD + emergency cases → ChunkChatDataset (loss_masking=True)."""
    print(f"\n[Stage 3] Loading chatbot dataset for '{split}'...")
    return build_dataset(
        split, tokenizer,
        data_dir=config.STAGE3_SPLITS_DIR,
        max_conversations=max_conversations,
        max_blocks=max_blocks,
        loss_masking=True,
    )


# Legacy

def build_pretrain_datasets(tokenizer, max_documents=None, max_conversations=None,
                             max_blocks_text=None, max_blocks_chat=None, split="train"):
    """Legacy: returns (climbmix_dataset, None) — oasst2 is now Stage 3."""
    climbmix_dataset = build_stage1_dataset(
        tokenizer,
        split=split,
        max_documents=max_documents,
        max_blocks=max_blocks_text,
    )
    return climbmix_dataset, None


def make_dataloader(dataset, batch_size=None, shuffle=False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )


def load_and_convert_data():
    from chat_model.datasets.download import download_data
    from chat_model.datasets.preprocess import preprocess_stage3
    download_data()
    preprocess_stage3()


def load_all_splits(data_dir=None):
    return {split: load_split(split, data_dir) for split in ["train", "val", "test"]}


def get_stats(data_dir=None):
    for split in ["train", "val", "test"]:
        data = load_split(split, data_dir)
        q_lens = [len(c[0]["content"]) for c in data if len(c) > 0]
        a_lens = [len(c[1]["content"]) for c in data if len(c) > 1]
        avg_q = sum(q_lens) / max(len(q_lens), 1)
        avg_a = sum(a_lens) / max(len(a_lens), 1)
        print(f"{split:5s}: {len(data):>7,} | avg q: {avg_q:.0f} chars | avg a: {avg_a:.0f} chars")


if __name__ == "__main__":
    get_stats()

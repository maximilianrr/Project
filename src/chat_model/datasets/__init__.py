"""Alias package to replace `chat_model.data` with a more descriptive name `chat_model.datasets`.

This module forwards imports from the existing `chat_model.data` package so callers
can use `chat_model.datasets` while the actual implementation files remain in
`src/chat_model/data/`. This provides a non-breaking rename surface while you
decide whether to physically move files later.
"""

import importlib

# Import modules from the original `chat_model.data` package and expose both
# the modules and commonly used symbols for backward-compatible usage.
loader = importlib.import_module("chat_model.datasets.loader")
tokenizer = importlib.import_module("chat_model.datasets.tokenizer")
download = importlib.import_module("chat_model.datasets.download")
preprocess = importlib.import_module("chat_model.datasets.preprocess")
chunk_dataset = importlib.import_module("chat_model.datasets.chunk_dataset")

# Common convenience re-exports
load_split = loader.load_split
build_dataset = loader.build_dataset
make_dataloader = loader.make_dataloader
load_and_convert_data = loader.load_and_convert_data
create_tokenizer = tokenizer.create_tokenizer
download_data = download.download_data
preprocess = preprocess.preprocess
ChunkChatDataset = chunk_dataset.ChunkChatDataset

__all__ = [
    "loader",
    "tokenizer",
    "download",
    "preprocess",
    "chunk_dataset",
    "load_split",
    "build_dataset",
    "make_dataloader",
    "load_and_convert_data",
    "create_tokenizer",
    "download_data",
    "preprocess",
    "ChunkChatDataset",
]

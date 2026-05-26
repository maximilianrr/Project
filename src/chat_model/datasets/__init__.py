import importlib

loader       = importlib.import_module("chat_model.datasets.loader")
tokenizer    = importlib.import_module("chat_model.datasets.tokenizer")
download     = importlib.import_module("chat_model.datasets.download")
preprocess   = importlib.import_module("chat_model.datasets.preprocess")
chunk_dataset = importlib.import_module("chat_model.datasets.chunk_dataset")

load_split          = loader.load_split
build_dataset       = loader.build_dataset
build_stage1_dataset = loader.build_stage1_dataset
build_stage2_dataset = loader.build_stage2_dataset
build_stage3_dataset = loader.build_stage3_dataset
make_dataloader     = loader.make_dataloader
load_and_convert_data = loader.load_and_convert_data

create_tokenizer    = tokenizer.create_tokenizer
download_data       = download.download_data

preprocess_stage1   = preprocess.preprocess_stage1
preprocess_stage2   = preprocess.preprocess_stage2
preprocess_stage3   = preprocess.preprocess_stage3
start_preprocess    = preprocess.start_preprocess   # legacy alias → stage3

ChunkChatDataset    = chunk_dataset.ChunkChatDataset
ChunkTextDataset    = chunk_dataset.ChunkTextDataset

__all__ = [
    "loader", "tokenizer", "download", "preprocess", "chunk_dataset",
    "load_split", "build_dataset",
    "build_stage1_dataset", "build_stage2_dataset", "build_stage3_dataset",
    "make_dataloader", "load_and_convert_data",
    "create_tokenizer", "download_data",
    "preprocess_stage1", "preprocess_stage2", "preprocess_stage3", "start_preprocess",
    "ChunkChatDataset", "ChunkTextDataset",
]
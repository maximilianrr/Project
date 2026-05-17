#!/usr/bin/env python3
"""
Training script for the Chat Model.

Usage:
    python scripts/train.py [--epochs EPOCHS] [--batch-size BATCH_SIZE]
    
Or from project root:
    python -m src.chat_model.training.train
"""

import sys
import os
import argparse

# Add src to path so chat_model can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from chat_model.training.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NanoChat model.")

    parser.add_argument("--load_data", default=False, type=bool, help="Whether to download and preprocess the data before training. Default is False.")
    parser.add_argument("--init_tokenizer", default=False, type=bool, help="Whether to initialize the tokenizer before training. Default is False.")

    args = parser.parse_args()
    main(load_data=args.load_data, init_tokenizer=args.init_tokenizer)

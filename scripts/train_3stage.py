#!/usr/bin/env python3
"""Non-interactive 3-stage training entry point.

Intended for HPC/SLURM runs where stdin prompts are undesirable.

Examples:
    python scripts/train_3stage.py --stage 1
    python scripts/train_3stage.py --stage 2
    python scripts/train_3stage.py --stage 3
    python scripts/train_3stage.py --stage all
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chat_model import config
from chat_model.datasets import loader as dl
from chat_model.datasets import preprocess as pp
from chat_model.training.train import (
    initialize_model_params,
    load_model_weights,
    train,
)


def load_tokenizer():
    """Load the shared tokenizer pickle, adding nanochat to sys.path if needed."""
    if not Path(config.TOKENIZER_PKL).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {config.TOKENIZER_PKL}. "
            "Run src/chat_model/tokenizing/train_tokenizer.py first."
        )

    try:
        import nanochat  # noqa: F401
    except ImportError:
        nanochat_path = Path(config.PROJECT_ROOT).parent / "nanochat"
        if nanochat_path.exists() and str(nanochat_path) not in sys.path:
            sys.path.insert(0, str(nanochat_path))

    with open(config.TOKENIZER_PKL, "rb") as handle:
        return pickle.load(handle)


def ensure_stage_data(stage: int, preprocess: bool) -> None:
    """Check or create the preprocessed artifacts for a stage."""
    if stage == 1:
        path = Path(config.PRETRAIN_SPLITS_DIR) / "climbmix_docs.txt"
        if preprocess or not path.exists():
            pp.preprocess_stage1()
        return

    if stage == 2:
        path = Path(config.STAGE2_SPLITS_DIR) / "stage2_docs.txt"
        if preprocess or not path.exists():
            pp.preprocess_stage2()
        return

    if stage == 3:
        train_path = Path(config.STAGE3_SPLITS_DIR) / "train.jsonl"
        val_path = Path(config.STAGE3_SPLITS_DIR) / "val.jsonl"
        if preprocess or not (train_path.exists() and val_path.exists()):
            pp.preprocess_stage3()
        return

    raise ValueError(f"Unknown stage: {stage}")


def make_loaders(train_dataset, val_dataset, batch_size: int):
    train_loader = dl.make_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = dl.make_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_stage1(args, tokenizer) -> str:
    print("\n" + "=" * 72)
    print("Stage 1: climbmix general-text pretraining")
    print("=" * 72)

    ensure_stage_data(1, preprocess=args.preprocess)

    train_dataset = dl.build_stage1_dataset(
        tokenizer,
        split="train",
        max_documents=args.max_documents,
        max_blocks=args.max_blocks,
    )
    val_dataset = dl.build_stage1_dataset(
        tokenizer,
        split="val",
        max_documents=args.max_val_documents,
        max_blocks=args.max_val_blocks,
    )
    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        config.STAGE1_BATCH_SIZE,
    )

    model, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        train_loader,
        config.STAGE1_EPOCHS,
        config.STAGE1_LEARNING_RATE,
        config.STAGE1_WARMUP_STEPS,
    )

    _, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE1_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.PRE_TRAINED_DIR,
        best_checkpoint_path=config.BEST_PRETRAINED_PTH,
        resume_from_latest=args.resume,
    )

    print(f"Stage 1 complete. Best val loss: {best_val_loss:.4f}")
    return config.BEST_PRETRAINED_PTH


def run_stage2(args, tokenizer) -> str:
    print("\n" + "=" * 72)
    print("Stage 2: PubMed + climbmix medical-text continued pretraining")
    print("=" * 72)

    ensure_stage_data(2, preprocess=args.preprocess)

    if not Path(config.BEST_PRETRAINED_PTH).exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint missing: {config.BEST_PRETRAINED_PTH}. "
            "Run --stage 1 first."
        )

    train_dataset = dl.build_stage2_dataset(
        tokenizer,
        split="train",
        max_documents=args.max_documents,
        max_blocks=args.max_blocks,
    )
    val_dataset = dl.build_stage2_dataset(
        tokenizer,
        split="val",
        max_documents=args.max_val_documents,
        max_blocks=args.max_val_blocks,
    )
    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        config.STAGE2_BATCH_SIZE,
    )

    model, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        train_loader,
        config.STAGE2_EPOCHS,
        config.STAGE2_LEARNING_RATE,
        config.STAGE2_WARMUP_STEPS,
    )
    load_model_weights(config.BEST_PRETRAINED_PTH, model, config.DEVICE)

    stage2_best_path = os.path.join(config.STAGE2_CHECKPOINT, "best_stage2.pth")
    _, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE2_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.STAGE2_CHECKPOINT,
        best_checkpoint_path=stage2_best_path,
        resume_from_latest=args.resume,
    )

    print(f"Stage 2 complete. Best val loss: {best_val_loss:.4f}")
    return stage2_best_path


def run_stage3(args, tokenizer) -> str:
    print("\n" + "=" * 72)
    print("Stage 3: chatbot fine-tuning")
    print("=" * 72)

    ensure_stage_data(3, preprocess=args.preprocess)

    stage2_best_path = os.path.join(config.STAGE2_CHECKPOINT, "best_stage2.pth")
    if not Path(stage2_best_path).exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint missing: {stage2_best_path}. "
            "Run --stage 2 first."
        )

    train_dataset = dl.build_stage3_dataset(
        "train",
        tokenizer,
        max_conversations=args.max_conversations,
        max_blocks=args.max_blocks,
    )
    val_dataset = dl.build_stage3_dataset(
        "val",
        tokenizer,
        max_conversations=args.max_val_conversations,
        max_blocks=args.max_val_blocks,
    )
    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        config.STAGE3_BATCH_SIZE,
    )

    model, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        train_loader,
        config.STAGE3_EPOCHS,
        config.STAGE3_LEARNING_RATE,
        config.STAGE3_WARMUP_STEPS,
    )
    load_model_weights(stage2_best_path, model, config.DEVICE)

    freeze_blocks = max(0, len(model.blocks) - 2)
    _, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE3_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.STAGE3_CHECKPOINT,
        best_checkpoint_path=config.BEST_MODEL_PTH,
        resume_from_latest=args.resume,
        freeze_epochs=config.STAGE3_FREEZE_EPOCHS,
        freeze_blocks=freeze_blocks,
    )

    print(f"Stage 3 complete. Best val loss: {best_val_loss:.4f}")
    return config.BEST_MODEL_PTH


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-interactive 3-stage training.")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Regenerate the preprocessed artifact for the selected stage before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pth inside the selected stage checkpoint directory.",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-val-documents", type=int, default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-val-conversations", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-val-blocks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()

    print(f"Project root : {config.PROJECT_ROOT}")
    print(f"Device       : {config.DEVICE}")
    print(f"Vocab size   : {config.VOCAB_SIZE:,}")
    print(f"Model        : emb={config.N_EMB}, layers={config.N_LAYER}, heads={config.N_HEAD}")

    if config.N_EMB % config.N_HEAD != 0:
        raise ValueError(
            f"N_EMB ({config.N_EMB}) must be divisible by N_HEAD ({config.N_HEAD})."
        )

    if args.stage in ("1", "all"):
        run_stage1(args, tokenizer)

    if args.stage in ("2", "all"):
        run_stage2(args, tokenizer)

    if args.stage in ("3", "all"):
        run_stage3(args, tokenizer)


if __name__ == "__main__":
    main()

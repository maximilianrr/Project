#!/usr/bin/env python3
"""
run_bianca.py — Non-interactive training launcher for Bianca (NAISS SUPR).

Replaces the interactive input() prompts in train.py with CLI flags so the
job can run unattended under SLURM.

Usage (called by run_training.sh inside the container):
    python run_bianca.py --mode pretrain               # Stage 1 only
    python run_bianca.py --mode finetune               # Stage 2 only (needs pretrained weights)
    python run_bianca.py --mode both                   # Stage 1 then Stage 2 (default)
"""

import argparse
import os
import sys
from torch.utils.data import ConcatDataset

# ── Make project src/ and nanochat/ importable 
_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_src_path     = os.path.join(_project_root, "src")
_nanochat     = os.path.join(_project_root, "..", "nanochat")

for _p in [_src_path, _nanochat]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Imports (after path setup)
from chat_model import config
from chat_model.training.train import (
    setup_training,
    build_dataloaders,
    initialize_model_params,
    train,
    load_model_weights,
    ensure_pretraining_splits,
)
from chat_model.datasets import loader as dl


def run_pretrain(tokenizer, load_data: bool = False):
    print("\n" + "=" * 60)
    print("  STAGE 1 — General pretraining (climbmix + oasst2)")
    print("=" * 60)

    ensure_pretraining_splits(load_data=load_data)

    # Build both Stage 1 dataset types:
    #   climbmix  → ChunkTextDataset  (raw text, predicts every token)
    #   oasst2    → ChunkChatDataset  (conversations, loss_masking=False)
    text_train, chat_train = dl.build_pretrain_datasets(tokenizer, split="train")
    text_val,   chat_val   = dl.build_pretrain_datasets(tokenizer, split="val")

    parts_train = [d for d in [text_train, chat_train] if d is not None]
    parts_val   = [d for d in [text_val,   chat_val]   if d is not None]

    if not parts_train:
        raise RuntimeError(
            "No Stage 1 pretraining data found. "
            "Run preprocess.py --stage pretrain first."
        )

    combined_train = ConcatDataset(parts_train) if len(parts_train) > 1 else parts_train[0]
    combined_val   = ConcatDataset(parts_val)   if len(parts_val)   > 1 else parts_val[0]

    print(f"\nStage 1 combined train size: {len(combined_train):,} chunks")
    print(f"Stage 1 combined val size  : {len(combined_val):,} chunks")

    pretrain_train_loader = dl.make_dataloader(combined_train, batch_size=config.STAGE1_BATCH_SIZE)
    pretrain_val_loader   = dl.make_dataloader(combined_val,   batch_size=config.STAGE1_BATCH_SIZE)

    pretrain_model, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        pretrain_train_loader,
        config.STAGE1_EPOCHS,
        config.STAGE1_LEARNING_RATE,
        config.STAGE1_WARMUP_STEPS,
        weight_decay=0.1,
    )

    _, pretrain_best_val_loss = train(
        model=pretrain_model,
        train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE1_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.PRE_TRAINED_DIR,
        best_checkpoint_path=config.BEST_PRETRAINED_PTH,
        resume_from_latest=True,
    )

    print(f"\nPretraining complete. Best val loss: {pretrain_best_val_loss:.4f}")
    return pretrain_best_val_loss


def run_finetune(tokenizer, train_dataset, val_dataset):
    print("\n" + "=" * 60)
    print("  STAGE 2 — Medical fine-tuning")
    print("=" * 60)

    finetune_train_loader, finetune_val_loader = build_dataloaders(
        train_dataset, val_dataset, config.STAGE2_BATCH_SIZE
    )

    finetune_model, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        finetune_train_loader,
        config.STAGE2_EPOCHS,
        config.STAGE2_LEARNING_RATE,
        config.STAGE2_WARMUP_STEPS,
        weight_decay=0.01,
    )

    # Load pretrained weights — required for fine-tuning
    if not os.path.exists(config.BEST_PRETRAINED_PTH):
        raise FileNotFoundError(
            f"Pretrained weights not found at: {config.BEST_PRETRAINED_PTH}\n"
            f"Run with --mode pretrain or --mode both first."
        )
    load_model_weights(config.BEST_PRETRAINED_PTH, finetune_model, config.DEVICE)

    # Freeze all but the last 2 transformer blocks for the first freeze_epochs
    freeze_blocks = max(0, len(finetune_model.blocks) - 2)

    _, best_val_loss = train(
        model=finetune_model,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=config.DEVICE,
        epochs=config.STAGE2_EPOCHS,
        patience=config.PATIENCE,
        vocab_size=config.VOCAB_SIZE,
        model_dir=config.CHECKPOINTS_DIR,
        best_checkpoint_path=config.BEST_MODEL_PTH,
        resume_from_latest=False,
        freeze_epochs=2,
        freeze_blocks=freeze_blocks,
    )

    print(f"\nFine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Non-interactive NanoChat training launcher for Bianca SLURM jobs."
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "both"],
        default="both",
        help=(
            "pretrain  — Stage 1 only (climbmix + oasst2 general pretraining)\n"
            "finetune  — Stage 2 only (medical fine-tuning, needs pretrained weights)\n"
            "both      — Stage 1 then Stage 2 (default)"
        ),
    )
    parser.add_argument(
        "--load_data",
        action="store_true",
        default=False,
        help="Download and preprocess raw data before training.",
    )
    parser.add_argument(
        "--init_tokenizer",
        action="store_true",
        default=False,
        help="Train the BPE tokenizer before training (needs raw data).",
    )
    args = parser.parse_args()

    print(f"\nNanoChat training — mode={args.mode} | device={config.DEVICE}")
    print(f"  Project root    : {config.PROJECT_ROOT}")
    print(f"  Checkpoints     : {config.CHECKPOINTS_DIR}")
    print(f"  Pretrained      : {config.PRE_TRAINED_DIR}")
    print(f"  Splits          : {config.SPLITS_DIR}")
    print(f"  Pretrain splits : {config.PRETRAIN_SPLITS_DIR}")

    # setup_training loads tokenizer + builds Stage 2 fine-tune datasets (splits/)
    tokenizer, train_dataset, val_dataset = setup_training(
        load_data=args.load_data,
        init_tokenizer=args.init_tokenizer,
    )

    if args.mode in ("pretrain", "both"):
        run_pretrain(tokenizer, load_data=args.load_data)

    if args.mode in ("finetune", "both"):
        run_finetune(tokenizer, train_dataset, val_dataset)

    print("\nAll done.")


if __name__ == "__main__":
    main()
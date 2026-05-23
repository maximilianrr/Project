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

# ── Make project src/ and nanochat/ importable ───────────────────────────────
# PYTHONPATH is set by run_training.sh, but set it here too as a fallback.
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)          # Project-2/
_src_path    = os.path.join(_project_root, "src")
_nanochat    = os.path.join(_project_root, "..", "nanochat")

for _p in [_src_path, _nanochat]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports (after path setup) ────────────────────────────────────────────────
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
    print("  STAGE 1 — General conversational pretraining (oasst2)")
    print("=" * 60)

    ensure_pretraining_splits(load_data=load_data)

    pretrain_train_dataset = dl.build_dataset(
        "train", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR
    )
    pretrain_val_dataset = dl.build_dataset(
        "val", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR
    )
    pretrain_train_loader, pretrain_val_loader = build_dataloaders(
        pretrain_train_dataset, pretrain_val_dataset, config.STAGE1_BATCH_SIZE
    )

    pretrain_model, criterion, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        pretrain_train_loader,
        config.STAGE1_EPOCHS,
        config.STAGE1_LEARNING_RATE,
        config.STAGE1_WARMUP_STEPS,
    )

    _, pretrain_best_val_loss = train(
        model=pretrain_model,
        train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader,
        criterion=criterion,
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

    finetune_model, criterion, optimizer, scheduler, scaler = initialize_model_params(
        tokenizer,
        finetune_train_loader,
        config.STAGE2_EPOCHS,
        config.STAGE2_LEARNING_RATE,
        config.STAGE2_WARMUP_STEPS,
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
        criterion=criterion,
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
            "pretrain  — Stage 1 only (oasst2 general pretraining)\n"
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
    print(f"  Project root : {config.PROJECT_ROOT}")
    print(f"  Checkpoints  : {config.CHECKPOINTS_DIR}")
    print(f"  Pretrained   : {config.PRE_TRAINED_DIR}")
    print(f"  Splits       : {config.SPLITS_DIR}")
    print(f"  Pretrain splits: {config.PRETRAIN_SPLITS_DIR}")

    # setup_training loads tokenizer + builds fine-tune datasets (splits/)
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

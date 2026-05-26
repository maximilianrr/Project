import os
import sys
import pickle
import json
import gc
import torch
from torch.amp import grad_scaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import argparse
from pathlib import Path

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model.datasets import loader as dl
from chat_model.datasets import preprocess as pp
from chat_model.model.model import NanoChat
from chat_model.datasets import create_tokenizer
from chat_model import config


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"checkpoint saved → {path}")


def load_checkpoint(path: str, model, device: torch.device, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"resumed from epoch {ckpt['epoch']}  (best val loss: {ckpt['best_val_loss']:.4f})")
    return ckpt["epoch"], ckpt["best_val_loss"]


def load_model_weights(path: str, model, device: torch.device):
    """Loads only model weights — used to bootstrap fine-tuning from pretrained weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained weights not found at: {path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"loaded model weights from {path}")


def freeze_for_finetuning(model, freeze_blocks: int = 0):
    """Freeze embeddings and first N transformer blocks for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    if freeze_blocks > 0:
        for param in model.token_embedding_table.parameters():
            param.requires_grad = False
        for block in model.blocks[:freeze_blocks]:
            for param in block.parameters():
                param.requires_grad = False
        print(f"frozen token embeddings and first {freeze_blocks} transformer blocks")


def unfreeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = True


def _clear_device_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps allocated" in message or "cuda out of memory" in message


def _select_microbatch_size(device: torch.device, outer_batch_size: int) -> int:
    override = os.getenv("NANOCHAT_MICROBATCH_SIZE")
    if override:
        try:
            return max(1, min(int(override), outer_batch_size))
        except ValueError:
            pass
    if outer_batch_size <= 1:
        return outer_batch_size
    if device.type == "mps":
        return 1
    if device.type == "cuda":
        return min(2, outer_batch_size)
    return outer_batch_size


def _run_microbatched_pass(
    model, inputs, labels, criterion, device, vocab_size,
    train_mode: bool, scaler=None,
):
    batch_size = inputs.size(0)
    microbatch_size = _select_microbatch_size(device, batch_size)
    total_loss = 0.0
    processed = 0

    if train_mode:
        model.train()
    else:
        model.eval()

    idx = 0
    while idx < batch_size:
        current_microbatch = min(microbatch_size, batch_size - idx)
        micro_inputs = inputs[idx: idx + current_microbatch]
        micro_labels = labels[idx: idx + current_microbatch]

        try:
            with autocast(device_type=device.type):
                outputs = model(micro_inputs, micro_labels)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                targets = micro_labels[:, 1:]
                preds   = logits[:, :-1, :].reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(preds, targets)

            total_loss += loss.item() * current_microbatch
            processed  += current_microbatch

            if train_mode:
                scaled_loss = loss * (current_microbatch / batch_size)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            idx += current_microbatch

        except RuntimeError as exc:
            if _is_oom_error(exc) and current_microbatch > 1:
                _clear_device_cache(device)
                gc.collect()
                microbatch_size = max(1, current_microbatch // 2)
                continue
            raise

    return total_loss / max(1, processed)


def train(
    model, train_loader, val_loader, criterion, optimizer, scaler, scheduler,
    device, epochs, patience, vocab_size, model_dir: str,
    best_checkpoint_path: str | None = None,
    resume_from_latest: bool = True,
    freeze_epochs: int = 0,
    freeze_blocks: int = 0,
):
    print("Starting training")
    print(f"Training on Device: {device}")
    model.to(device)
    use_amp = device.type == "cuda"

    patience_counter = 0
    best_val_loss = float("inf")
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    os.makedirs(model_dir, exist_ok=True)
    latest_ckp = os.path.join(model_dir, "latest.pth")
    if resume_from_latest and os.path.exists(latest_ckp):
        start_epoch, best_val_loss = load_checkpoint(latest_ckp, model, device, optimizer, scheduler)

    if freeze_epochs > 0:
        freeze_for_finetuning(model, freeze_blocks=freeze_blocks)

    for epoch in range(start_epoch, epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            unfreeze_all_parameters(model)
            print("unfroze all model parameters for full fine-tuning")

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in loop:
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = inputs.to(device), labels.to(device)

            loss = _run_microbatched_pass(
                model=model, inputs=inputs, labels=labels, criterion=criterion,
                device=device, vocab_size=vocab_size, train_mode=True, scaler=scaler,
            )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            train_loss += loss
            loop.set_postfix(loss=train_loss / max(1, len(train_loader)))

        model.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = _run_microbatched_pass(
                    model=model, inputs=inputs, labels=labels, criterion=criterion,
                    device=device, vocab_size=vocab_size, train_mode=False,
                )
                val_loss += loss

            avg_train_loss = train_loss / max(1, len(train_loader))
            avg_val_loss   = val_loss   / max(1, len(val_loader))

            loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            print(f"Epoch {epoch+1:03d} | train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        current_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        if best_checkpoint_path is None:
            best_checkpoint_path = os.path.join(model_dir, "best.pth")

        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss

        current_state["best_val_loss"] = best_val_loss
        save_checkpoint(current_state, latest_ckp)

        if improved:
            save_checkpoint(current_state, best_checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(current_state, os.path.join(model_dir, f"epoch_{epoch+1:03d}.pth"))

        if patience_counter >= patience:
            print(f"\n early stopping triggered at epoch {epoch+1}")
            break

    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return history, best_val_loss


def setup_training(load_data: bool = False, init_tokenizer: bool = False):
    """Set up Stage 2 fine-tuning datasets and tokenizer."""
    if load_data:
        dl.load_and_convert_data()
        pp.start_preprocess()

    if init_tokenizer:
        create_tokenizer()

    try:
        import nanochat
    except ImportError:
        nanochat_path = os.path.join(config.PROJECT_ROOT, "..", "nanochat")
        if os.path.exists(nanochat_path) and nanochat_path not in sys.path:
            sys.path.insert(0, nanochat_path)
        else:
            print("Warning: nanochat not found. If pickle fails, install with:")
            print("   pip install git+https://github.com/karpathy/nanochat.git")

    with open(config.TOKENIZER_PKL, "rb") as file:
        try:
            tokenizer = pickle.load(file)
        except ModuleNotFoundError as e:
            print(f"Error loading tokenizer: {e}")
            print("\nThe tokenizer was created with nanochat.tokenizer.RustBPETokenizer")
            print("which is not available. Please install nanochat:")
            print("\n  pip install git+https://github.com/karpathy/nanochat.git")
            print("\nOr check that nanochat is cloned to: ../nanochat/")
            raise

    # Stage 2: conversational medical fine-tuning datasets (ChunkChatDataset, loss_masking=True)
    train_dataset = dl.build_dataset(
        "train", tokenizer, data_dir=config.SPLITS_DIR,
        max_conversations=None, loss_masking=True,
    )
    val_dataset = dl.build_dataset(
        "val", tokenizer, data_dir=config.SPLITS_DIR,
        max_conversations=None, loss_masking=True,
    )

    return tokenizer, train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size: int):
    train_loader = dl.make_dataloader(train_dataset, batch_size=batch_size)
    val_loader   = dl.make_dataloader(val_dataset,   batch_size=batch_size)
    return train_loader, val_loader


def ensure_pretraining_splits(load_data: bool = False):
    climbmix_file = Path(config.PRETRAIN_SPLITS_DIR) / "climbmix_docs.txt"
    oasst2_train  = Path(config.PRETRAIN_SPLITS_DIR) / "train.jsonl"

    if load_data or not (climbmix_file.exists() and oasst2_train.exists()):
        pp.preprocess_pretrain()


def initialize_model_params(tokenizer, train_loader, epochs, lr, warmup_steps):
    num_batches_per_epoch = max(1, len(train_loader))
    total_training_steps  = epochs * num_batches_per_epoch
    if total_training_steps > 1:
        warmup_steps = min(max(1, warmup_steps), total_training_steps - 1)
    else:
        warmup_steps = 0

    model = NanoChat(config=config)
    model.to(config.DEVICE)

    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, "get_bos_token_id"):
        pad_token_id = tokenizer.get_bos_token_id()
    else:
        raise AttributeError("Tokenizer has no `pad_token_id` or `get_bos_token_id` method")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = NanoChat.create_configured_optimizer(model, weight_decay=0.01, lr=lr)

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_T_max = max(1, total_training_steps - warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T_max, eta_min=lr * 0.1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    scaler = grad_scaler.GradScaler(enabled=config.DEVICE.type == "cuda")
    return model, criterion, optimizer, scheduler, scaler


def main(load_data: bool = False, init_tokenizer: bool = False):
    """
    Main two-stage training loop.

    Stage 1 — Pretraining:
      Trains on climbmix raw text (ChunkTextDataset) and oasst2 conversations
      (ChunkChatDataset, loss_masking=False) combined via ConcatDataset.

    Stage 2 — Medical fine-tuning:
      Fine-tunes on MedDialog / MedQuAD / MEDIQA conversational data
      (ChunkChatDataset, loss_masking=True), starting from Stage 1 weights.
    """
    tokenizer, train_dataset, val_dataset = setup_training(load_data, init_tokenizer)

    start_pre_training = input(
        "Data and tokenizer setup complete. Start pre-training? (y/n): "
    ).lower().startswith("y")

    if start_pre_training:
        print("\nStarting Stage 1 pretraining...")
        ensure_pretraining_splits(load_data)

        # Build both pretraining dataset types
        pretrain_text_train, pretrain_chat_train = dl.build_pretrain_datasets(
            tokenizer, split="train"
        )
        pretrain_text_val, pretrain_chat_val = dl.build_pretrain_datasets(
            tokenizer, split="val"
        )

        # Combine climbmix (ChunkTextDataset) and oasst2 (ChunkChatDataset) via ConcatDataset
        # Both return (x, y) pairs so they are directly compatible.
        pretrain_parts_train = [d for d in [pretrain_text_train, pretrain_chat_train] if d is not None]
        pretrain_parts_val   = [d for d in [pretrain_text_val,   pretrain_chat_val]   if d is not None]

        if not pretrain_parts_train:
            print("No pretraining data found. Run download_data() + preprocess_pretrain() first.")
        else:
            combined_train = ConcatDataset(pretrain_parts_train) if len(pretrain_parts_train) > 1 else pretrain_parts_train[0]
            combined_val   = ConcatDataset(pretrain_parts_val)   if len(pretrain_parts_val)   > 1 else pretrain_parts_val[0]

            print(f"\nStage 1 combined train size: {len(combined_train):,} chunks")
            print(f"Stage 1 combined val size  : {len(combined_val):,} chunks")

            pretrain_train_loader = dl.make_dataloader(combined_train, batch_size=config.STAGE1_BATCH_SIZE)
            pretrain_val_loader   = dl.make_dataloader(combined_val,   batch_size=config.STAGE1_BATCH_SIZE)

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

            print(f"\nStage 1 pretraining complete. Best val loss: {pretrain_best_val_loss:.4f}")

    else:
        print("Skipping pre-training. Starting fine-tuning with existing pretrained weights (if available)...")

    start_fine_tuning = input("Start Stage 2 fine-tuning? (y/n): ").lower().startswith("y")

    if start_fine_tuning:
        print("\nStarting Stage 2 medical fine-tuning...")
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

        load_model_weights(config.BEST_PRETRAINED_PTH, finetune_model, config.DEVICE)

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

        print(f"\nStage 2 fine-tuning complete. Best val loss: {best_val_loss:.4f}")

    else:
        print("Fine-tuning skipped. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NanoChat model.")
    parser.add_argument("--load_data",       default=False, type=bool,
                        help="Download and preprocess data before training.")
    parser.add_argument("--init_tokenizer",  default=False, type=bool,
                        help="Initialize the tokenizer before training.")
    args = parser.parse_args()
    main(load_data=args.load_data, init_tokenizer=args.init_tokenizer)
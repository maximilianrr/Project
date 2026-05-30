import os
import sys
import pickle
import json
import gc
import torch 
from torch.amp import grad_scaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import argparse
from pathlib import Path

# Add src to path if needed (allows running this script directly)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model.datasets import loader as dl
from chat_model.datasets import preprocess as pp
from chat_model.model.model import NanoChat
from chat_model.datasets import create_tokenizer
from chat_model import config


# method copied from Francisca's notebook file 
def save_checkpoint(state: dict, path: str):
    """
    Saves the model at a certain checkpoint 

    :param state: the model state to save 
    :param path: the path to the location to save the model at
    """

    torch.save(state, path)
    print(f"checkpoint saved → {path}")

# method copied from Francisca's notebook file 
def load_checkpoint(path: str, model, device: torch.device, optimizer=None, scheduler=None):
    """
    Loads the saved checkpoint

    :param path: the path to the checkpoint that should be loaded 
    :param model: the model to which the saved state should be applied 
    :param optimizer: an optimizer that is applied on the model 
    :param scheduler: a scheduler that is applied to the model

    :return: the epoch and the validation loss of the saved model
    """

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer and "optimizer" in ckpt: 
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt: 
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"resumed from epoch {ckpt['epoch']}  (best val loss: {ckpt['best_val_loss']:.4f})")

    return ckpt["epoch"], ckpt["best_val_loss"]


def load_model_weights(path: str, model, device: torch.device):
    """
    Loads only model weights from a checkpoint or state dict file.
    Used to bootstrap fine-tuning from pretrained weights without restoring optimizer state.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained weights not found at: {path}")

    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"loaded model weights from {path}")


def freeze_for_finetuning(model, freeze_blocks: int = 0):
    """
    Freeze the embedding table and the first N transformer blocks.
    Because the output head is weight-tied to the token embedding table, freezing the embedding
    also freezes the output head.
    """

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
    model,
    inputs,
    labels,
    device,
    vocab_size,
    train_mode: bool,
    scaler=None,
):
    """
    Runs a forward pass in smaller chunks to avoid allocating a full B x T x V tensor at once.
    When train_mode is True, the loss is backpropagated per microbatch and accumulated before
    the optimizer step.
    """

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
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    logits = outputs[0]
                    loss = outputs[1] #The model already calculates loss internally
                else:
                    logits = outputs
                    loss = None

            total_loss += loss.item() * current_microbatch
            processed += current_microbatch

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
    model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    scheduler,
    device,
    epochs,
    patience,
    vocab_size,
    model_dir: str,
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

    # load exisiting checkpoints (if existing)
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

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for inputs, labels in loop:
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = inputs.to(device), labels.to(device)

            loss = _run_microbatched_pass(
                model=model,
                inputs=inputs,
                labels=labels,
                device=device,
                vocab_size=vocab_size,
                train_mode=True,
                scaler=scaler,
            )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # advance LR scheduler per training step (batch)
            scheduler.step()

            train_loss += loss
            loop.set_postfix(loss=train_loss / max(1, len(train_loader)))

        model.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                loss = _run_microbatched_pass(
                    model=model,
                    inputs=inputs,
                    labels=labels,
                    device=device,
                    vocab_size=vocab_size,
                    train_mode=False,
                )

                val_loss += loss

            # scheduler is advanced per training step (batch), so no step here

            avg_train_loss = train_loss / max(1, len(train_loader))
            avg_val_loss = val_loss / max(1, len(val_loader))

            loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["lr"].append(optimizer.param_groups[0]['lr'])

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

        # Save best model separately
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
    if load_data == True: 
        # Download and preprocess data, then create tokenizer and tokenized datasets
        dl.load_and_convert_data()
        pp.start_preprocess()

    if init_tokenizer == True:
        # Initialize the tokenizer
        create_tokenizer()

    # Try to make nanochat available for unpickling the tokenizer
    # Check if nanochat is already importable
    try:
        import nanochat
    except ImportError:
        # Try to find nanochat in a sibling directory
        nanochat_path = os.path.join(config.PROJECT_ROOT, "..", "nanochat")
        if os.path.exists(nanochat_path) and nanochat_path not in sys.path:
            sys.path.insert(0, nanochat_path)
        else:
            print("Warning: nanochat not found. If pickle fails, install with:")
            print("   pip install git+https://github.com/karpathy/nanochat.git")

    with open(config.TOKENIZER_PKL, 'rb') as file:
        try:
            tokenizer = pickle.load(file)
        except ModuleNotFoundError as e:
            print(f"Error loading tokenizer: {e}")
            print("\nThe tokenizer was created with nanochat.tokenizer.RustBPETokenizer")
            print("which is not available. Please install nanochat:")
            print("\n  pip install git+https://github.com/karpathy/nanochat.git")
            print("\nOr check that nanochat is cloned to: ../nanochat/")
            raise

    train_dataset = dl.build_dataset("train", tokenizer, data_dir=config.SPLITS_DIR, max_conversations=None, loss_masking=True)
    val_dataset = dl.build_dataset("val",   tokenizer, data_dir=config.SPLITS_DIR, max_conversations=None, loss_masking=True)

    return tokenizer, train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size: int):
    train_loader = dl.make_dataloader(train_dataset, batch_size=batch_size)
    val_loader = dl.make_dataloader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def ensure_pretraining_splits(load_data: bool = False):
    train_split = Path(config.PRETRAIN_SPLITS_DIR) / "train.jsonl"
    val_split = Path(config.PRETRAIN_SPLITS_DIR) / "val.jsonl"

    if load_data or not (train_split.exists() and val_split.exists()):
        pp.preprocess_pretrain()


def initialize_model_params(tokenizer, train_loader, epochs, lr, warmup_steps,weight_decay=0.01):

    # compute scheduler units in training *steps* (batches) rather than epochs
    num_batches_per_epoch = max(1, len(train_loader))
    total_training_steps = epochs * num_batches_per_epoch
    # ensure warmup is valid for the actual training horizon
    if total_training_steps > 1:
        warmup_steps = min(max(1, warmup_steps), total_training_steps - 1)
    else:
        warmup_steps = 0

    model = NanoChat(config=config)
    model.to(config.DEVICE)

    # Determine a pad token id for the loss's `ignore_index`.
    # Nanochat's RustBPETokenizer does not expose `pad_token_id` — reuse BOS as pad token.
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, "get_bos_token_id"):
        pad_token_id = tokenizer.get_bos_token_id()
    else:
        raise AttributeError("Tokenizer has no `pad_token_id` or `get_bos_token_id` method")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = NanoChat.create_configured_optimizer(model, weight_decay=0.1, lr=lr)
    if warmup_steps > 0:
        # linear warmup for `warmup_steps` training steps from 0→100% of base LR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )

        # cosine decay after warmup. T_max expressed in training steps (remaining steps)
        cosine_T_max = max(1, total_training_steps - warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T_max, eta_min=lr * 0.1
        )

        # milestones for SequentialLR are in the same units as total_iters (steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = grad_scaler.GradScaler(enabled=config.DEVICE.type == "cuda")

    return model, optimizer, scheduler, scaler



def main(load_data: bool = False, init_tokenizer: bool = False): 
    """
    Main training loop for the NanoChat model.
    Args:
        load_data: If True, download and preprocess the data before training.
        init_tokenizer: If True, initialize the tokenizer before training.
    """

    tokenizer, train_dataset, val_dataset = setup_training(load_data, init_tokenizer)

    start_pre_training = input("Data and tokenizer setup complete. Start pre-training? (y/n): ").lower().startswith("y")

    if start_pre_training:
        print("Starting pretraining...")
        ensure_pretraining_splits(load_data)
        pretrain_train_dataset = dl.build_dataset("train", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR, max_conversations = None, loss_masking=False)
        pretrain_val_dataset = dl.build_dataset("val", tokenizer, data_dir=config.PRETRAIN_SPLITS_DIR, max_conversations = None, loss_masking=False)
        pretrain_train_loader, pretrain_val_loader = build_dataloaders(pretrain_train_dataset, pretrain_val_dataset, config.STAGE1_BATCH_SIZE)
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

        print(f"Pretraining complete. Best val loss: {pretrain_best_val_loss:.4f}")
    else: 
        print("Skipping pre-training. Starting fine-tuning with existing pretrained weights (if available)...")


    start_fine_tuning = input("Start fine-tuning? (y/n): ").lower().startswith("y")

    if start_fine_tuning:
        print("Starting fine-tuning...")
        finetune_train_loader, finetune_val_loader = build_dataloaders(train_dataset, val_dataset, config.STAGE2_BATCH_SIZE)
        finetune_model, optimizer, scheduler, scaler = initialize_model_params(
            tokenizer,
            finetune_train_loader,
            config.STAGE2_EPOCHS,
            config.STAGE2_LEARNING_RATE,
            config.STAGE2_WARMUP_STEPS,
            weight_decay=0.01,
        )

        load_model_weights(config.BEST_PRETRAINED_PTH, finetune_model, config.DEVICE)

        # Freeze most of the model for the first two epochs of fine-tuning.
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

        print(f"\n Training complete. Best val loss: {best_val_loss:.4f}")

    else: 
        print("Fine-tuning skipped. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NanoChat model.")

    parser.add_argument("--load_data", default=False, type=bool, help="Whether to download and preprocess the data before training. Default is False.")
    parser.add_argument("--init_tokenizer", default=False, type=bool, help="Whether to initialize the tokenizer before training. Default is False.")

    args = parser.parse_args()
    main(load_data=args.load_data, init_tokenizer=args.init_tokenizer)

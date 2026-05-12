import os
import pickle
import json
import torch 
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import argparse

import utils.data_loader as dl
from models.model import NanoChat
from utils.tokenizer import create_tokenizer
import config


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


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience, checkpoint_dir=config.CHECKPOINTS_DIR): 
    print("Starting training")
    print(f"Training on Device: {device}")
    model.to(device)

    patience_counter = 0
    best_val_loss = float("inf")
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # load exisiting checkpoints (if existing)
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckp = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(latest_ckp): 
        start_epoch, best_val_loss = load_checkpoint(latest_ckp, model, device, optimizer, scheduler)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                _, loss = model(inputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=train_loss / (len(train_loader)))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(device_type=device.type):
                    _, loss = model(inputs, labels)
                val_loss += loss.item()

            scheduler.step(val_loss)
            loop.set_postfix(train_loss=train_loss / len(train_loader), val_loss=val_loss / len(val_loader))

            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader))
            history["lr"].append(optimizer.param_groups[0]['lr'])

            print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        save_checkpoint(state, latest_ckp)

        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, os.path.join(checkpoint_dir, "best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pth"))

        if patience_counter >= patience:
            print(f"\n early stopping triggered at epoch {epoch+1}")
            break

    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return history, best_val_loss


def main(load_data: bool = False, init_tokenizer: bool = False): 
    """
    Main training loop for the NanoChat model.
    Args:
        load_data: If True, download and preprocess the data before training.
        init_tokenizer: If True, initialize the tokenizer before training.
    """

    if load_data == True: 
        # Download and preprocess data, then create tokenizer and tokenized datasets
        dl.load_and_convert_data()

    if init_tokenizer == True:
        # Initialize the tokenizer
        create_tokenizer()

    with open(config.TOKENIZER_PKL, 'rb') as file:
        tokenizer = pickle.load(file)

    train_dataset = dl.build_dataset("train", tokenizer, data_dir=config.SPLITS_DIR, max_conversations=100)
    val_dataset   = dl.build_dataset("val",   tokenizer, data_dir=config.SPLITS_DIR, max_conversations=100)
    train_loader = dl.make_dataloader(train_dataset, batch_size=config.BATCH_SIZE)
    val_loader   = dl.make_dataloader(val_dataset,   batch_size=config.BATCH_SIZE)

    model = NanoChat(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    device = config.DEVICE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE)

    _, best_val_loss = train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=config.EPOCHS, patience=config.PATIENCE)

    print(f"\n Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NanoChat model.")

    parser.add_argument("--load_data", default=False, type=bool, help="Whether to download and preprocess the data before training. Default is False.")
    parser.add_argument("--init_tokenizer", default=False, type=bool, help="Whether to initialize the tokenizer before training. Default is False.")

    args = parser.parse_args()
    main(load_data=args.load_data, init_tokenizer=args.init_tokenizer)
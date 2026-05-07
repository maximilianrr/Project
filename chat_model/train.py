import os
import json
import random
import pickle
import torch 
import numpy as np
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import argparse

import utils.data_loader as dl
from models.model import NanoChat
import config
from torch.cuda.amp import GradScaler

def train_trial(model, train_loader, val_loader, optimizer, device, trial_config, trial_name): 
    """Trains a single hyperparameter combination."""
    epochs = config.EPOCHS
    patience = 3  # Early stopping within a trial
    best_val_loss = float('inf')
    es_counter = 0
    
    history = {
        "config": trial_config,
        "train_loss": [],
        "val_loss": []
    }
    print(trial_config)
    scaler = GradScaler()

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"{trial_name} | Epoch {epoch+1} [Train]")
        
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                _, loss = model(inputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_loop.set_postfix(loss=train_loss / (train_loop.n + 1))

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"{trial_name} | Epoch {epoch+1} [Val]", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type=device.type):
                    _, loss = model(inputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Best Model & Early Stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best weights for THIS specific trial
            torch.save(model.state_dict(), f"output/weights_{trial_name}_best.pth")
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= patience:
            print(f"Early stopping trial at epoch {epoch+1}")
            break

    # Save trial history
    with open(f"output/metrics_{trial_name}.json", "w") as f:
        json.dump(history, f, indent=4)

    return best_val_loss

def main(load_data=False, init_tokenizer=False, num_trials=10):
    # --- MOVED THIS HERE ---
    import sys
    import os
    if "/root" not in sys.path:
        sys.path.append("/root")
    # -----------------------

    # Optional: Keep the logic if you ever need it again!
    if load_data: 
        dl.load_and_convert_data()

    if init_tokenizer:
        from utils.tokenizer import create_tokenizer
        create_tokenizer()

    from nanochat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(os.path.join('data', 'tokenized'))
    dl.debug_boundaries(tokenizer)

    # ... rest of your main function stays the same ...

    # Search Space
    lr_options = [1e-3, 1e-4, 1e-5, 1e-6]
    batch_size_options = [8, 16, 32, 48]
    
    best_overall_loss = float('inf')
    best_overall_config = None
    leaderboard = []

    print(f"\n{'='*50}\nSTARTING HYPERPARAMETER SEARCH\n{'='*50}")
    train_dataset = dl.build_dataset("train", tokenizer, data_dir="data/splits")
    val_dataset   = dl.build_dataset("val",   tokenizer, data_dir="data/splits")
    for trial in range(num_trials):
        # Sample parameters
        temp_lr = random.choice(lr_options)
        temp_bs = random.choice(batch_size_options)
        temp_dp = round(random.uniform(0.1, 0.5), 2)
        
        trial_config = {
            "lr": temp_lr,
            "batch_size": temp_bs,
            "dropout": temp_dp,
            "epochs": config.EPOCHS
        }
        
        trial_name = f"trial_{trial}_LR{temp_lr}_BS{temp_bs}_DP{temp_dp}"
        print(f"\n--- Trial {trial+1}/{num_trials} | {trial_name} ---")

        # Dynamically set config for this trial
        config.LEARNING_RATE = trial_config["lr"]
        config.BATCH_SIZE = trial_config["batch_size"]
        config.DROPOUT = trial_config["dropout"]

        # Re-init loaders with new Batch Size
        train_loader = dl.make_dataloader(train_dataset, batch_size=temp_bs)
        val_loader   = dl.make_dataloader(val_dataset,   batch_size=temp_bs)

        # Re-init model and optimizer
        model = NanoChat(config=config)
        model = model.to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=temp_lr)
        
        # Execute training
        trial_best_val = train_trial(model, train_loader, val_loader, optimizer, config.DEVICE, trial_config, trial_name)

        # Update Leaderboard
        result = {"trial": trial_name, "config": trial_config, "best_val_loss": trial_best_val}
        leaderboard.append(result)

        if trial_best_val < best_overall_loss:
            best_overall_loss = trial_best_val
            best_overall_config = trial_config
            print(f"⭐ New Leaderboard Leader! Val Loss: {trial_best_val:.4f}")

        # Cleanup memory before next trial
        del model
        torch.cuda.empty_cache()

    # Final leaderboard save
    with open("output/leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=4)

    print("\n" + "="*50)
    print(f"SEARCH COMPLETE\nBest Loss: {best_overall_loss:.4f}\nBest Config: {best_overall_config}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", default=10, type=int)
    args = parser.parse_args()
    main(num_trials=args.num_trials)

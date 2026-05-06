import os
import pickle

import torch 
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import argparse

import utils.data_loader as dl
from models.model import NanoChat
from utils.tokenizer import create_tokenizer
import config

def train(model, train_loader, val_loader, optimizer, device, epochs): 
    print("Starting training")
    print(f"Training on Device: {device}")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                _, loss, _ = model(inputs, labels)

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
                    _, loss, _ = model(inputs, labels)
                val_loss += loss.item()

            loop.set_postfix(train_loss=train_loss / len(train_loader), val_loss=val_loss / len(val_loader))


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

    with open(os.path.join('data', 'tokenized', 'tokenizer.pkl'), 'rb') as file:
        tokenizer = pickle.load(file)

    train_loader = dl.load_tokenized_dataloader("train", tokenizer=tokenizer)
    val_loader = dl.load_tokenized_dataloader("val", tokenizer=tokenizer)

    model = NanoChat(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = config.DEVICE

    train(model, train_loader, val_loader, optimizer, device, epochs=config.EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NanoChat model.")

    parser.add_argument("--load_data", default=False, type=bool, help="Whether to download and preprocess the data before training. Default is False.")
    parser.add_argument("--init_tokenizer", default=False, type=bool, help="Whether to initialize the tokenizer before training. Default is False.")

    args = parser.parse_args()
    main(load_data=args.load_data, init_tokenizer=args.init_tokenizer)

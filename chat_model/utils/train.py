import torch 
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs): 
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for _, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

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
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

            loop.set_postfix(train_loss=train_loss / len(train_loader), val_loss=val_loss / len(val_loader))

import torch
from tqdm import tqdm

def train_backprop(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="Backprop Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(train_loader)

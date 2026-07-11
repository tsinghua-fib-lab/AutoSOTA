import torch



def evaluate(model, loader, device="cpu"):
    # Compute accuracy over a DataLoader
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total
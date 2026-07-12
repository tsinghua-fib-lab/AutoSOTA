from tqdm import tqdm

def train_ff(model, train_loader, device):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="FF Train", leave=False)
    for pos_x, neg_x, neutral_x, y in pbar:
        pos_x, neg_x, neutral_x, y = pos_x.to(device), neg_x.to(device), neutral_x.to(device), y.to(device)
        loss = model(pos_x, neg_x, neutral_x, y)
        total_loss += loss
        pbar.set_postfix(loss=f"{loss:.4f}")
    return total_loss / len(train_loader)

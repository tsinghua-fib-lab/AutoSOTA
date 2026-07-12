import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_noprop(model, train_loader, optimizers, device):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="NoProp Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        u_y = F.one_hot(y, num_classes=model.num_classes).float()
        z_t_list = [torch.sqrt(model.alpha[t]) * u_y + torch.sqrt(1 - model.alpha[t]) * torch.randn_like(u_y) for t in range(model.T)]
        x_features = model.cnn(x)
        losses = [F.mse_loss(model.mlps[t](z_t_list[t].detach(), x_features), u_y) for t in range(model.T)]
        loss = sum(losses)
        for opt in optimizers: opt.zero_grad()
        loss.backward()
        for opt in optimizers: opt.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(train_loader)

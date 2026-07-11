import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from create_data.dataset import TransformedDataset
from utils.params import get_flat_params


def train_prior(model_cls, data_dir, save_path, batch_size=128, lr=1e-3, epochs=5, seed=0, device="cpu",
                weight_decay=0.0):

    torch.manual_seed(seed)

    dataset = TransformedDataset(os.path.join(data_dir, "prior.pt"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model_cls().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[prior] epoch {epoch:02d} loss={avg_loss:.4f}")

    prior_mu = get_flat_params(model).cpu()

    # Save structured prior (PAC-Bayes ready)
    torch.save({
        "mu": prior_mu,
        "sigma": 5e-2  # can be tuned
    }, save_path)

    return prior_mu
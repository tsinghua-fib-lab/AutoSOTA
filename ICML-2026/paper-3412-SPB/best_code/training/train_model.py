import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from create_data.dataset import TransformedDataset
from utils.params import set_flat_params, get_flat_params
from training.evaluate import evaluate


def train_model(model_cls, data_dir, prior_path, save_path, batch_size=128, lr=1e-3, epochs=20, seed=0, device="cpu", num_workers=2, label_smoothing=0.0, use_cosine=False, kl_lambda=0.0):
    """
    Generic training function for all experiments.

    Args:
        model_cls: class of the model (e.g. BaselineCNN, EquivariantCNN)
        data_dir: directory containing train.pt / val.pt / test.pt
        prior_path: path to flat parameter vector
        save_path: where to save best model
    """

    torch.manual_seed(seed)

    # ---- data loading ----
    def load_split(name):
        path = os.path.join(data_dir, f"{name}.pt")
        return TransformedDataset(path)

    train_loader = DataLoader(
        load_split("train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        load_split("val"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        load_split("test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Instantiate model and initialize weights to the pre-trained prior mean
    model = model_cls().to(device)

    prior_mu = torch.load(prior_path, map_location=device)
    set_flat_params(model, prior_mu["mu"])

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader)) if use_cosine else None

    best_val_acc = 0.0

    # ---- training loop ----
    for epoch in range(1, epochs + 1):

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            if kl_lambda > 0:
                flat_w = torch.cat([p.view(-1) for p in model.parameters()])
                kl_penalty = kl_lambda * ((flat_w - prior_mu["mu"].to(device))**2).sum() / (2 * 0.05**2 * len(train_loader.dataset))
                loss = loss + kl_penalty
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Test acc: {test_acc:.4f} | "
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # ---- test evaluation ----
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)

    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    return model
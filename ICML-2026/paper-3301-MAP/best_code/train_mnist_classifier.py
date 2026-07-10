import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from datasets import _load_mnist_tensors
from torch.utils.data import DataLoader, TensorDataset, random_split


# -------------------------
# LeNet-style MNIST classifier
# -------------------------
class SimpleMNISTClassifier(nn.Module):
    """
    LeNet-style CNN for MNIST.

    - Input: (B, 1, 28, 28) or (B, 28, 28)
    - Output logits: (B, 10)
    - Features: (B, feat_dim) from penultimate layer (for FID-style embeddings)
    """
    def __init__(self, feat_dim: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 28->24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 12->8
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, feat_dim)             # penultimate features
        self.fc3 = nn.Linear(feat_dim, 10)

    def forward(self, x, return_features: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,28,28) -> (B,1,28,28)

        x = self.pool(F.relu(self.conv1(x)))  # (B,6,12,12)
        x = self.pool(F.relu(self.conv2(x)))  # (B,16,4,4)

        x = torch.flatten(x, 1)               # (B,256)
        x = F.relu(self.fc1(x))               # (B,120)
        feats = F.relu(self.fc2(x))           # (B,feat_dim)
        logits = self.fc3(feats)              # (B,10)

        if return_features:
            return logits, feats
        return logits


# -------------------------
# Helpers
# -------------------------
def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_mnist_classifier(path="models/mnist_classifier.pth", device=None, feat_dim: int | None = None):
    """
    Loads a checkpoint saved by train_classifier below.

    If feat_dim is None, attempt to infer the feature dimensionality from the
    checkpoint (preferred). If provided, feat_dim must match the model used
    for training that checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the raw state dict first so we can infer feature dimensionality
    state = torch.load(path, map_location=device)
    state_dict = state if isinstance(state, dict) else {}

    inferred_feat = None
    try:
        # prefer fc2.weight shape -> (feat_dim, 120)
        if "fc2.weight" in state_dict:
            inferred_feat = int(state_dict["fc2.weight"].shape[0])
        # fallback: infer from fc3 weight shape -> (10, feat_dim)
        elif "fc3.weight" in state_dict:
            inferred_feat = int(state_dict["fc3.weight"].shape[1])
    except Exception:
        inferred_feat = None

    if feat_dim is None and inferred_feat is not None:
        feat_dim = inferred_feat
    elif feat_dim is None:
        # default to 84 if nothing can be inferred
        feat_dim = 84

    model = SimpleMNISTClassifier(feat_dim=feat_dim)
    # Now load the state dict (may raise if mismatched)
    try:
        model.load_state_dict(state)
    except Exception:
        # If the saved checkpoint wrapped state under other keys, try common ones
        if isinstance(state, dict):
            for k in ("model_state", "model_state_dict", "state_dict"):
                if k in state:
                    model.load_state_dict(state[k])
                    break
            else:
                # re-raise original exception for visibility
                raise
        else:
            raise
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_training_plots(history: dict, out_dir: str, prefix: str = "mnist_classifier"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    # ---- Global style for paper-level aesthetics ---- #
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })

    figsize = (6.5, 4.5)  # fits a single column cleanly

    # ---- Loss plot ---- #
    plt.figure(figsize=figsize)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    if len(history.get("test_loss_curve", [])) == len(history["train_loss"]):
        plt.plot(history["test_loss_curve"], label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.svg"))
    plt.close()

    # ---- Accuracy plot ---- #
    plt.figure(figsize=figsize)
    plt.plot(history["val_acc"], label="Validation Accuracy")
    if len(history.get("test_acc_curve", [])) == len(history["val_acc"]):
        plt.plot(history["test_acc_curve"], label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_acc.svg"))
    plt.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Training with train/val/test split
# -------------------------
def train_classifier(
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_path: str = "models/mnist_classifier.pth",
    plots_dir: str = "plots",
    feat_dim: int = 84,           # must match when you reload checkpoint
    val_frac: float = 0.1,        # fraction of TRAIN set used for validation
    seed: int = 0,
    num_workers: int = 0,         # safest cross-platform
):
    """
    Uses:
      - MNIST train=True split -> further split into train/val via random_split
      - MNIST train=False split -> test
    Logs train/val curves each epoch and reports final test once at the end.
    """
    assert 0.0 < val_frac < 1.0, "val_frac must be in (0,1)"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base datasets from in-repository MNIST IDX files.
    train_images, train_labels = _load_mnist_tensors(root="data", train=True)
    test_images, test_labels = _load_mnist_tensors(root="data", train=False)
    train_full = TensorDataset((train_images.float() / 255.0).unsqueeze(1), train_labels)
    test_dataset = TensorDataset((test_images.float() / 255.0).unsqueeze(1), test_labels)

    # Train/Val split from train_full
    n_total = len(train_full)
    n_val = int(round(val_frac * n_total))
    n_train = n_total - n_val

    # Ensure deterministic split
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_dataset, val_dataset = random_split(train_full, [n_train, n_val], generator=gen)

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Model
    model = SimpleMNISTClassifier(feat_dim=feat_dim).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__} (LeNet-style)")
    print(f"Parameters: total={total_params:,} | trainable={trainable_params:,}")
    print(f"Split sizes: train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "final_test_loss": None,
        "final_test_acc": None,
        # (optional) if you ever want per-epoch test curves; left empty by default
        "test_loss_curve": [],
        "test_acc_curve": [],
    }

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%"
        )

    # Option: evaluate final test using best-val checkpoint (recommended)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    history["final_test_loss"] = float(test_loss)
    history["final_test_acc"] = float(test_acc)

    print(f"FINAL TEST (best-val model) | loss={test_loss:.4f} | acc={test_acc*100:.2f}%")

    # Save checkpoint (same path as before)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save plots
    os.makedirs(plots_dir, exist_ok=True)
    save_training_plots(history, out_dir=plots_dir, prefix="mnist_classifier")
    print(f"Saved plots to {plots_dir}/mnist_classifier_loss.svg and {plots_dir}/mnist_classifier_acc.svg")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MNIST classifier")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/mnist_classifier.pth", help="Checkpoint path")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory for training plots")
    parser.add_argument("--feat_dim", type=int, default=84, help="Embedding dimension")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train set used for validation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        plots_dir=args.plots_dir,
        feat_dim=args.feat_dim,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
    )

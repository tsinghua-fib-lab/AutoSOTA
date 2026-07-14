#!/usr/bin/env python3
"""
Train an MLP classifier on saved VDW embeddings.

Example:
    python tests/run_embedding_mlp.py \\
        --fold_dir /path/to/experiments/.../fold_0 \\
        --mlp_layers 256,128 \\
        --dropout 0.2 \\
        --epochs 200 \\
        --patience 25
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy


def load_split(
    split_name: str,
    fold_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load embeddings and labels tensors for a split.
    """
    canonical_name = split_name
    filename = f"{canonical_name}_embeddings.pt"
    path = os.path.join(fold_dir, "embeddings", filename)

    if (not os.path.exists(path)) and (canonical_name == "valid"):
        alternate = os.path.join(fold_dir, "embeddings", "val_embeddings.pt")
        if os.path.exists(alternate):
            path = alternate

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find embeddings file for split '{split_name}' at {path}"
        )

    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    labels = payload["labels"].long()
    return embeddings, labels


def parse_layers(layers_str: str) -> List[int]:
    """
    Convert a comma-delimited list to integers.
    """
    values: List[int] = []
    for token in layers_str.split(","):
        stripped = token.strip()
        if stripped:
            values.append(int(stripped))
    return values or [256, 128]


class EmbeddingMLP(nn.Module):
    """
    Simple fully-connected classifier for embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.network(x)


@dataclass
class TrainConfig:
    """
    Container for CLI hyperparameters.
    """

    fold_dir: str
    mlp_layers: List[int]
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    device: str


def create_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Build a TensorDataset-backed DataLoader.
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric: Accuracy,
) -> float:
    """
    Evaluate accuracy on a dataloader.
    """
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            preds = torch.argmax(logits, dim=-1)
            metric.update(preds.cpu(), batch_y)
    return metric.compute().item()


def parse_args() -> TrainConfig:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on VDW embeddings."
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Path to the fold_* directory containing the embeddings/ folder.",
    )
    parser.add_argument(
        "--mlp_layers",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes, e.g., '256,128,64'.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied after each hidden ReLU layer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (epochs without validation improvement).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device identifier.",
    )
    args = parser.parse_args()
    return TrainConfig(
        fold_dir=os.path.abspath(args.fold_dir),
        mlp_layers=parse_layers(args.mlp_layers),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
    )


def main() -> None:
    """
    Entry point.
    """
    cfg = parse_args()
    device = torch.device(cfg.device)

    print(f"[INFO] Loading embeddings from: {cfg.fold_dir}")
    x_train, y_train = load_split("train", cfg.fold_dir)
    x_valid, y_valid = load_split("valid", cfg.fold_dir)
    x_test, y_test = load_split("test", cfg.fold_dir)

    input_dim = x_train.shape[1]
    num_classes = int(torch.max(torch.cat([y_train, y_valid, y_test], dim=0)).item() + 1)

    train_loader = create_dataloader(x_train, y_train, cfg.batch_size, shuffle=True)
    valid_loader = create_dataloader(x_valid, y_valid, cfg.batch_size, shuffle=False)
    test_loader = create_dataloader(x_test, y_test, cfg.batch_size, shuffle=False)

    model = EmbeddingMLP(
        input_dim=input_dim,
        hidden_dims=cfg.mlp_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    val_metric = Accuracy(task="multiclass", num_classes=num_classes)

    best_val_acc = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, valid_loader, device, val_metric)
        print(f"[EPOCH {epoch:03d}] validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience:
                print(f"[INFO] Early stopping triggered after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)

    train_metric = Accuracy(task="multiclass", num_classes=num_classes)
    train_acc = evaluate_accuracy(model, train_loader, device, train_metric)
    print(f"[RESULT] Train accuracy: {train_acc:.4f}")

    val_metric.reset()
    final_val_acc = evaluate_accuracy(model, valid_loader, device, val_metric)
    print(f"[RESULT] Validation accuracy (best checkpoint): {final_val_acc:.4f}")

    test_metric = Accuracy(task="multiclass", num_classes=num_classes)
    test_acc = evaluate_accuracy(model, test_loader, device, test_metric)
    print(f"[RESULT] Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()


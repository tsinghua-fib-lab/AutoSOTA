#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original DP-training script + optional linear LR-decay.
Nothing else is changed.
Run with --lr_decay to linearly decay LR → 0 across epochs.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as scio

# Differential Privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01):
    classification_loss = -torch.mean(
        y_true * torch.log_softmax(y_pred, dim=1)
    )
    weight = model._module.weight if hasattr(model, "_module") else model.weight
    RG_loss = (
        1
        / n
        * torch.norm(weight.unsqueeze(1) - weight.unsqueeze(0), p=2, dim=2)
        .pow(p)
        .sum()
    )
    return classification_loss + lam * RG_loss


def R_MLR(para):
    # ---------- data ----------
    path = f"./dataset/{para.data}.mat"
    X = scio.loadmat(path)["X"]
    y = scio.loadmat(path)["Y"].squeeze()
    print(f"Loaded {X.shape}, labels {y.shape}")

    n, d = X.shape
    num_class = len(np.unique(y))

    if para.If_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = y - 1  # ensure 0-based labels
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=para.test_size, random_state=para.state
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=128, shuffle=True
    )

    # ---------- model ----------
    model = torch.nn.Linear(d, num_class)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # ---------- optimiser ----------
    if para.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=para.lr, weight_decay=para.weight_decay
        )
    elif para.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=para.lr, weight_decay=para.weight_decay
        )
    elif para.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=para.lr)
    else:  # SGD as default
        optimizer = torch.optim.SGD(
            model.parameters(), lr=para.lr, weight_decay=para.weight_decay
        )
    print(f"Using {para.optimizer.upper()}  lr={para.lr}")

    # ---------- privacy (skip for LBFGS) ----------
    scheduler = None
    if para.optimizer != "lbfgs":
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=para.num_epoch,
            target_epsilon=para.eps,
            target_delta=para.delta,
            max_grad_norm=10.0,
        )
        print(f"σ (noise multiplier) = {optimizer.noise_multiplier:.4f}")

        # scheduler must be created **after** make_private
        if para.lr_decay and para.optimizer in {"sgd", "adam", "adagrad"}:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=para.num_epoch,
            )
            print("⇢ Linear LR-decay enabled")

    # ---------- loss ----------
    if para.loss == "svm":
        loss_fn = Regularized_loss
    else:  # "CE"
        loss_fn = torch.nn.CrossEntropyLoss()

    # ---------- training ----------
    for epoch in range(para.num_epoch):
        total_loss = 0.0
        num_batches = 0

        for X_b, y_b in train_loader:
            y_b_oh = F.one_hot(y_b, num_classes=num_class).float()

            def closure():
                optimizer.zero_grad()
                pred = model(X_train)
                if para.loss == "svm":
                    loss = loss_fn(model, n, pred, F.one_hot(y_train, num_classes=num_class).float(), para.p, para.lam)
                else:
                    loss = loss_fn(pred, y_train)
                loss.backward()
                return loss

            if para.optimizer == "lbfgs":
                loss = optimizer.step(closure)
                batch_loss = loss.item()
            else:
                optimizer.zero_grad()
                pred = model(X_b)
                if para.loss == "svm":
                    loss = loss_fn(model, n, pred, y_b_oh, para.p, para.lam)
                else:
                    loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()

            total_loss += batch_loss
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        with torch.no_grad():
            test_pred = model(X_test)
            correct = (torch.argmax(test_pred, 1) == y_test).sum().item()
            test_acc = correct / len(X_test)

        print(
            f"Epoch {epoch+1:3d}/{para.num_epoch}  "
            f"loss={avg_loss:.4f}  acc={test_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}",
            end=""
        )

        if para.optimizer != "lbfgs":
            eps_spent = privacy_engine.get_epsilon(delta=para.delta)
            print(f"  ε={eps_spent:.2f}")
        else:
            print()

    # ---------- final evaluation ----------
    with torch.no_grad():
        final_pred = model(X_test)
        final_acc = (torch.argmax(final_pred, 1) == y_test).float().mean().item()
    print(f"\nTest Accuracy: {final_acc:.4f}")

    if para.optimizer != "lbfgs":
        eps_final = privacy_engine.get_epsilon(delta=para.delta)
        print(f"Final Privacy Guarantee: ε = {eps_final:.2f}, δ = {para.delta}")

    print("Training complete!")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="train with differential privacy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="vehicle_uni")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--state", type=int, default=42)
    parser.add_argument("--If_scale", default=True)
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "lbfgs", "adagrad"],
    )
    parser.add_argument("--loss", type=str, default="svm", choices=["svm", "CE"])
    parser.add_argument(
        "--lr_decay",
        default=True,
        help="Enable linear LR-decay to 0 over epochs",
    )

    para = parser.parse_args()
    para.test_size = 0.2

    # dataset-specific presets (unchanged)
    if para.data == "HHAR":
        para.lr, para.lam, para.p = 0.02, 0.0005, 2
    elif para.data == "Cornell":
        para.lr, para.lam, para.p, para.If_scale = 0.1, 0.005, 2, False
    elif para.data == "USPS":
        para.lr, para.lam, para.p = 0.01, 0.001, 2
    elif para.data == "ISOLET":
        para.lr, para.lam, para.p, para.If_scale = 0.001, 0.001, 2, False
    elif para.data == "ORL":
        para.lr, para.lam, para.p = 0.01, 0.01, 2
    elif para.data == "Dermatology":
        para.lr, para.lam, para.p, para.If_scale = 0.01, 0.1, 2, False
    elif para.data == "Vehicle":
        para.lr, para.lam, para.p = 0.05, 0.0001, 2
    elif para.data == "Glass":
        para.lr, para.lam, para.p = 0.01, 0.0001, 2

    para.lr *= para.lr_scale
    if para.optimizer == "sgd":
        para.lr *= 100  # original heuristic

    R_MLR(para)

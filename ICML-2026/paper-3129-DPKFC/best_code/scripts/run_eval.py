#!/usr/bin/env python3
"""Reproduction evaluation script for DP-KFC MNIST experiment.

Runs the MNIST CNN experiment at epsilon=1.0 with 5 seeds and saves
aggregated results to results/cnn_mnist_results.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from rich.console import Console
from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.trainer import Trainer

EPSILON = 1.0
SEEDS = [42, 7, 91, 23, 58]
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

console = Console()


def build_trainer(device):
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE)
    public_loader, _, _ = get_fashionmnist_loaders(batch_size=BATCH_SIZE)
    model = SimpleCNN(in_channels=1, num_classes=10)
    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        public_loader=public_loader, device=device, learning_rate=LR,
        optimizer_type="adam", is_text=False,
    )
    return trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/bold cyan] {device}")
    trainer = build_trainer(device)
    results = []

    for seed in SEEDS:
        console.print(f"\n[bold]Seed {seed}[/bold]")

        # Plain SGD
        console.print("[green]Plain SGD[/green]")
        history = trainer.train_baseline(epochs=EPOCHS, seed=seed)
        results.append({
            "Method": "Plain SGD", "Epsilon": 0.0, "Seed": seed,
            "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"],
        })

        # DP-SGD
        console.print("[yellow]DP-SGD[/yellow]")
        history = trainer.train_dp_sgd(
            epochs=EPOCHS, epsilon=EPSILON, delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM, seed=seed,
        )
        results.append({
            "Method": "DP-SGD", "Epsilon": EPSILON, "Seed": seed,
            "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"],
        })

        # KFAC (Public)
        console.print("[magenta]KFAC (Public)[/magenta]")
        history = trainer.train_dp_kfac(
            epochs=EPOCHS, epsilon=EPSILON, delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM, seed=seed, use_public_data=True,
        )
        results.append({
            "Method": "KFAC (Public)", "Epsilon": EPSILON, "Seed": seed,
            "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"],
        })

        # KFAC (White Noise)
        console.print("[blue]KFAC (White Noise)[/blue]")
        history = trainer.train_dp_kfac(
            epochs=EPOCHS, epsilon=EPSILON, delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM, seed=seed,
            use_public_data=False, use_pink_noise=False,
        )
        results.append({
            "Method": "KFAC (White Noise)", "Epsilon": EPSILON, "Seed": seed,
            "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"],
        })

        # KFAC (Pink Noise) - Synthetic DP-KFC
        console.print("[red]KFAC (Pink Noise)[/red]")
        history = trainer.train_dp_kfac(
            epochs=EPOCHS, epsilon=EPSILON, delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM, seed=seed,
            use_public_data=False, use_pink_noise=True,
        )
        results.append({
            "Method": "KFAC (Pink Noise)", "Epsilon": EPSILON, "Seed": seed,
            "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"],
        })

    df = pd.DataFrame(results)
    df.to_csv("results/cnn_mnist_results.csv", index=False)

    console.print("\n[bold]Summary (epsilon=1.0):[/bold]")
    for method in ["Plain SGD", "DP-SGD", "KFAC (Public)", "KFAC (White Noise)", "KFAC (Pink Noise)"]:
        sub = df[df["Method"] == method]
        eps_val = 0.0 if method == "Plain SGD" else EPSILON
        acc = sub[sub["Epsilon"] == eps_val]["Accuracy"]
        console.print(f"  {method:<22} mean={acc.mean():.4f}  std={acc.std():.4f}  n={len(acc)}")


if __name__ == "__main__":
    main()

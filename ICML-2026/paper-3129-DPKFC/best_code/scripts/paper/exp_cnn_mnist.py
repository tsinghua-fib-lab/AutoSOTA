"""Experiment: CNN on MNIST -- replicates the paper results."""

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import pandas as pd
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.trainer import Trainer
from dp_kfac.results import save_results_csv, print_summary_table

EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

console = Console()


def build_trainer(device: torch.device) -> Trainer:
    """Create a Trainer wired to MNIST (private) and FashionMNIST (public)."""
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE)
    public_loader, _, _ = get_fashionmnist_loaders(batch_size=BATCH_SIZE)

    model = SimpleCNN(in_channels=1, num_classes=10)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        public_loader=public_loader,
        device=device,
        learning_rate=LR,
        optimizer_type="adam",
        is_text=False,
    )
    return trainer


def extract_final(history: List[Dict[str, Any]]) -> tuple:
    """Return (accuracy, loss) from the last epoch entry."""
    last = history[-1]
    return last["accuracy"], last["test_loss"]


def run_experiments(
    epsilons: List[float],
    seeds: List[int],
    output_dir: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/bold cyan] {device}")
    console.print(f"[bold cyan]Epsilons:[/bold cyan] {epsilons}")
    console.print(f"[bold cyan]Seeds:[/bold cyan] {seeds}")

    trainer = build_trainer(device)

    results: List[Dict[str, Any]] = []

    # Plain SGD baseline
    for seed in seeds:
        console.print(
            f"[bold green]Plain SGD[/bold green]  seed={seed}"
        )
        history = trainer.train_baseline(epochs=EPOCHS, seed=seed)
        acc, loss = extract_final(history)
        results.append({
            "Method": "Plain SGD",
            "Epsilon": 0.0,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(
            f"  -> Accuracy={acc:.4f}  Loss={loss:.4f}"
        )

    # DP-SGD
    for epsilon in epsilons:
        for seed in seeds:
            console.print(
                f"[bold yellow]DP-SGD[/bold yellow]  eps={epsilon}  seed={seed}"
            )
            history = trainer.train_dp_sgd(
                epochs=EPOCHS,
                epsilon=epsilon,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
            )
            acc, loss = extract_final(history)
            results.append({
                "Method": "DP-SGD",
                "Epsilon": epsilon,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(
                f"  -> Accuracy={acc:.4f}  Loss={loss:.4f}"
            )

    # KFAC (Public)
    for epsilon in epsilons:
        for seed in seeds:
            console.print(
                f"[bold magenta]KFAC (Public)[/bold magenta]  eps={epsilon}  seed={seed}"
            )
            history = trainer.train_dp_kfac(
                epochs=EPOCHS,
                epsilon=epsilon,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=True,
            )
            acc, loss = extract_final(history)
            results.append({
                "Method": "KFAC (Public)",
                "Epsilon": epsilon,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(
                f"  -> Accuracy={acc:.4f}  Loss={loss:.4f}"
            )

    # KFAC (White Noise)
    for epsilon in epsilons:
        for seed in seeds:
            console.print(
                f"[bold blue]KFAC (White Noise)[/bold blue]  eps={epsilon}  seed={seed}"
            )
            history = trainer.train_dp_kfac(
                epochs=EPOCHS,
                epsilon=epsilon,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=False,
                use_pink_noise=False,
            )
            acc, loss = extract_final(history)
            results.append({
                "Method": "KFAC (White Noise)",
                "Epsilon": epsilon,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(
                f"  -> Accuracy={acc:.4f}  Loss={loss:.4f}"
            )

    # KFAC (Pink Noise)
    for epsilon in epsilons:
        for seed in seeds:
            console.print(
                f"[bold red]KFAC (Pink Noise)[/bold red]  eps={epsilon}  seed={seed}"
            )
            history = trainer.train_dp_kfac(
                epochs=EPOCHS,
                epsilon=epsilon,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=False,
                use_pink_noise=True,
            )
            acc, loss = extract_final(history)
            results.append({
                "Method": "KFAC (Pink Noise)",
                "Epsilon": epsilon,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(
                f"  -> Accuracy={acc:.4f}  Loss={loss:.4f}"
            )

    output_path = Path(output_dir) / "cnn_mnist_results.csv"
    save_results_csv(
        results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    df = pd.DataFrame(results)
    print_summary_table(df, metric="Accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN on MNIST -- paper experiment replication",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test: single seed (42) and single epsilon (1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a single specific seed (e.g. --seed 42)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Run with a single specific epsilon (e.g. --epsilon 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for output CSV (default: results)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.fast:
        epsilons = [1.0]
        seeds = [42]
        console.print("[bold yellow]FAST MODE:[/bold yellow] seed=42, epsilon=1.0")
    else:
        epsilons = EPSILONS
        seeds = SEEDS

    if args.epsilon is not None:
        epsilons = [args.epsilon]
    if args.seed is not None:
        seeds = [args.seed]

    run_experiments(
        epsilons=epsilons,
        seeds=seeds,
        output_dir=args.output_dir,
    )

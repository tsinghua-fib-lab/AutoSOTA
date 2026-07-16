"""Experiment: CrossViT on CIFAR-100 -- replicating paper results.

The CrossViT backbone is frozen; only the linear classifier head is trained.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from rich.console import Console
from rich.table import Table

from dp_kfac.models import CrossViTClassifier
from dp_kfac.data import get_cifar100_loaders, get_cifar10_loaders
from dp_kfac.trainer import Trainer
from dp_kfac.results import save_results_csv

EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrossViT CIFAR-100 experiment (paper replication)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test: fewer epsilons, seeds, and epochs.",
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
        help="Directory for output CSV (default: results).",
    )
    return parser.parse_args()


def run_experiment(fast: bool, output_dir: str,
                   override_seed: int | None = None,
                   override_epsilon: float | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/bold green] {device}")

    # Adjust for fast mode
    epsilons = EPSILONS[:2] if fast else EPSILONS
    seeds = SEEDS[:2] if fast else SEEDS
    epochs = 1 if fast else EPOCHS

    if override_epsilon is not None:
        epsilons = [override_epsilon]
    if override_seed is not None:
        seeds = [override_seed]

    console.print(f"[bold]Epsilons:[/bold] {epsilons}")
    console.print(f"[bold]Seeds:[/bold]    {seeds}")
    console.print(f"[bold]Epochs:[/bold]   {epochs}")
    console.print(f"[bold]Batch:[/bold]    {BATCH_SIZE}")
    console.print(f"[bold]LR:[/bold]       {LR}")

    console.rule("[bold cyan]Loading datasets")
    console.print("Loading CIFAR-100 (private) ...")
    cifar100_train_loader, cifar100_test_loader, _ = get_cifar100_loaders(
        BATCH_SIZE, img_size=240, use_imagenet_norm=True,
    )

    console.print("Loading CIFAR-10 (public) ...")
    cifar10_train_loader, _, _ = get_cifar10_loaders(
        BATCH_SIZE, img_size=240, use_imagenet_norm=True,
    )

    console.rule("[bold cyan]Initialising CrossViT backbone")
    model = CrossViTClassifier(num_classes=100, img_size=240)

    trainer = Trainer(
        model=model,
        train_loader=cifar100_train_loader,
        test_loader=cifar100_test_loader,
        public_loader=cifar10_train_loader,
        device=device,
        learning_rate=LR,
    )

    # Oracle trainer: uses private data as "public" (upper bound)
    oracle_trainer = Trainer(
        model=model,
        train_loader=cifar100_train_loader,
        test_loader=cifar100_test_loader,
        public_loader=cifar100_train_loader,
        device=device,
        learning_rate=LR,
    )

    all_results: list[dict] = []

    # Plain SGD baseline
    console.rule("[bold cyan]Plain SGD (baseline)")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        history = trainer.train_baseline(epochs=epochs, seed=seed)
        acc = history[-1]["accuracy"]
        loss = history[-1]["test_loss"]
        all_results.append({
            "Method": "Plain SGD",
            "Epsilon": float("inf"),
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    # DP methods
    for eps in epsilons:
        console.rule(f"[bold yellow]Epsilon = {eps}")
        for seed in seeds:
            console.print(f"  [DP-SGD] eps={eps}, seed={seed} ...")
            history = trainer.train_dp_sgd(
                epochs=epochs,
                epsilon=eps,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
            )
            acc = history[-1]["accuracy"]
            loss = history[-1]["test_loss"]
            all_results.append({
                "Method": "DP-SGD",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

            console.print(f"  [Public KFAC] eps={eps}, seed={seed} ...")
            history = trainer.train_dp_kfac(
                epochs=epochs,
                epsilon=eps,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=True,
            )
            acc = history[-1]["accuracy"]
            loss = history[-1]["test_loss"]
            all_results.append({
                "Method": "Public KFAC",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

            console.print(f"  [Pink KFAC] eps={eps}, seed={seed} ...")
            history = trainer.train_dp_kfac(
                epochs=epochs,
                epsilon=eps,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=False,
                use_pink_noise=True,
            )
            acc = history[-1]["accuracy"]
            loss = history[-1]["test_loss"]
            all_results.append({
                "Method": "Pink KFAC",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

            console.print(f"  [Private Oracle KFAC] eps={eps}, seed={seed} ...")
            history = oracle_trainer.train_dp_kfac(
                epochs=epochs,
                epsilon=eps,
                delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                seed=seed,
                use_public_data=True,
            )
            acc = history[-1]["accuracy"]
            loss = history[-1]["test_loss"]
            all_results.append({
                "Method": "Private Oracle KFAC",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    output_path = Path(output_dir) / "crossvit_cifar100_results.csv"
    save_results_csv(
        all_results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    console.rule("[bold cyan]Summary")
    table = Table(title="CrossViT CIFAR-100 Results")
    table.add_column("Method", style="bold")
    table.add_column("Epsilon", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Accuracy (%)", justify="right")
    table.add_column("Loss", justify="right")

    for r in all_results:
        eps_str = "inf" if r["Epsilon"] == float("inf") else f"{r['Epsilon']:.1f}"
        table.add_row(
            r["Method"],
            eps_str,
            str(r["Seed"]),
            f"{r['Accuracy'] * 100:.2f}",
            f"{r['Loss']:.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(fast=args.fast, output_dir=args.output_dir,
                   override_seed=args.seed, override_epsilon=args.epsilon)

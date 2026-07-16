"""Experiment: StackOverflow Duplicate Detection with frozen RoBERTa backbone."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from rich.console import Console
from rich.table import Table

from dp_kfac.data import get_text_loaders
from dp_kfac.models import BERTClassifier
from dp_kfac.trainer import Trainer
from dp_kfac.results import save_results_csv

EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(fast: bool = False, output_dir: str = "results",
                   override_seed: int | None = None,
                   override_epsilon: float | None = None) -> None:
    console = Console()

    epsilons = EPSILONS[:2] if fast else EPSILONS
    seeds = SEEDS[:2] if fast else SEEDS
    epochs = 1 if fast else EPOCHS

    if override_epsilon is not None:
        epsilons = [override_epsilon]
    if override_seed is not None:
        seeds = [override_seed]

    console.rule("[bold cyan]StackOverflow Duplicate Detection Experiment")
    console.print(f"Device:   {DEVICE}")
    console.print(f"Epsilons: {epsilons}")
    console.print(f"Seeds:    {seeds}")
    console.print(f"Epochs:   {epochs}")
    console.print(f"LR:       {LR}")
    console.print(f"Batch:    {BATCH_SIZE}")
    console.print(f"Delta:    {DELTA}")
    console.print()

    console.print("[yellow]Loading datasets...[/yellow]")
    train_loader, test_loader, public_loader, train_size = get_text_loaders(
        "stackoverflow",
        "agnews",
        BATCH_SIZE,
        max_length=MAX_LENGTH,
        max_samples=5000,
        tokenizer_name="bert-base-uncased",
    )
    console.print(f"[green]Train size: {train_size}[/green]")

    model = BERTClassifier(num_classes=2, freeze_backbone=True)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        public_loader,
        DEVICE,
        learning_rate=LR,
        optimizer_type="adam",
        is_text=True,
        vocab_size=30522,
        max_seq_len=MAX_LENGTH,
    )

    results = []

    # Plain SGD baseline
    console.rule("[bold]Plain SGD (no DP)")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        history = trainer.train_baseline(epochs=epochs, seed=seed)
        acc = history[-1]["accuracy"]
        loss = history[-1]["test_loss"]
        results.append({
            "Method": "Plain SGD",
            "Epsilon": 0.0,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    # DP methods
    methods = [
        ("DP-SGD", dict(kfac=False)),
        ("KFAC (Public)", dict(kfac=True, use_public_data=True)),
        ("KFAC (Synthetic)", dict(kfac=True, use_public_data=False, use_pink_noise=False)),
    ]

    for eps in epsilons:
        console.rule(f"[bold]Epsilon = {eps}")
        for method_name, method_kwargs in methods:
            for seed in seeds:
                console.print(f"  {method_name} | eps={eps} | seed={seed} ...")

                if method_kwargs.get("kfac", False):
                    history = trainer.train_dp_kfac(
                        epochs=epochs,
                        epsilon=eps,
                        delta=DELTA,
                        max_grad_norm=MAX_GRAD_NORM,
                        seed=seed,
                        use_public_data=method_kwargs.get("use_public_data", True),
                        use_pink_noise=method_kwargs.get("use_pink_noise", False),
                    )
                else:
                    history = trainer.train_dp_sgd(
                        epochs=epochs,
                        epsilon=eps,
                        delta=DELTA,
                        max_grad_norm=MAX_GRAD_NORM,
                        seed=seed,
                    )

                acc = history[-1]["accuracy"]
                loss = history[-1]["test_loss"]
                results.append({
                    "Method": method_name,
                    "Epsilon": eps,
                    "Seed": seed,
                    "Accuracy": acc,
                    "Loss": loss,
                })
                console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    output_path = Path(output_dir) / "stackoverflow_results.csv"
    save_results_csv(
        results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    table = Table(title="StackOverflow Results Summary")
    table.add_column("Method", style="cyan")
    table.add_column("Epsilon", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Accuracy (%)", justify="right")
    table.add_column("Loss", justify="right")

    for r in results:
        table.add_row(
            r["Method"],
            f"{r['Epsilon']:.1f}",
            str(r["Seed"]),
            f"{r['Accuracy'] * 100:.2f}",
            f"{r['Loss']:.4f}",
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StackOverflow duplicate detection experiment"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run with fewer seeds and epsilons",
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
        help="Directory to save the results CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(fast=args.fast, output_dir=args.output_dir,
                   override_seed=args.seed, override_epsilon=args.epsilon)

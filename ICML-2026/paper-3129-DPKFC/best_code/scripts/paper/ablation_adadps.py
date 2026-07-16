"""Ablation: AdaDPS vs KFAC preconditioning (MNIST -> FashionMNIST)."""

import sys
import argparse
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.trainer import Trainer, set_seed, evaluate
from dp_kfac.methods import estimate_adadps_preconditioner, precondition_per_sample_gradients_adadps
from dp_kfac.privacy import clip_and_noise_gradients
from dp_kfac.results import save_results_csv

from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

EPSILON = 2.0
DELTA = 1e-5
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 15
LR = 1e-2
BATCH_SIZE = 256
MAX_GRAD_NORM = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_adadps(model_template, train_loader, public_loader, test_loader, device,
                 epochs, epsilon, delta, max_grad_norm, seed):
    set_seed(seed)
    model = copy.deepcopy(model_template).to(device)
    model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9)

    train_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    sample_rate = batch_size / train_size
    total_steps = epochs * (train_size // batch_size)
    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                             sample_rate=sample_rate, steps=total_steps, accountant="rdp")

    preconditioner = estimate_adadps_preconditioner(model, public_loader, device)

    accountant = RDPAccountant()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            bs = data.size(0)
            model.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            precondition_per_sample_gradients_adadps(model, preconditioner)
            clip_and_noise_gradients(model, noise_multiplier, max_grad_norm, bs)
            optimizer.step()
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    acc, test_loss = evaluate(model, test_loader, device)
    return acc, test_loss


def run_experiment(fast: bool = False, output_dir: str = "results") -> None:
    console = Console()

    seeds = [42] if fast else SEEDS
    epochs = EPOCHS

    console.rule("[bold cyan]AdaDPS Comparison Ablation")
    console.print(f"Device:        {DEVICE}")
    console.print(f"Epsilon:       {EPSILON}")
    console.print(f"Delta:         {DELTA}")
    console.print(f"Seeds:         {seeds}")
    console.print(f"Epochs:        {epochs}")
    console.print(f"LR:            {LR}")
    console.print(f"Batch size:    {BATCH_SIZE}")
    console.print(f"Max grad norm: {MAX_GRAD_NORM}")
    console.print()

    console.print("[yellow]Loading datasets...[/yellow]")
    mnist_train, mnist_test, mnist_size = get_mnist_loaders(BATCH_SIZE)
    fmnist_train, fmnist_test, fmnist_size = get_fashionmnist_loaders(BATCH_SIZE)
    console.print(f"[green]Public (MNIST) train size:        {mnist_size}[/green]")
    console.print(f"[green]Private (FashionMNIST) train size: {fmnist_size}[/green]")
    console.print()

    model_template = SimpleCNN(in_channels=1, num_classes=10)

    trainer = Trainer(
        model=SimpleCNN(in_channels=1, num_classes=10),
        train_loader=fmnist_train,
        test_loader=fmnist_test,
        public_loader=mnist_train,
        device=DEVICE,
        learning_rate=LR,
        optimizer_type="sgd",
    )

    results = []

    console.rule("[bold]DP-SGD")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        history = trainer.train_dp_sgd(
            epochs=epochs,
            epsilon=EPSILON,
            delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
            seed=seed,
        )
        acc = history[-1]["accuracy"]
        loss = history[-1]["test_loss"]
        results.append({
            "Method": "DP-SGD",
            "Epsilon": EPSILON,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    console.rule("[bold]AdaDPS")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        acc, loss = train_adadps(
            model_template=model_template,
            train_loader=fmnist_train,
            public_loader=mnist_train,
            test_loader=fmnist_test,
            device=DEVICE,
            epochs=epochs,
            epsilon=EPSILON,
            delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
            seed=seed,
        )
        results.append({
            "Method": "AdaDPS",
            "Epsilon": EPSILON,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    console.rule("[bold]KFAC (Public)")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        history = trainer.train_dp_kfac(
            epochs=epochs,
            epsilon=EPSILON,
            delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
            seed=seed,
            use_public_data=True,
        )
        acc = history[-1]["accuracy"]
        loss = history[-1]["test_loss"]
        results.append({
            "Method": "KFAC (Public)",
            "Epsilon": EPSILON,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    console.rule("[bold]KFAC (Pink Noise)")
    for seed in seeds:
        console.print(f"  Seed {seed} ...")
        history = trainer.train_dp_kfac(
            epochs=epochs,
            epsilon=EPSILON,
            delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
            seed=seed,
            use_public_data=False,
            use_pink_noise=True,
        )
        acc = history[-1]["accuracy"]
        loss = history[-1]["test_loss"]
        results.append({
            "Method": "KFAC (Pink Noise)",
            "Epsilon": EPSILON,
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss,
        })
        console.print(f"    Accuracy: {acc * 100:.2f}%  Loss: {loss:.4f}")

    output_path = Path(output_dir) / "adadps_comparison.csv"
    save_results_csv(
        results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    table = Table(title="AdaDPS Comparison Ablation (MNIST -> FashionMNIST)")
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
        description="AdaDPS comparison ablation: MNIST -> FashionMNIST transfer"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run with a single seed (42)",
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
        help="Override epsilon budget (default: 2.0)",
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
    if args.epsilon is not None:
        EPSILON = args.epsilon
    if args.seed is not None:
        SEEDS = [args.seed]
    run_experiment(fast=args.fast, output_dir=args.output_dir)

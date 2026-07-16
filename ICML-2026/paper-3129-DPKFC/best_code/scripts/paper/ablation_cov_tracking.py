"""Ablation: Covariance tracking over DP-SGD training epochs on MNIST."""

import sys
import copy
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from rich.console import Console
from rich.table import Table

from dp_kfac import SimpleCNN, set_seed, evaluate
from dp_kfac.analysis import track_covariances_epoch
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.privacy import clip_and_noise_gradients

EPSILON = 1.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_tracking(fast: bool = False, output_dir: str = "results") -> None:
    console = Console()

    epochs = 3 if fast else EPOCHS

    console.rule("[bold cyan]Covariance Tracking Ablation")
    console.print(f"Device:        {DEVICE}")
    console.print(f"Epsilon:       {EPSILON}")
    console.print(f"Delta:         {DELTA}")
    console.print(f"Epochs:        {epochs}")
    console.print(f"LR:            {LR}")
    console.print(f"Batch size:    {BATCH_SIZE}")
    console.print(f"Max grad norm: {MAX_GRAD_NORM}")
    console.print(f"Seed:          {SEED}")
    console.print()

    set_seed(SEED)

    console.print("[yellow]Loading MNIST (private)...[/yellow]")
    train_loader, test_loader, train_size = get_mnist_loaders(BATCH_SIZE)
    console.print(f"[green]MNIST train size: {train_size}[/green]")

    console.print("[yellow]Loading FashionMNIST (public)...[/yellow]")
    public_loader, _, _ = get_fashionmnist_loaders(BATCH_SIZE)
    console.print("[green]FashionMNIST loaded.[/green]")
    console.print()

    model = SimpleCNN(in_channels=1, num_classes=10).to(DEVICE)
    gsm_model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
    optimizer = torch.optim.Adam(gsm_model.parameters(), lr=LR)

    sample_rate = BATCH_SIZE / train_size
    total_steps = epochs * (train_size // BATCH_SIZE)

    noise_multiplier = get_noise_multiplier(
        target_epsilon=EPSILON,
        target_delta=DELTA,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )

    accountant = RDPAccountant()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    console.print(f"Noise multiplier: {noise_multiplier:.4f}")
    console.print(f"Sample rate:      {sample_rate:.6f}")
    console.print(f"Total steps:      {total_steps}")
    console.print()

    tracking_data = []

    for epoch in range(epochs):
        gsm_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)

            gsm_model.zero_grad(set_to_none=True)
            output = gsm_model(data)
            loss = criterion(output, target)
            loss.backward()

            clip_and_noise_gradients(
                gsm_model, noise_multiplier, MAX_GRAD_NORM, batch_size
            )
            optimizer.step()
            accountant.step(
                noise_multiplier=noise_multiplier, sample_rate=sample_rate
            )

            epoch_loss += loss.item() / batch_size
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        eps_spent = accountant.get_epsilon(delta=DELTA)
        acc, test_loss = evaluate(gsm_model, test_loader, DEVICE)

        console.print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={avg_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"acc={acc * 100:.2f}%  "
            f"eps={eps_spent:.2f}"
        )

        # Track covariances with a clean model copy (no Opacus hooks)
        console.print(f"  [dim]Tracking covariances...[/dim]")
        clean_model = SimpleCNN(in_channels=1, num_classes=10).to(DEVICE)
        clean_model.load_state_dict(gsm_model._module.state_dict())
        snapshot = track_covariances_epoch(
            clean_model,
            train_loader,
            public_loader,
            DEVICE,
            num_classes=10,
            num_batches=10,
            input_shape=(1, 28, 28),
        )
        del clean_model
        tracking_data.append({"epoch": epoch + 1, **snapshot})

        # Log summary for this epoch
        for source_name in ["public", "pink_noise"]:
            source_data = snapshot[source_name]
            cos_A_vals = [v["cos_sim_A"] for v in source_data.values()]
            cos_G_vals = [v["cos_sim_G"] for v in source_data.values()]
            mean_cos_A = sum(cos_A_vals) / len(cos_A_vals) if cos_A_vals else 0.0
            mean_cos_G = sum(cos_G_vals) / len(cos_G_vals) if cos_G_vals else 0.0
            console.print(
                f"  [dim]{source_name}: "
                f"mean cos_sim_A={mean_cos_A:.4f}  "
                f"mean cos_sim_G={mean_cos_G:.4f}[/dim]"
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pkl_path = output_path / "cov_tracking_data.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(tracking_data, f)

    console.print(f"\n[green]Tracking data saved to {pkl_path}[/green]")

    table = Table(title="Covariance Tracking Summary")
    table.add_column("Epoch", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Mean cos_sim_A", justify="right")
    table.add_column("Mean cos_sim_G", justify="right")
    table.add_column("Mean frob_rel_A", justify="right")
    table.add_column("Mean frob_rel_G", justify="right")

    for entry in tracking_data:
        epoch = entry["epoch"]
        for source_name in ["public", "pink_noise"]:
            source_data = entry[source_name]
            layers = list(source_data.values())
            if not layers:
                continue
            mean_cos_A = sum(v["cos_sim_A"] for v in layers) / len(layers)
            mean_cos_G = sum(v["cos_sim_G"] for v in layers) / len(layers)
            mean_frob_A = sum(v["frob_rel_A"] for v in layers) / len(layers)
            mean_frob_G = sum(v["frob_rel_G"] for v in layers) / len(layers)
            table.add_row(
                str(epoch),
                source_name,
                f"{mean_cos_A:.4f}",
                f"{mean_cos_G:.4f}",
                f"{mean_frob_A:.4f}",
                f"{mean_frob_G:.4f}",
            )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Covariance tracking ablation over DP-SGD training epochs"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run with fewer epochs (3 instead of 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed (default: 42)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override the epsilon budget (default: 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the tracking pickle",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        SEED = args.seed
    if args.epsilon is not None:
        EPSILON = args.epsilon
    run_tracking(fast=args.fast, output_dir=args.output_dir)

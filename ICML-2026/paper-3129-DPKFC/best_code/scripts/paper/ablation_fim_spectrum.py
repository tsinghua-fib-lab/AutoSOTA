"""FIM eigenvalue spectrum ablation across architectures and preconditioner sources."""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rich.console import Console
from rich.table import Table

from dp_kfac.analysis import compute_kfac_eigenvalues, compute_condition_number
from dp_kfac.models import MLP, SimpleCNN, TinyViT
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders, get_cifar10_loaders

BATCH_SIZE = 256
NUM_BATCHES = 50
INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_cifar10_grayscale_loader(batch_size: int) -> DataLoader:
    """Load CIFAR-10 resized to 28x28 and converted to single-channel grayscale.

    Uses the same normalisation as the default MNIST loader so that the
    pixel-value distribution is comparable.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _build_models(device: torch.device):
    """Return a dict of fresh model instances keyed by short name."""
    mlp = MLP(input_dim=784, num_classes=NUM_CLASSES).to(device)
    cnn = SimpleCNN(in_channels=1, num_classes=NUM_CLASSES, img_size=28).to(device)
    vit = TinyViT(img_size=28, in_channels=1, num_classes=NUM_CLASSES).to(device)
    return {"mlp": mlp, "cnn": cnn, "attention": vit}


def _build_sources(batch_size: int):
    """Return an ordered list of (source_name, loader, noise_type, input_shape).

    For real-data sources ``noise_type`` is ``None`` and ``input_shape`` is
    ``None``.  For synthetic sources the loader is re-used only to obtain
    ``batch_size``; actual data comes from the noise generator.
    """
    mnist_train, _, _ = get_mnist_loaders(batch_size)
    fmnist_train, _, _ = get_fashionmnist_loaders(batch_size)
    cifar10_gray = _get_cifar10_grayscale_loader(batch_size)

    sources = [
        ("Oracle (Private)", mnist_train, None, None),
        ("Public (FashionMNIST)", fmnist_train, None, None),
        ("Public (CIFAR-10)", cifar10_gray, None, None),
        ("Pink Noise", mnist_train, "pink", INPUT_SHAPE),
        ("White Noise", mnist_train, "white", INPUT_SHAPE),
    ]
    return sources


def run_experiment(fast: bool = False, output_dir: str = "results") -> None:
    console = Console()
    num_batches = 5 if fast else NUM_BATCHES

    console.rule("[bold cyan]FIM Eigenvalue Spectrum Ablation")
    console.print(f"Device:      {DEVICE}")
    console.print(f"Batch size:  {BATCH_SIZE}")
    console.print(f"Num batches: {num_batches}")
    console.print()

    console.print("[yellow]Loading data sources...[/yellow]")
    sources = _build_sources(BATCH_SIZE)
    console.print("[green]Data sources ready.[/green]\n")

    results = {}

    for model_name in ("mlp", "cnn", "attention"):
        results[model_name] = {}
        console.rule(f"[bold]Model: {model_name.upper()}")

        for source_name, loader, noise_type, input_shape in sources:
            console.print(
                f"  [cyan]{source_name}[/cyan] — computing eigenvalues "
                f"({num_batches} batches)..."
            )

            models = _build_models(DEVICE)
            model = models[model_name]

            eig_data = compute_kfac_eigenvalues(
                model,
                loader,
                DEVICE,
                num_batches=num_batches,
                num_classes=NUM_CLASSES,
                noise_type=noise_type,
                input_shape=input_shape,
            )

            results[model_name][source_name] = eig_data

            for layer_name, eigs in eig_data.items():
                kappa_A = compute_condition_number(eigs["eig_A"])
                kappa_G = compute_condition_number(eigs["eig_G"])
                console.print(
                    f"    {layer_name:30s}  "
                    f"kappa(A)={kappa_A:12.2f}  "
                    f"kappa(G)={kappa_G:12.2f}"
                )

        console.print()

    for model_name in ("mlp", "cnn", "attention"):
        table = Table(
            title=f"Condition Numbers — {model_name.upper()}",
            show_lines=True,
        )
        table.add_column("Source", style="cyan")
        table.add_column("Layer", style="white")
        table.add_column("kappa(A)", justify="right")
        table.add_column("kappa(G)", justify="right")

        for source_name in results[model_name]:
            for layer_name, eigs in results[model_name][source_name].items():
                kappa_A = compute_condition_number(eigs["eig_A"])
                kappa_G = compute_condition_number(eigs["eig_G"])
                table.add_row(
                    source_name,
                    layer_name,
                    f"{kappa_A:.2f}",
                    f"{kappa_G:.2f}",
                )

        console.print(table)
        console.print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pkl_path = output_path / "fim_spectrum_data.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    console.print(f"[green]Results saved to {pkl_path}[/green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FIM eigenvalue spectrum ablation experiment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output pickle file (default: results)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run with fewer batches (5 instead of 50)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(fast=args.fast, output_dir=args.output_dir)

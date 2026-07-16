"""IMDB Logistic Regression with DP-KFAC vs AdaDPS."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from rich.console import Console

from dp_kfac.data import get_imdb_data, get_agnews_data, get_tfidf_features
from dp_kfac.models import LogisticRegression
from dp_kfac.trainer import Trainer, evaluate, set_seed
from dp_kfac.privacy import clip_and_noise_gradients
from dp_kfac.methods import estimate_adadps_preconditioner, precondition_per_sample_gradients_adadps
from dp_kfac.results import save_results_csv

EPSILONS = [0.5, 1.0, 1.5, 2.8, 8.0]
SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
EPOCHS = 32
LR = 0.1
BATCH_SIZE = 256
MAX_GRAD_NORM_DPSGD = 0.1
MAX_GRAD_NORM_KFAC = 2.0
MAX_FEATURES = 10000
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

console = Console()


def train_adadps(
    model: nn.Module,
    train_loader: DataLoader,
    public_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    seed: int,
) -> tuple:
    """Train with AdaDPS diagonal preconditioning."""
    set_seed(seed)
    model = model.to(device)
    model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    # Compute noise multiplier
    train_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size or BATCH_SIZE
    sample_rate = batch_size / train_size
    total_steps = epochs * (train_size // batch_size)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )

    # Estimate preconditioner from public data
    preconditioner = estimate_adadps_preconditioner(model, public_loader, device)

    accountant = RDPAccountant()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            model.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            precondition_per_sample_gradients_adadps(model, preconditioner)
            clip_and_noise_gradients(model, noise_multiplier, max_grad_norm, data.size(0))
            optimizer.step()
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    acc, test_loss = evaluate(model, test_loader, device)
    return acc, test_loss


def main():
    parser = argparse.ArgumentParser(
        description="IMDB logistic regression: DP-KFAC vs AdaDPS"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test with fewer epsilons and seeds",
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
        help="Directory to save result CSV (default: results)",
    )
    args = parser.parse_args()

    epsilons = EPSILONS if not args.fast else [1.0, 4.0]
    seeds = SEEDS if not args.fast else [42, 43]
    epochs = EPOCHS if not args.fast else 2

    if args.epsilon is not None:
        epsilons = [args.epsilon]
    if args.seed is not None:
        seeds = [args.seed]

    console.print("[bold cyan]Loading IMDB data...[/bold cyan]")
    train_texts, train_labels, test_texts, test_labels = get_imdb_data()

    console.print("[bold cyan]Loading AG News (public proxy)...[/bold cyan]")
    public_texts, public_labels = get_agnews_data()

    console.print("[bold cyan]Extracting TF-IDF features...[/bold cyan]")
    train_feats, test_feats, public_feats = get_tfidf_features(
        train_texts, test_texts, public_texts=public_texts, max_features=MAX_FEATURES
    )

    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)
    public_labels_t = torch.tensor(public_labels, dtype=torch.long)

    input_dim = train_feats.shape[1]

    train_ds = TensorDataset(train_feats, train_labels_t)
    test_ds = TensorDataset(test_feats, test_labels_t)
    public_ds = TensorDataset(public_feats, public_labels_t)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    public_loader = DataLoader(
        public_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    train_size = len(train_ds)
    delta = 1.0 / train_size

    console.print(
        f"[green]Train: {train_size}  Test: {len(test_ds)}  "
        f"Public: {len(public_ds)}  delta={delta:.2e}[/green]"
    )

    results = []
    base_model = LogisticRegression(input_dim=input_dim, num_classes=NUM_CLASSES)

    trainer_dpsgd = Trainer(
        model=base_model,
        train_loader=train_loader,
        test_loader=test_loader,
        public_loader=public_loader,
        device=DEVICE,
        learning_rate=LR,
        optimizer_type="sgd",
        is_text=False,
    )

    trainer_kfac = Trainer(
        model=base_model,
        train_loader=train_loader,
        test_loader=test_loader,
        public_loader=public_loader,
        device=DEVICE,
        learning_rate=LR,
        optimizer_type="sgd",
        is_text=False,
    )

    for eps in epsilons:
        console.print(f"\n[bold yellow]===  Epsilon = {eps}  ===[/bold yellow]")

        for seed in seeds:
            console.print(f"[dim]  Seed {seed}[/dim]")

            console.print("    [cyan]DP-SGD[/cyan]")
            run = trainer_dpsgd.train_dp_sgd(
                epochs=epochs,
                epsilon=eps,
                delta=delta,
                max_grad_norm=MAX_GRAD_NORM_DPSGD,
                seed=seed,
            )
            acc, loss = run[-1]["accuracy"], run[-1]["test_loss"]
            results.append({
                "Method": "DP-SGD",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"      acc={acc:.4f}  loss={loss:.4f}")

            console.print("    [cyan]KFAC (Public)[/cyan]")
            run = trainer_kfac.train_dp_kfac(
                epochs=epochs,
                epsilon=eps,
                delta=delta,
                max_grad_norm=MAX_GRAD_NORM_KFAC,
                seed=seed,
                use_public_data=True,
            )
            acc, loss = run[-1]["accuracy"], run[-1]["test_loss"]
            results.append({
                "Method": "KFAC (Public)",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"      acc={acc:.4f}  loss={loss:.4f}")

            console.print("    [cyan]KFAC (Synthetic)[/cyan]")
            run = trainer_kfac.train_dp_kfac(
                epochs=epochs,
                epsilon=eps,
                delta=delta,
                max_grad_norm=MAX_GRAD_NORM_KFAC,
                seed=seed,
                use_public_data=False,
                use_pink_noise=False,
            )
            acc, loss = run[-1]["accuracy"], run[-1]["test_loss"]
            results.append({
                "Method": "KFAC (Synthetic)",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"      acc={acc:.4f}  loss={loss:.4f}")

            console.print("    [cyan]AdaDPS[/cyan]")
            adadps_model = LogisticRegression(
                input_dim=input_dim, num_classes=NUM_CLASSES
            )
            acc, loss = train_adadps(
                model=adadps_model,
                train_loader=train_loader,
                public_loader=public_loader,
                test_loader=test_loader,
                device=DEVICE,
                epochs=epochs,
                epsilon=eps,
                delta=delta,
                max_grad_norm=MAX_GRAD_NORM_KFAC,
                seed=seed,
            )
            results.append({
                "Method": "AdaDPS",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss,
            })
            console.print(f"      acc={acc:.4f}  loss={loss:.4f}")

    output_dir = Path(args.output_dir)
    output_path = output_dir / "imdb_logreg_results.csv"
    save_results_csv(
        results,
        output_path,
        columns=["Method", "Epsilon", "Seed", "Accuracy", "Loss"],
    )
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  {'Method':<20} {'Epsilon':>8} {'Seed':>6} {'Accuracy':>10} {'Loss':>10}")
    console.print("  " + "-" * 60)
    for r in results:
        console.print(
            f"  {r['Method']:<20} {r['Epsilon']:>8.1f} {r['Seed']:>6} "
            f"{r['Accuracy']:>10.4f} {r['Loss']:>10.4f}"
        )


if __name__ == "__main__":
    main()

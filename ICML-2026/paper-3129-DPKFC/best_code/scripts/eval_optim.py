#!/usr/bin/env python3
"""Focused evaluation for SOTA optimization of DP-KFC on MNIST."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from rich.console import Console
from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.trainer import Trainer
from dp_kfac.types import KFACConfig

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="DP-KFC optimization eval")
    p.add_argument("--seeds", type=str, default="42,7,91,23,58")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--kfac-damping", type=float, default=1e-3)
    p.add_argument("--precond-steps", type=int, default=1)
    p.add_argument("--pink-noise-alpha", type=float, default=1.0)
    p.add_argument("--cov-ema-decay", type=float, default=0.0)
    p.add_argument("--update-freq", type=int, default=1)
    p.add_argument("--alpha-schedule-start", type=float, default=0.0)
    p.add_argument("--alpha-schedule-end", type=float, default=0.0)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-dp-sgd", action="store_true")
    p.add_argument("--lr-warmup-epochs", type=int, default=0)
    p.add_argument("--lr-cosine", action="store_true")
    p.add_argument("--output", type=str, default="results/optim_results.csv")
    return p.parse_args()


def build_trainer(device, args):
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=args.batch_size)
    public_loader, _, _ = get_fashionmnist_loaders(batch_size=args.batch_size)
    model = SimpleCNN(in_channels=1, num_classes=10)
    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        public_loader=public_loader, device=device, learning_rate=args.lr,
        optimizer_type=args.optimizer, is_text=False,
    )
    return trainer


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/bold cyan] {device}")
    console.print(f"[bold cyan]Seeds:[/bold cyan] {seeds}")
    console.print(f"[bold cyan]Config:[/bold cyan] lr={args.lr}, bs={args.batch_size}, "
                  f"epochs={args.epochs}, eps={args.epsilon}")
    console.print(f"[bold cyan]KFAC:[/bold cyan] damping={args.kfac_damping}, "
                  f"precond_steps={args.precond_steps}, alpha={args.pink_noise_alpha}")
    if args.alpha_schedule_start > 0:
        console.print(f"[bold cyan]Alpha schedule:[/bold cyan] "
                      f"{args.alpha_schedule_start} -> {args.alpha_schedule_end}")
    if args.lr_warmup_epochs > 0:
        console.print(f"[bold cyan]LR warmup:[/bold cyan] {args.lr_warmup_epochs} epochs")
    if args.lr_cosine:
        console.print(f"[bold cyan]LR schedule:[/bold cyan] cosine decay")

    kfac_config = KFACConfig(
        damping=args.kfac_damping,
        cov_ema_decay=args.cov_ema_decay,
        update_freq=args.update_freq,
        precond_steps=args.precond_steps,
        pink_noise_alpha=args.pink_noise_alpha,
        alpha_schedule_start=args.alpha_schedule_start,
        alpha_schedule_end=args.alpha_schedule_end,
    )

    import random, os, numpy as np
    results = []

    for seed in seeds:
        console.print(f"\n[bold]Seed {seed}[/bold]")

        # Plain SGD
        if not args.skip_baseline:
            console.print("[green]Plain SGD[/green]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            sgd_trainer = build_trainer(device, args)
            history = sgd_trainer.train_baseline(epochs=args.epochs, seed=seed)
            results.append({"Method": "Plain SGD", "Epsilon": 0.0, "Seed": seed,
                           "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"]})

        # DP-SGD
        if not args.skip_dp_sgd:
            console.print("[yellow]DP-SGD[/yellow]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            dp_trainer = build_trainer(device, args)
            history = dp_trainer.train_dp_sgd(
                epochs=args.epochs, epsilon=args.epsilon, delta=args.delta,
                max_grad_norm=args.max_grad_norm, seed=seed)
            results.append({"Method": "DP-SGD", "Epsilon": args.epsilon, "Seed": seed,
                           "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"]})

        # KFAC (Pink Noise) - primary
        console.print("[red]KFAC (Pink Noise)[/red]")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        kfac_trainer = build_trainer(device, args)
        history = kfac_trainer.train_dp_kfac(
            epochs=args.epochs, epsilon=args.epsilon, delta=args.delta,
            max_grad_norm=args.max_grad_norm, seed=seed,
            use_public_data=False, use_pink_noise=True, kfac_config=kfac_config,
            lr_warmup_epochs=args.lr_warmup_epochs, lr_cosine=args.lr_cosine)
        results.append({"Method": "KFAC (Pink Noise)", "Epsilon": args.epsilon, "Seed": seed,
                       "Accuracy": history[-1]["accuracy"], "Loss": history[-1]["test_loss"]})

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    console.print(f"\n[bold]Results:[/bold]")
    for method in ["Plain SGD", "DP-SGD", "KFAC (Pink Noise)"]:
        sub = df[df["Method"] == method]
        if len(sub) == 0:
            continue
        eps_val = 0.0 if method == "Plain SGD" else args.epsilon
        acc = sub[sub["Epsilon"] == eps_val]["Accuracy"]
        console.print(f"  {method:<22} mean={acc.mean():.4f}  std={acc.std():.4f}  n={len(acc)}")

    pink = df[df["Method"] == "KFAC (Pink Noise)"]
    pink = pink[pink["Epsilon"] == args.epsilon]
    if len(pink) > 0:
        acc_mean = pink["Accuracy"].mean()
        acc_std = pink["Accuracy"].std()
        console.print(f"\n[bold green]PRIMARY: KFAC Pink Noise Accuracy = "
                      f"{acc_mean:.4f} +/- {acc_std:.4f}[/bold green]")


if __name__ == "__main__":
    main()

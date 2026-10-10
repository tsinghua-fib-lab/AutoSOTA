"""Command-line argument parsing and dispatch only."""

from __future__ import annotations
import argparse
from pathlib import Path
from .data import pack_command
from .training import train_command
from .inference import baseline_command, infer_command
from .evaluation import evaluate_command


def build_parser():
    parser = argparse.ArgumentParser(
        description="ODELIA final independent-head method and identity baseline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser(
        "pack",
        help="convert seven-fold manifests and sample caches to one training NPZ",
    )
    pack.add_argument("--cache-root", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    train = commands.add_parser(
        "train", help="train and save seven heads using training and validation only"
    )
    train.add_argument("--packed-cache", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--strict-historical", action="store_true")
    for name in ("infer", "baseline"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--inputs",
            type=Path,
            help="NPZ with sample_ids, features and released_logits",
        )
        sub.add_argument(
            "--predictions",
            type=Path,
            help="legacy MeisenMeister per-fold predictions JSON",
        )
        sub.add_argument(
            "--block-features",
            type=Path,
            help="legacy NPZ descriptors, aligned by sample ID",
        )
        sub.add_argument("--output-dir", required=True, type=Path)
        if name == "infer":
            sub.add_argument("--run-dir", required=True, type=Path)
            sub.add_argument("--device", default="cuda:0")
    evaluate = commands.add_parser(
        "evaluate", help="score saved predictions without changing models"
    )
    evaluate.add_argument("--prediction-archive", required=True, type=Path)
    evaluate.add_argument("--labels", required=True, type=Path)
    evaluate.add_argument("--output", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train" and (args.epochs < 0 or args.batch_size < 1):
        raise ValueError("epochs must be nonnegative and batch size positive")
    return {
        "pack": pack_command,
        "train": train_command,
        "infer": infer_command,
        "baseline": baseline_command,
        "evaluate": evaluate_command,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

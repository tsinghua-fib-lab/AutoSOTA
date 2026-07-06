#!/usr/bin/env python3
"""
Export MNIST or Fashion-MNIST into plain-text files that the VHDL testbench can read.

Each image line contains 784 integers in [0, 255], separated by spaces.
Each label line contains one integer in [0, 9].

Examples
--------
python tools/export_dataset_to_txt.py --dataset mnist  --outdir data --n-train 256 --n-test 64
python tools/export_dataset_to_txt.py --dataset fmnist --outdir data --n-train 256 --n-test 64
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from torchvision import datasets
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "torchvision is required for this exporter. "
        "Install PyTorch + torchvision first."
    ) from exc


def _load_dataset(name: str, root: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    name = name.lower()
    if name == "mnist":
        train_ds = datasets.MNIST(root=str(root), train=True, download=True)
        test_ds = datasets.MNIST(root=str(root), train=False, download=True)
    elif name in {"fmnist", "fashion-mnist", "fashion_mnist"}:
        train_ds = datasets.FashionMNIST(root=str(root), train=True, download=True)
        test_ds = datasets.FashionMNIST(root=str(root), train=False, download=True)
        name = "fmnist"
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    x_train = train_ds.data.numpy().astype(np.uint8)
    y_train = np.array(train_ds.targets, dtype=np.int64)
    x_test = test_ds.data.numpy().astype(np.uint8)
    y_test = np.array(test_ds.targets, dtype=np.int64)
    return (x_train, y_train), (x_test, y_test)


def _pick_subset(
    x: np.ndarray,
    y: np.ndarray,
    count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    indices = list(range(len(x)))
    rng.shuffle(indices)
    indices = indices[:count]
    return x[indices], y[indices]


def _write_images(path: Path, images: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for img in images:
            flat = img.reshape(-1)
            f.write(" ".join(str(int(v)) for v in flat))
            f.write("\n")


def _write_labels(path: Path, labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(f"{int(lbl)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fmnist"], required=True)
    parser.add_argument("--root", type=Path, default=Path("./dataset_cache"))
    parser.add_argument("--outdir", type=Path, default=Path("./data"))
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = _load_dataset(args.dataset, args.root)

    x_train, y_train = _pick_subset(x_train, y_train, args.n_train, args.seed)
    x_test, y_test = _pick_subset(x_test, y_test, args.n_test, args.seed + 1)

    prefix = "mnist" if args.dataset == "mnist" else "fmnist"

    train_img_path = args.outdir / f"{prefix}_train_images.txt"
    train_lbl_path = args.outdir / f"{prefix}_train_labels.txt"
    test_img_path = args.outdir / f"{prefix}_test_images.txt"
    test_lbl_path = args.outdir / f"{prefix}_test_labels.txt"

    _write_images(train_img_path, x_train)
    _write_labels(train_lbl_path, y_train)
    _write_images(test_img_path, x_test)
    _write_labels(test_lbl_path, y_test)

    print(f"Wrote {len(x_train)} training samples to {train_img_path}")
    print(f"Wrote {len(x_test)} test samples to {test_img_path}")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate k-NN probe accuracy on saved VDW embeddings.

Example:
    python tests/run_embedding_knn.py --fold_dir /path/to/experiments/.../fold_0 --k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Ensure the repository root (code directory) is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from models.custom_metrics import compute_knn_probe_accuracy


def load_split(
    split_name: str,
    fold_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and labels for a specific split.
    """
    canonical_name = split_name
    split_filename = f"{canonical_name}_embeddings.pt"
    split_path = os.path.join(fold_dir, "embeddings", split_filename)

    if (not os.path.exists(split_path)) and (canonical_name == "valid"):
        alt_path = os.path.join(fold_dir, "embeddings", "val_embeddings.pt")
        if os.path.exists(alt_path):
            split_path = alt_path

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Could not find embeddings for split '{split_name}' at {split_path}"
        )

    payload = torch.load(split_path, map_location="cpu")
    embeddings = payload["embeddings"].detach().cpu().numpy()
    labels = payload["labels"].detach().cpu().numpy()
    return embeddings, labels


def _np_to_tensor(
    array: np.ndarray,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a NumPy array to a torch tensor with the requested dtype.
    """
    tensor = torch.from_numpy(array)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def parse_args() -> argparse.Namespace:
    """
    CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate k-NN probe accuracy on VDW embeddings."
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Path to the fold_* directory containing the embeddings/ folder.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of neighbors to use in the probe (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()
    fold_dir = os.path.abspath(args.fold_dir)

    print(f"[INFO] Loading embeddings from: {fold_dir}")
    train_embeddings_np, train_labels_np = load_split("train", fold_dir)
    valid_embeddings_np, valid_labels_np = load_split("valid", fold_dir)
    test_embeddings_np, test_labels_np = load_split("test", fold_dir)

    train_embeddings = _np_to_tensor(train_embeddings_np, dtype=torch.float32)
    valid_embeddings = _np_to_tensor(valid_embeddings_np, dtype=torch.float32)
    test_embeddings = _np_to_tensor(test_embeddings_np, dtype=torch.float32)

    train_labels = _np_to_tensor(train_labels_np, dtype=torch.long)
    valid_labels = _np_to_tensor(valid_labels_np, dtype=torch.long)
    test_labels = _np_to_tensor(test_labels_np, dtype=torch.long)

    print(f"[INFO] Evaluating k-NN probe with k={args.k}")
    valid_acc = compute_knn_probe_accuracy(
        train_embeddings,
        train_labels,
        valid_embeddings,
        valid_labels,
        k=args.k,
    )
    test_acc = compute_knn_probe_accuracy(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        k=args.k,
    )

    print(f"[RESULT] Validation accuracy: {valid_acc.item():.4f}")
    print(f"[RESULT] Test accuracy: {test_acc.item():.4f}")


if __name__ == "__main__":
    main()


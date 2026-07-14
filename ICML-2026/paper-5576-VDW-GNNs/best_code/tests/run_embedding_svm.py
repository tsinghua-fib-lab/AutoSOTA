#!/usr/bin/env python3
"""
Grid-search SVM on saved VDW embeddings.

Example:
    python tests/run_embedding_svm.py \
    --fold_dir /path/to/experiments/.../fold_0 \
    --kernel rbf \
    --Cs 0.8 0.9 1.0 1.1 1.2 \
    --gammas 0.001 0.01 0.1 1.0
"""
import argparse
import os
from itertools import product
from typing import Tuple, Sequence, List

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


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

    payload = torch.load(split_path, map_location="cpu", weights_only=True)
    embeddings = payload["embeddings"].detach().cpu().numpy()
    labels = payload["labels"].detach().cpu().numpy()
    return embeddings, labels


def run_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    kernel: str,
    c_values: Sequence[float],
    gamma_values: Sequence[float],
    use_auto_gamma: bool,
) -> Tuple[SVC, float, Tuple[float, float], List[Tuple[float, float, float]]]:
    """
    Train SVMs over a grid of hyperparameters and select the best validation score.
    """
    best_score = -1.0
    best_model = None
    best_params = (None, None)
    score_rows: List[Tuple[float, float, float]] = []

    if use_auto_gamma:
        gamma_values = ['scale']

    for C_val, gamma_val in product(c_values, gamma_values):
        model = SVC(C=C_val, gamma=gamma_val, kernel=kernel)
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        score = accuracy_score(y_valid, preds)
        score_rows.append((C_val, gamma_val, score))

        if score > best_score:
            best_score = score
            best_model = model
            best_params = (C_val, gamma_val)

    if best_model is None:
        raise RuntimeError("Grid search failed to train any models.")

    return best_model, best_score, best_params, score_rows


def parse_args() -> argparse.Namespace:
    """
    CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Grid-search SVM on VDW embeddings."
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Path to the fold_* directory containing the embeddings/ folder.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["linear", "poly", "rbf", "sigmoid"],
        default="rbf",
        help="Kernel type to use.",
    )
    parser.add_argument(
        "--Cs",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
        help="List of C values to try.",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="List of gamma values to try.",
    )
    parser.add_argument(
        "--use_auto_gamma",
        action="store_true",
        help="Use auto-gamma mode ('scale' in sklearn's SVC)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()
    fold_dir = os.path.abspath(args.fold_dir)

    print(f"[INFO] Loading embeddings from: {fold_dir}")
    x_train, y_train = load_split("train", fold_dir)
    x_valid, y_valid = load_split("valid", fold_dir)
    x_test, y_test = load_split("test", fold_dir)

    best_model, best_val_score, best_params, score_rows = run_grid_search(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        kernel=args.kernel,
        c_values=args.Cs,
        gamma_values=args.gammas,
        use_auto_gamma=args.use_auto_gamma,
    )

    print("\n[RESULT] Validation accuracy grid:")
    header = f"{'C':>10} | {'gamma':>10} | {'val_acc':>10}"
    print(header)
    print("-" * len(header))
    for C_val, gamma_val, score in score_rows:
        if gamma_val == 'scale':
            gamma_val_str = 'scale'
        else:
            gamma_val_str = f"{gamma_val:10.4g}"
        print(f"{C_val:10.4g} | {gamma_val_str} | {score:10.4f}")

    print(
        f"[RESULT] Best params: C={best_params[0]}, gamma={best_params[1]} "
        f"(validation accuracy: {best_val_score:.4f})"
    )

    test_preds = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"[RESULT] Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    print("\n[RESULT] Test confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()


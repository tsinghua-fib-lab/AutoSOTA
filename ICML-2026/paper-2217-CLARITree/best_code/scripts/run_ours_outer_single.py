"""Run a single outer fold evaluation at fixed (lambda, kappa).

Like run_ours_outer.py but evaluates only one hyperparameter combo.
Used by eval_california_housing_fast.py for quick optimization iterations.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from clari_tree import CLARITree, Greedy


METHODS = {
    "claritree": CLARITree,
    "greedy": Greedy,
}
RESULT_COLUMNS = [
    "dataset",
    "outer",
    "method",
    "depth",
    "lambda",
    "kappa",
    "cv_folds",
    "n_thresholds",
    "thresholds_strategy",
    "min_leaf_node_size",
    "val_r2",
    "val_mse",
    "loss",
    "train_time_s",
    "train_mse",
    "train_r2",
    "test_mse",
    "test_r2",
    "n_leaves",
]


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def load_xy(csv_path):
    df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument("--thresholds_strategy", default="quantile")
    parser.add_argument("--min_leaf_node_size", type=int, default=0)
    parser.add_argument("--lambda", type=float, required=True, dest="lambda_")
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--refine_kappa_factor", type=float, default=1.0,
                        help="Factor for post-training leaf coefficient refinement (1.0 = no refinement)")

    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label.",
    )
    args = parser.parse_args()

    model_cls = METHODS[args.method]
    split_dir = Path("data") / args.name / "splits" / f"outer_{args.outer}"
    X_train, y_train = load_xy(split_dir / "train.csv")
    X_test, y_test = load_xy(split_dir / "test.csv")

    lambda_ = args.lambda_
    kappa = args.kappa

    print("=" * 80)
    print(f"Dataset: {args.name}")
    print(f"Outer: outer_{args.outer}")
    print(f"Method: {args.method}")
    print(f"Depth: {args.depth}")
    print(f"CV folds: {args.cv_folds}")
    print(f"n_thresholds: {args.n_thresholds}")
    print(f"lambda: {lambda_}, kappa: {kappa}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("=" * 80)
    sys.stdout.flush()

    start_all = time.time()

    # 3-fold CV for validation score
    val_r2s, val_mses = [], []
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        model = model_cls(
            kappa=float(kappa),
            depth=int(args.depth),
            lambda_=float(lambda_),
            n_thresholds=int(args.n_thresholds),
            thresholds_strategy=args.thresholds_strategy,
            verbose=False,
            min_leaf_node_size=int(args.min_leaf_node_size),
        )
        model.refine_kappa_factor = args.refine_kappa_factor
        model.fit(X_train[train_idx], y_train[train_idx])
        pred_val = model.predict(X_train[val_idx])
        val_r2s.append(r2(y_train[val_idx], pred_val))
        val_mses.append(mse(y_train[val_idx], pred_val))

    val_r2 = float(np.mean(val_r2s))
    val_mse = float(np.mean(val_mses))

    # Final fit on full training set, evaluate on test
    model = model_cls(
        kappa=float(kappa),
        depth=int(args.depth),
        lambda_=float(lambda_),
        n_thresholds=int(args.n_thresholds),
        thresholds_strategy=args.thresholds_strategy,
        verbose=False,
        min_leaf_node_size=int(args.min_leaf_node_size),
    )
    model.refine_kappa_factor = args.refine_kappa_factor
    t0 = time.time()
    loss = model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    final = {
        "loss": float(loss),
        "train_time_s": float(train_time),
        "train_mse": mse(y_train, pred_train),
        "train_r2": r2(y_train, pred_train),
        "test_mse": mse(y_test, pred_test),
        "test_r2": r2(y_test, pred_test),
        "n_leaves": int(model.n_leaves()),
    }

    # Determine threshold directory label
    tlabel = args.threshold_label if args.threshold_label else str(args.n_thresholds)

    out_dir = (
        Path("results")
        / "ours"
        / args.method
        / f"linear_regression_tree_depth{args.depth}_threshold_{tlabel}"
        / args.name
        / f"outer{args.outer}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = Path(args.name).name
    out_csv = out_dir / f"{dataset_filename}_outer{args.outer}_d{args.depth}.csv"

    row = {
        "dataset": args.name,
        "outer": f"outer_{args.outer}",
        "method": args.method,
        "depth": args.depth,
        "lambda": lambda_,
        "kappa": kappa,
        "cv_folds": args.cv_folds,
        "n_thresholds": args.n_thresholds,
        "thresholds_strategy": args.thresholds_strategy,
        "min_leaf_node_size": args.min_leaf_node_size,
        "val_r2": val_r2,
        "val_mse": val_mse,
        **final,
    }

    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(out_csv, index=False)

    print(f"val_r2={val_r2:.6f}, test_r2={final['test_r2']:.6f}, "
          f"train_r2={final['train_r2']:.6f}, leaves={final['n_leaves']}, "
          f"time={train_time:.1f}s")
    print("=" * 80)
    print(f"Saved to {out_csv}")
    print(f"Total time: {(time.time() - start_all) / 60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()

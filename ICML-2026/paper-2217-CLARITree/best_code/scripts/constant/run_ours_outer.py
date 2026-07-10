from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from clari_tree import CLARITreeConst, GreedyConst


LAMBDAS = [0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0005, 0.0001]
METHODS = {
    "claritree": CLARITreeConst,
    "greedy": GreedyConst,
}
RESULT_COLUMNS = [
    "dataset",
    "outer",
    "method",
    "depth",
    "lambda",
    "cv_folds",
    "n_thresholds",
    "threshold_label",
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


def fit_eval(model_cls, args, lambda_, X_train, y_train, X_test, y_test):
    model = model_cls(
        depth=int(args.depth),
        lambda_=float(lambda_),
        n_thresholds=int(args.n_thresholds),
        thresholds_strategy=args.thresholds_strategy,
        verbose=False,
        min_leaf_node_size=int(args.min_leaf_node_size),
    )

    start = time.time()
    loss = model.fit(X_train, y_train)
    train_time = time.time() - start

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "loss": float(loss),
        "train_time_s": float(train_time),
        "train_mse": mse(y_train, pred_train),
        "train_r2": r2(y_train, pred_train),
        "test_mse": mse(y_test, pred_test),
        "test_r2": r2(y_test, pred_test),
        "n_leaves": int(model.n_leaves()),
    }


def cv_score(model_cls, args, lambda_, X, y):
    val_r2s, val_mses = [], []
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        res = fit_eval(
            model_cls,
            args,
            lambda_,
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
        )
        val_r2s.append(res["test_r2"])
        val_mses.append(res["test_mse"])
    return float(np.mean(val_r2s)), float(np.mean(val_mses))


def threshold_dir_label(args):
    return str(args.n_thresholds)


def output_csv(args):
    out_dir = (
        Path("results")
        / "ours"
        / args.method
        / f"constant_regression_tree_depth{args.depth}_threshold_{threshold_dir_label(args)}"
        / args.name
        / f"outer{args.outer}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = Path(args.name).name
    return out_dir / f"{dataset_filename}_outer{args.outer}_d{args.depth}.csv"


def run_outer(args):
    model_cls = METHODS[args.method]
    split_dir = Path("data") / args.name / "splits" / f"outer_{args.outer}"
    X_train, y_train = load_xy(split_dir / "train.csv")
    X_test, y_test = load_xy(split_dir / "test.csv")

    total = len(LAMBDAS)
    rows = []
    start_all = time.time()

    print("=" * 80)
    print(f"Dataset: {args.name}")
    print(f"Outer: outer_{args.outer}")
    print(f"Method: {args.method}")
    print(f"Depth: {args.depth}")
    print(f"CV folds: {args.cv_folds}")
    print(f"n_thresholds: {args.n_thresholds}")
    print(f"threshold directory label: {threshold_dir_label(args)}")
    print(f"thresholds_strategy: {args.thresholds_strategy}")
    print(f"min_leaf_node_size: {args.min_leaf_node_size}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Total grid jobs: {total}")
    print("=" * 80)
    sys.stdout.flush()

    for job_id, lambda_ in enumerate(LAMBDAS, start=1):
        step_start = time.time()
        print(f"[{job_id}/{total}] lambda={lambda_}", flush=True)

        val_r2, val_mse = cv_score(model_cls, args, lambda_, X_train, y_train)
        final = fit_eval(model_cls, args, lambda_, X_train, y_train, X_test, y_test)
        rows.append(
            {
                "dataset": args.name,
                "outer": f"outer_{args.outer}",
                "method": args.method,
                "depth": args.depth,
                "lambda": lambda_,
                "cv_folds": args.cv_folds,
                "n_thresholds": args.n_thresholds,
                "threshold_label": threshold_dir_label(args),
                "thresholds_strategy": args.thresholds_strategy,
                "min_leaf_node_size": args.min_leaf_node_size,
                "val_r2": val_r2,
                "val_mse": val_mse,
                **final,
            }
        )

        elapsed = time.time() - start_all
        eta = elapsed / job_id * (total - job_id)
        print(
            f"    -> val_r2={val_r2:.6f}, test_r2={final['test_r2']:.6f}, "
            f"leaves={final['n_leaves']}, time={time.time() - step_start:.1f}s, "
            f"elapsed={elapsed / 60:.1f}min, ETA={eta / 60:.1f}min",
            flush=True,
        )

    out_csv = output_csv(args)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_csv, index=False)
    print("=" * 80)
    print(f"Saved to {out_csv}")
    print(f"Total time: {(time.time() - start_all) / 60:.2f} min")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Disabled for constant runs; threshold label is fixed to 20.",
    )
    parser.add_argument("--thresholds_strategy", default="quantile")
    parser.add_argument("--min_leaf_node_size", type=int, default=1)
    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2")
    if args.n_thresholds <= 0:
        raise ValueError("--n_thresholds must be positive")
    if args.n_thresholds != 20:
        raise ValueError("constant experiments are fixed to --n_thresholds 20")
    if args.threshold_label:
        raise ValueError("constant experiments are fixed to threshold label 20")
    if args.min_leaf_node_size < 1:
        raise ValueError("--min_leaf_node_size must be at least 1 for constant-leaf trees")

    run_outer(args)


if __name__ == "__main__":
    main()

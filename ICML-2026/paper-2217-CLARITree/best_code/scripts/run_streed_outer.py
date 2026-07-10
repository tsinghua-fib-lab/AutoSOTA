import time
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pystreed import STreeDPiecewiseLinearRegressor


COST_COMPLEXITIES = [0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0005, 0.0001]
RIDGE_PENALTIES = [0.1, 0.01, 0.001, 0.0001, 0.00001]


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def r2(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def load_xy(path):
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)


def build_model(args, cost_complexity, ridge_penalty, lasso_penalty):
    return STreeDPiecewiseLinearRegressor(
        simple=(args.method == "streed_s"),
        max_depth=args.depth,
        max_num_nodes=None,
        verbose=False,
        cost_complexity=cost_complexity,
        lasso_penalty=lasso_penalty,
        ridge_penalty=ridge_penalty,
        n_thresholds=args.n_thresholds,
    )


def fit_eval(model, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    try:
        n_leaves = int(model.get_n_leaves())
    except Exception:
        n_leaves = np.nan

    return {
        "train_time_s": float(train_time),
        "train_mse": mse(y_train, pred_train),
        "train_r2": r2(y_train, pred_train),
        "test_mse": mse(y_test, pred_test),
        "test_r2": r2(y_test, pred_test),
        "n_leaves": n_leaves,
    }


def cv_score(args, X, y, cost_complexity, ridge_penalty, lasso_penalty):
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    val_r2s = []
    val_mses = []

    for tr_idx, val_idx in kf.split(X):
        model = build_model(args, cost_complexity, ridge_penalty, lasso_penalty)
        res = fit_eval(model, X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
        val_r2s.append(res["test_r2"])
        val_mses.append(res["test_mse"])

    return float(np.mean(val_r2s)), float(np.mean(val_mses))


def threshold_dir_label(args):
    return args.threshold_label if args.threshold_label else str(args.n_thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--method", choices=["streed", "streed_s"], required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label, e.g. full; model still uses --n_thresholds.",
    )
    args = parser.parse_args()

    if args.threshold_label and ("/" in args.threshold_label or "\\" in args.threshold_label):
        raise ValueError("--threshold_label must not contain path separators")

    lasso_grid = [0.1, 0.001] if args.method == "streed" else [0.0]

    split_dir = Path("data") / args.name / "splits" / f"outer_{args.outer}"
    X_train, y_train = load_xy(split_dir / "train.csv")
    X_test, y_test = load_xy(split_dir / "test.csv")

    total_jobs = len(COST_COMPLEXITIES) * len(RIDGE_PENALTIES) * len(lasso_grid)
    job_id = 0
    start_all = time.time()
    rows = []

    print("=" * 80)
    print(f"Dataset: {args.name}")
    print(f"Outer: outer_{args.outer}")
    print(f"Method: {args.method}")
    print(f"Depth: {args.depth}")
    print(f"CV folds: {args.cv_folds}")
    print(f"n_thresholds: {args.n_thresholds}")
    print(f"threshold directory label: {threshold_dir_label(args)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Total grid jobs: {total_jobs}")
    print("=" * 80)
    sys.stdout.flush()

    for cost_complexity in COST_COMPLEXITIES:
        for ridge_penalty in RIDGE_PENALTIES:
            for lasso_penalty in lasso_grid:
                job_id += 1
                t0 = time.time()

                print(
                    f"[{job_id}/{total_jobs}] "
                    f"{args.method} outer_{args.outer} | "
                    f"cc={cost_complexity}, ridge={ridge_penalty}, lasso={lasso_penalty}"
                )
                sys.stdout.flush()

                val_r2, val_mse = cv_score(
                    args,
                    X_train,
                    y_train,
                    cost_complexity,
                    ridge_penalty,
                    lasso_penalty,
                )

                model = build_model(args, cost_complexity, ridge_penalty, lasso_penalty)
                final_res = fit_eval(model, X_train, y_train, X_test, y_test)

                rows.append({
                    "dataset": args.name,
                    "outer": f"outer_{args.outer}",
                    "method": args.method,
                    "depth": args.depth,
                    "n_thresholds": args.n_thresholds,
                    "cv_folds": args.cv_folds,
                    "cost_complexity": cost_complexity,
                    "ridge_penalty": ridge_penalty,
                    "lasso_penalty": lasso_penalty,
                    "val_r2": val_r2,
                    "val_mse": val_mse,
                    **final_res,
                })

                dt = time.time() - t0
                elapsed = time.time() - start_all
                avg_time = elapsed / job_id
                eta = avg_time * (total_jobs - job_id)

                print(
                    f"    -> val_r2={val_r2:.6f}, "
                    f"test_r2={final_res['test_r2']:.6f}, "
                    f"leaves={final_res['n_leaves']}, "
                    f"time={dt:.1f}s, "
                    f"elapsed={elapsed/60:.1f}min, "
                    f"ETA={eta/60:.1f}min"
                )
                sys.stdout.flush()

    out_dir = (
        Path("results") / "baseline" / args.method
        / f"linear_regression_tree_depth{args.depth}_threshold_{threshold_dir_label(args)}"
        / args.name / f"outer{args.outer}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_filename = Path(args.name).name
    out_csv = out_dir / f"{dataset_filename}_outer{args.outer}_d{args.depth}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("=" * 80)
    print(f"Finished {args.method} {args.name} outer_{args.outer}")
    print(f"Saved to {out_csv}")
    print(f"Total time: {(time.time() - start_all) / 60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from clari_tree import CLARITree, Greedy


LAMBDAS = [0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0005, 0.0001]
KAPPAS = [0.1, 0.01, 0.001, 0.0001, 0.00001]
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


def fit_eval(model_cls, args, lambda_, kappa, X_train, y_train, X_test, y_test):
    model = model_cls(
        kappa=float(kappa),
        depth=int(args.depth),
        lambda_=float(lambda_),
        n_thresholds=int(args.n_thresholds),
        thresholds_strategy=args.thresholds_strategy,
        verbose=False,
        min_leaf_node_size=int(args.min_leaf_node_size),
    )

    if hasattr(args, "refine_kappa_factor"):
        model.refine_kappa_factor = float(args.refine_kappa_factor)

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


def cv_score(model_cls, args, lambda_, kappa, X, y):
    val_r2s, val_mses = [], []
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        res = fit_eval(
            model_cls,
            args,
            lambda_,
            kappa,
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
        )
        val_r2s.append(res["test_r2"])
        val_mses.append(res["test_mse"])
    return float(np.mean(val_r2s)), float(np.mean(val_mses))


def threshold_dir_label(args):
    return args.threshold_label if args.threshold_label else str(args.n_thresholds)


def output_csv(args):
    out_dir = (
        Path("results")
        / "ours"
        / args.method
        / f"linear_regression_tree_depth{args.depth}_threshold_{threshold_dir_label(args)}"
        / args.name
        / f"outer{args.outer}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = Path(args.name).name
    return out_dir / f"{dataset_filename}_outer{args.outer}_d{args.depth}.csv"


def _evaluate_combo_worker(combo_spec):
    """Worker function for parallel grid evaluation.

    Receives a tuple of (split_dir, method_name, lambda_, kappa, depth,
    n_thresholds, thresholds_strategy, min_leaf_node_size, cv_folds,
    dataset_name, outer) and returns a result dict.

    This is a module-level function so it can be pickled for
    ProcessPoolExecutor on platforms that use 'spawn'.
    """
    (split_dir, method_name, lambda_, kappa, depth, n_thresholds,
     thresholds_strategy, min_leaf_node_size, cv_folds,
     dataset_name, outer, refine_kappa_factor) = combo_spec

    # Re-import inside worker for spawn compatibility
    import numpy as _np
    import pandas as _pd
    import time as _time
    from pathlib import Path as _Path
    from sklearn.model_selection import KFold as _KFold
    from clari_tree import CLARITree as _CLARITree, Greedy as _Greedy

    _METHODS = {"claritree": _CLARITree, "greedy": _Greedy}
    model_cls = _METHODS[method_name]

    # Local helpers
    def _mse(y_true, y_pred):
        return float(_np.mean((_np.asarray(y_true) - _np.asarray(y_pred)) ** 2))

    def _r2(y_true, y_pred):
        y_true = _np.asarray(y_true)
        ss_res = _np.sum((y_true - _np.asarray(y_pred)) ** 2)
        ss_tot = _np.sum((y_true - _np.mean(y_true)) ** 2)
        return 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    def _load_xy(csv_path):
        df = _pd.read_csv(csv_path).apply(_pd.to_numeric, errors="coerce").fillna(0.0)
        return df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)

    class _Args:
        pass

    _args = _Args()
    _args.depth = depth
    _args.n_thresholds = n_thresholds
    _args.thresholds_strategy = thresholds_strategy
    _args.min_leaf_node_size = min_leaf_node_size
    _args.cv_folds = cv_folds

    X_train, y_train = _load_xy(_Path(split_dir) / "train.csv")
    X_test, y_test = _load_xy(_Path(split_dir) / "test.csv")

    # 3-fold CV scoring
    val_r2s, val_mses = [], []
    kf = _KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        model = model_cls(
            kappa=float(kappa),
            depth=int(depth),
            lambda_=float(lambda_),
            n_thresholds=int(n_thresholds),
            thresholds_strategy=thresholds_strategy,
            verbose=False,
            min_leaf_node_size=int(min_leaf_node_size),
        )
        model.refine_kappa_factor = refine_kappa_factor
        model.fit(X_train[train_idx], y_train[train_idx])
        pred_val = model.predict(X_train[val_idx])
        val_r2s.append(_r2(y_train[val_idx], pred_val))
        val_mses.append(_mse(y_train[val_idx], pred_val))

    # Final fit on full training set
    model = model_cls(
        kappa=float(kappa),
        depth=int(depth),
        lambda_=float(lambda_),
        n_thresholds=int(n_thresholds),
        thresholds_strategy=thresholds_strategy,
        verbose=False,
        min_leaf_node_size=int(min_leaf_node_size),
    )
    model.refine_kappa_factor = refine_kappa_factor
    t0 = _time.time()
    loss = model.fit(X_train, y_train)
    train_time = _time.time() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "dataset": dataset_name,
        "outer": f"outer_{outer}",
        "method": method_name,
        "depth": depth,
        "lambda": lambda_,
        "kappa": kappa,
        "cv_folds": cv_folds,
        "n_thresholds": n_thresholds,
        "thresholds_strategy": thresholds_strategy,
        "min_leaf_node_size": min_leaf_node_size,
        "val_r2": float(_np.mean(val_r2s)),
        "val_mse": float(_np.mean(val_mses)),
        "loss": float(loss),
        "train_time_s": float(train_time),
        "train_mse": _mse(y_train, pred_train),
        "train_r2": _r2(y_train, pred_train),
        "test_mse": _mse(y_test, pred_test),
        "test_r2": _r2(y_test, pred_test),
        "n_leaves": int(model.n_leaves()),
    }


def _run_outer_parallel(args, split_dir, X_train, y_train, X_test, y_test):
    """Execute grid evaluation in parallel using ProcessPoolExecutor."""
    total = len(LAMBDAS) * len(KAPPAS)
    workers = min(args.workers, total)

    # Build combo specs for all (lambda, kappa) pairs
    combos = []
    for lambda_ in LAMBDAS:
        for kappa in KAPPAS:
            combos.append((
                str(split_dir), args.method, lambda_, kappa,
                args.depth, args.n_thresholds, args.thresholds_strategy,
                args.min_leaf_node_size, args.cv_folds,
                args.name, args.outer, args.refine_kappa_factor,
            ))

    print(f"Dispatching {total} grid jobs across {workers} workers ...", flush=True)

    rows = []
    start_all = time.time()
    completed = 0

    # Set OMP_NUM_THREADS=1 in worker subprocesses to avoid oversubscription
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_combo = {
            executor.submit(_evaluate_combo_worker, combo): combo
            for combo in combos
        }

        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            _, _, lambda_, kappa, _, _, _, _, _, _, _, _ = combo
            try:
                result = future.result()
                rows.append(result)
                completed += 1
                elapsed = time.time() - start_all
                eta = elapsed / completed * (total - completed) if completed > 0 else 0
                print(
                    f"[{completed}/{total}] lambda={lambda_}, kappa={kappa} "
                    f"-> val_r2={result['val_r2']:.6f}, test_r2={result['test_r2']:.6f}, "
                    f"leaves={result['n_leaves']}, "
                    f"elapsed={elapsed / 60:.1f}min, ETA={eta / 60:.1f}min",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{completed}/{total}] lambda={lambda_}, kappa={kappa} -> FAILED: {exc}", flush=True)
                completed += 1

    # ALGO-05: Two-stage grid refinement around best (lambda, kappa)
    if hasattr(args, "refine_grid") and args.refine_grid:
        # Find best (lambda, kappa) by val_r2 from stage 1
        best = max(rows, key=lambda r: r["val_r2"])
        best_lambda = best["lambda"]
        best_kappa = best["kappa"]
        # Create geometrically refined candidates: lambda * [1/1.3, 1.0, 1.3]
        refined_lambdas = sorted(set([best_lambda / 1.3, best_lambda, best_lambda * 1.3]))
        refined_kappas = sorted(set([best_kappa / 1.3, best_kappa, best_kappa * 1.3]))
        refined_combos = []
        for rl in refined_lambdas:
            for rk in refined_kappas:
                if abs(rl - best_lambda) < 1e-15 and abs(rk - best_kappa) < 1e-15:
                    continue  # already evaluated
                refined_combos.append((rl, rk))

        if refined_combos:
            print("\n[ALGO-05] Stage 2: refining %d combos around lambda=%.6f, kappa=%.6f ..." % (
                len(refined_combos), best_lambda, best_kappa), flush=True)
            refined_start = time.time()
            # Use a fresh executor for the refinement stage
            with ProcessPoolExecutor(max_workers=workers) as ref_executor:
                refined_futures = {}
                for rl, rk in refined_combos:
                    combo_spec = (
                        str(split_dir), args.method, rl, rk,
                        args.depth, args.n_thresholds, args.thresholds_strategy,
                        args.min_leaf_node_size, args.cv_folds,
                        args.name, args.outer, args.refine_kappa_factor,
                    )
                    refined_futures[ref_executor.submit(_evaluate_combo_worker, combo_spec)] = (rl, rk)

                refined_done = 0
                for future in as_completed(refined_futures):
                    rl, rk = refined_futures[future]
                    try:
                        result = future.result()
                        rows.append(result)
                        refined_done += 1
                        print("[ALGO-05] refined [%d/%d] lambda=%.6f kappa=%.6f -> val_r2=%.6f test_r2=%.6f" % (
                            refined_done, len(refined_combos), rl, rk, result["val_r2"], result["test_r2"]), flush=True)
                    except Exception as exc:
                        print("[ALGO-05] refined [%d/%d] lambda=%.6f kappa=%.6f -> FAILED: %s" % (
                            refined_done + 1, len(refined_combos), rl, rk, exc), flush=True)
                        refined_done += 1
            print("[ALGO-05] Stage 2 complete in %.1f min" % ((time.time() - refined_start) / 60), flush=True)

    # Sort rows to match original order (lambda descending, kappa descending)
    lambda_order = {lam: i for i, lam in enumerate(LAMBDAS)}
    kappa_order = {kap: i for i, kap in enumerate(KAPPAS)}
    rows.sort(key=lambda r: (lambda_order.get(r["lambda"], 99), kappa_order.get(r["kappa"], 99)))

    return rows


def run_outer(args):
    model_cls = METHODS[args.method]
    split_dir = Path("data") / args.name / "splits" / f"outer_{args.outer}"
    X_train, y_train = load_xy(split_dir / "train.csv")
    X_test, y_test = load_xy(split_dir / "test.csv")

    total = len(LAMBDAS) * len(KAPPAS)
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
    print(f"Workers: {args.workers}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Total grid jobs: {total}")
    print("=" * 80)
    sys.stdout.flush()

    if args.workers > 1:
        rows = _run_outer_parallel(args, split_dir, X_train, y_train, X_test, y_test)
    else:
        # Original sequential path (unchanged)
        rows = []
        job_id = 0
        for lambda_ in LAMBDAS:
            for kappa in KAPPAS:
                job_id += 1
                step_start = time.time()
                print(f"[{job_id}/{total}] lambda={lambda_}, kappa={kappa}", flush=True)

                val_r2, val_mse = cv_score(model_cls, args, lambda_, kappa, X_train, y_train)
                final = fit_eval(model_cls, args, lambda_, kappa, X_train, y_train, X_test, y_test)
                rows.append(
                    {
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
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for grid evaluation (default: 1 = sequential)")
    parser.add_argument("--refine_kappa_factor", type=float, default=1.0,
                        help="Factor for post-training leaf coefficient refinement (1.0 = no refinement)")
    parser.add_argument("--refine_grid", action="store_true",
                        help="ALGO-05: two-stage grid refinement around best (lambda, kappa)")
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label, e.g. full; model still uses --n_thresholds.",
    )
    parser.add_argument("--thresholds_strategy", default="quantile")
    parser.add_argument(
        "--min_leaf_node_size",
        type=int,
        default=0,
        help="0 means CLARITree/Greedy auto default",
    )
    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2")
    if args.n_thresholds <= 0:
        raise ValueError("--n_thresholds must be positive")
    if args.threshold_label and ("/" in args.threshold_label or "\\" in args.threshold_label):
        raise ValueError("--threshold_label must not contain path separators")

    run_outer(args)


if __name__ == "__main__":
    main()

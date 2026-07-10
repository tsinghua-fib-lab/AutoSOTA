from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_mpl_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.getuid())}-{os.environ.get('SLURM_ARRAY_TASK_ID', 'local')}"
_mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"matplotlib-{_mpl_id}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from scripts.processors.guide import GuideProcessor
from scripts.processors.guide_utils import (
    find_guide_training_out,
    parse_guide_train_r2_and_elapsed,
    parse_mse,
    parse_r2,
    run_rscript,
    update_guide_model_r,
)
from scripts.processors.m5 import M5Processor
from scripts.processors.pilot import PilotProcessor


METHODS = {"m5", "pilot", "guide"}
LAMBDAS = [0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0005, 0.0001]
GUIDE_NODES = [2, 4, 6, 8, 10, 12, 14, 16]
GUIDE_ALIASES = {
    "temperature_min": "t_min",
    "temperature_max": "t_max",
    "california_housing": "ch",
}
RESULT_COLUMNS = [
    "dataset",
    "outer",
    "method",
    "depth",
    "n_thresholds",
    "cv_folds",
    "cost_complexity",
    "pilot_grid_index",
    "pilot_target_complexity",
    "pilot_min_sample_leaf",
    "pilot_min_sample_split",
    "ridge_penalty",
    "lasso_penalty",
    "val_r2",
    "val_mse",
    "train_time_s",
    "train_mse",
    "train_r2",
    "test_mse",
    "test_r2",
    "n_leaves",
]


def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)


def mse(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def r2(y, pred):
    y = np.asarray(y)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 if ss_tot == 0 else float(1.0 - np.sum((y - pred) ** 2) / ss_tot)


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def cv_fold_count(n_samples: int, requested: int):
    if n_samples < 2:
        return 0
    return max(2, min(int(requested), int(n_samples)))


def timed(fn):
    start = time.time()
    out = fn()
    return out, time.time() - start


def processor_params(method: str, args, n_train: int, complexity: float, extra=None):
    params = {"depth": args.depth, "cost_complexity": float(complexity)}
    if method == "m5":
        params.update(
            criterion="squared_error",
            splitter="best",
            min_samples_split=2,
            min_samples_leaf=1,
            use_pruning=False,
            use_smoothing=False,
            lambda_scaled_by_tss=True,
            random_state=42,
        )
    elif method == "pilot":
        params.update(
            split_criterion="BIC",
            max_model_depth=100,
            stride=max(1, int(n_train) // int(args.n_thresholds)),
            min_sample_split=2,
            min_sample_leaf=1,
            random_state=42,
        )
    if extra:
        params.update(extra)
    return params


def fit_eval_processor(proc, params, X_train, y_train, X_test, y_test):
    artifact, train_time = timed(lambda: proc.fit(proc.build(**params), X_train, y_train))
    pred_train = proc.predict(artifact.model, X_train)
    pred_test = proc.predict(artifact.model, X_test)
    try:
        leaves = int(artifact.complexity)
    except Exception:
        leaves = np.nan
    return {
        "train_time_s": float(train_time),
        "train_mse": mse(y_train, pred_train),
        "train_r2": r2(y_train, pred_train),
        "test_mse": mse(y_test, pred_test),
        "test_r2": r2(y_test, pred_test),
        "n_leaves": leaves,
    }


def cv_score(proc, method, args, X, y, complexity, extra=None):
    n_splits = cv_fold_count(len(X), args.cv_folds)
    if n_splits < 2:
        return float("nan"), float("inf")

    val_r2, val_mse = [], []
    for train_idx, val_idx in KFold(n_splits, shuffle=True, random_state=42).split(X):
        params = processor_params(method, args, len(train_idx), complexity, extra)
        try:
            res = fit_eval_processor(proc, params, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        except Exception as exc:
            print(f"[WARN] {method} CV failed for complexity={complexity}: {exc}", file=sys.stderr, flush=True)
            return float("nan"), float("inf")
        val_r2.append(res["test_r2"])
        val_mse.append(res["test_mse"])
    return float(np.mean(val_r2)), float(np.mean(val_mse))


def pilot_grid(n_train: int):
    fixed = [64, 32, 24, 20, 16, 14, 12, 10, 8, 6, 4, 2]
    dynamic = [int(n_train)]
    value = int(n_train) // 2
    while value >= 64:
        dynamic.append(value)
        value //= 2

    targets = sorted({x for x in dynamic + fixed if 2 <= x <= int(n_train)}, reverse=True)
    pairs, seen = [], set()
    for target in targets:
        min_leaf = int(np.clip(np.ceil(int(n_train) / target), 1, max(1, int(n_train) // 2)))
        if min_leaf not in seen:
            seen.add(min_leaf)
            pairs.append((int(target), min_leaf))
    return pairs


def make_row(args, method, complexity, val_r2, val_mse, final, extra=None):
    extra = extra or {}
    return {
        "dataset": args.name,
        "outer": f"outer_{args.outer}",
        "method": method,
        "depth": args.depth,
        "n_thresholds": args.n_thresholds,
        "cv_folds": args.cv_folds,
        "cost_complexity": complexity,
        "pilot_grid_index": extra.get("pilot_grid_index", np.nan) if method == "pilot" else np.nan,
        "pilot_target_complexity": complexity if method == "pilot" else np.nan,
        "pilot_min_sample_leaf": extra.get("min_sample_leaf", np.nan) if method == "pilot" else np.nan,
        "pilot_min_sample_split": extra.get("min_sample_split", np.nan) if method == "pilot" else np.nan,
        "ridge_penalty": np.nan,
        "lasso_penalty": np.nan,
        "val_r2": val_r2,
        "val_mse": val_mse,
        **final,
    }


def run_processor_method(args, method, X_train, y_train, X_test, y_test):
    proc = M5Processor() if method == "m5" else PilotProcessor()
    candidates = [(lam, None) for lam in LAMBDAS]
    if method == "pilot":
        candidates = [
            (float(target), {"min_sample_leaf": leaf, "min_sample_split": max(2, 2 * leaf)})
            for target, leaf in pilot_grid(len(X_train))
        ]

    rows, start = [], time.time()
    for i, (complexity, extra) in enumerate(candidates, start=1):
        if method == "pilot":
            extra = {**(extra or {}), "pilot_grid_index": i - 1}
        step_start = time.time()
        print(f"[{i}/{len(candidates)}] {method} outer_{args.outer} | complexity={complexity}", flush=True)
        val_r2, val_mse = cv_score(proc, method, args, X_train, y_train, complexity, extra)
        if not np.isfinite(val_mse):
            print(f"[WARN] skip {method} complexity={complexity}: invalid CV", flush=True)
            continue
        params = processor_params(method, args, len(X_train), complexity, extra)
        final = fit_eval_processor(proc, params, X_train, y_train, X_test, y_test)
        rows.append(make_row(args, method, complexity, val_r2, val_mse, final, extra))
        log_progress(final, val_r2, time.time() - step_start, time.time() - start, i, len(candidates))
    return rows


def write_guide_csv(path: Path, X, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(X.shape[1])])
    df["target"] = y
    df.to_csv(path, index=False)


def relative_to_root(path: Path):
    try:
        return f"./{path.resolve().relative_to(PROJECT_ROOT).as_posix()}"
    except ValueError:
        return path.as_posix()


def guide_eval_r(r_path: Path, train_csv: Path, eval_csv: Path, eval_r: Path):
    text = r_path.read_text(encoding="utf-8", errors="ignore")
    for src, dst in [
        (relative_to_root(train_csv), relative_to_root(eval_csv)),
        (train_csv.resolve().as_posix(), eval_csv.resolve().as_posix()),
        (train_csv.as_posix(), eval_csv.as_posix()),
        (train_csv.name, eval_csv.name),
    ]:
        text = text.replace(src, dst)
    eval_r.write_text(text, encoding="utf-8")
    update_guide_model_r(eval_r)
    out = run_rscript(eval_r)
    return parse_r2(out), parse_mse(out)


def fit_guide(proc, args, train_csv: Path, work_dir: Path, max_nodes: int):
    work_dir.mkdir(parents=True, exist_ok=True)
    model = proc.build(csv_path=train_csv, depth=args.depth, max_nodes=int(max_nodes), work_dir=work_dir)
    artifact, fit_time = timed(lambda: proc.fit(model, None, None))

    train_r2, elapsed = float("nan"), float("nan")
    train_out = find_guide_training_out(work_dir)
    if train_out is not None:
        train_r2, elapsed = parse_guide_train_r2_and_elapsed(train_out)
    fit_time = elapsed if np.isfinite(elapsed) else (artifact.extras or {}).get("fit_time", fit_time)
    return artifact, work_dir / "guide_model.R", float(fit_time), train_r2


def guide_cv(proc, args, X, y, root: Path, max_nodes: int):
    n_splits = cv_fold_count(len(X), args.cv_folds)
    if n_splits < 2:
        return float("nan"), float("inf")

    val_r2, val_mse = [], []
    for fold, (train_idx, val_idx) in enumerate(KFold(n_splits, shuffle=True, random_state=42).split(X), start=1):
        fold_dir = root / f"n{max_nodes}" / f"cv{fold}"
        train_csv, val_csv = fold_dir / "train_cv.csv", fold_dir / "valid_cv.csv"
        write_guide_csv(train_csv, X[train_idx], y[train_idx])
        write_guide_csv(val_csv, X[val_idx], y[val_idx])
        try:
            _, r_path, _, _ = fit_guide(proc, args, train_csv, fold_dir / "work", max_nodes)
            r2_val, mse_val = guide_eval_r(r_path, train_csv, val_csv, fold_dir / "work" / "valid_eval.R")
        except Exception as exc:
            print(f"[WARN] guide CV failed for max_nodes={max_nodes} fold={fold}: {exc}", file=sys.stderr, flush=True)
            return float("nan"), float("inf")
        val_r2.append(r2_val)
        val_mse.append(mse_val)
    return float(np.mean(val_r2)), float(np.mean(val_mse))


def run_guide(args, X_train, y_train, X_test, y_test):
    proc = GuideProcessor()
    root = Path("guide_save") / GUIDE_ALIASES.get(args.name, args.name) / f"outer{args.outer}"
    train_csv, test_csv = root / "train_raw.csv", root / "test_raw.csv"
    write_guide_csv(train_csv, X_train, y_train)
    write_guide_csv(test_csv, X_test, y_test)

    rows, start = [], time.time()
    for i, max_nodes in enumerate(args.guide_nodes, start=1):
        step_start = time.time()
        print(f"[{i}/{len(args.guide_nodes)}] guide outer_{args.outer} | max_nodes={max_nodes}", flush=True)
        val_r2, val_mse = guide_cv(proc, args, X_train, y_train, root, max_nodes)
        if not np.isfinite(val_mse):
            print(f"[WARN] skip guide max_nodes={max_nodes}: invalid CV", flush=True)
            continue

        try:
            artifact, r_path, train_time, train_r2 = fit_guide(proc, args, train_csv, root / f"n{max_nodes}" / "final", max_nodes)
            test_r2, test_mse = guide_eval_r(r_path, train_csv, test_csv, r_path.with_name("test_eval.R"))
            train_eval_r2, train_mse = guide_eval_r(r_path, train_csv, train_csv, r_path.with_name("train_eval.R"))
        except Exception as exc:
            print(f"[WARN] skip guide max_nodes={max_nodes}: final fit failed: {exc}", file=sys.stderr, flush=True)
            continue

        try:
            leaves = int(artifact.complexity)
        except Exception:
            leaves = int(max_nodes)
        final = {
            "train_time_s": train_time,
            "train_mse": train_mse,
            "train_r2": train_r2 if np.isfinite(train_r2) else train_eval_r2,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "n_leaves": leaves,
        }
        rows.append(make_row(args, "guide", float(max_nodes), val_r2, val_mse, final))
        log_progress(final, val_r2, time.time() - step_start, time.time() - start, i, len(args.guide_nodes))
    return rows


def log_progress(final, val_r2, step_time, total_time, done, total):
    eta = total_time / max(1, done) * (total - done)
    print(
        f"    -> val_r2={val_r2:.6f}, test_r2={final['test_r2']:.6f}, "
        f"leaves={final['n_leaves']}, time={step_time:.1f}s, "
        f"elapsed={total_time / 60:.1f}min, ETA={eta / 60:.1f}min",
        flush=True,
    )


def threshold_dir_label(args):
    return args.threshold_label if args.threshold_label else str(args.n_thresholds)


def save_rows(args, method, rows):
    if not rows:
        raise RuntimeError(f"No rows produced for {method} {args.name} outer_{args.outer}")
    out_dir = (
        Path("results")
        / "baseline"
        / method
        / f"linear_regression_tree_depth{args.depth}_threshold_{threshold_dir_label(args)}"
        / args.name
        / f"outer{args.outer}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = Path(args.name).name
    out_csv = out_dir / f"{dataset_filename}_outer{args.outer}_d{args.depth}.csv"
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_csv, index=False)
    return out_csv


def run_method(args, method):
    split_dir = Path("data") / args.name / "splits" / f"outer_{args.outer}"
    X_train, y_train = load_xy(split_dir / "train.csv")
    X_test, y_test = load_xy(split_dir / "test.csv")
    print(f"{args.name} outer_{args.outer} | {method} | train={X_train.shape} test={X_test.shape}", flush=True)

    rows = (
        run_guide(args, X_train, y_train, X_test, y_test)
        if method == "guide"
        else run_processor_method(args, method, X_train, y_train, X_test, y_test)
    )
    print(f"Saved to {save_rows(args, method, rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--method", default="m5,pilot,guide")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label, e.g. full; model still uses --n_thresholds.",
    )
    parser.add_argument("--guide_nodes", default=",".join(map(str, GUIDE_NODES)))
    args = parser.parse_args()

    if args.threshold_label and ("/" in args.threshold_label or "\\" in args.threshold_label):
        raise ValueError("--threshold_label must not contain path separators")

    args.guide_nodes = parse_list(args.guide_nodes, int)
    methods = parse_list(args.method, str)
    invalid = sorted(set(methods) - METHODS)
    if invalid:
        raise ValueError(f"Unknown method(s): {invalid}. Valid methods: {sorted(METHODS)}")

    for method in methods:
        run_method(args, method)


if __name__ == "__main__":
    main()

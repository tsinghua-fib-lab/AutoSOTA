# ./claritree-env/bin/python scripts/constant/merge_constant.py \
#   --depth 5 \
#   --n_thresholds 20 \
#   --require_complete

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASELINE_METHODS = {"cart", "guide", "streed"}
OURS_METHODS = {"claritree", "greedy"}
METHODS = ["claritree", "streed", "guide", "greedy", "cart"]
DATASETS = [
    "airfoil",
    "auction",
    "auto_mpg",
    "energe_c",
    "energe_h",
    "insurance",
    "optical_net",
    "real_estate",
    "servo",
    "synch",
    "yacht",
    "california_housing",
    "seoul_bike",
    "temperature_max",
    "temperature_min",
    "walmart",
]
OPENML_DATA_ROOT = Path("data/openml")
METRIC_COLS = [
    "val_r2",
    "val_mse",
    "train_r2",
    "test_r2",
    "train_mse",
    "test_mse",
    "loss",
    "train_time_s",
    "n_leaves",
]


def discover_openml_datasets() -> list[str]:
    if not OPENML_DATA_ROOT.exists():
        return []
    return [f"openml/{path.name}" for path in sorted(OPENML_DATA_ROOT.iterdir()) if path.is_dir()]


def dataset_file_stem(dataset: str) -> str:
    return Path(dataset).name


def method_root(method: str) -> Path:
    root = "baseline" if method in BASELINE_METHODS else "ours"
    return Path("results") / root / method


def experiment_root(method: str, depth: int, n_thresholds: int) -> Path:
    return method_root(method) / f"constant_regression_tree_depth{depth}_threshold_{n_thresholds}"


def experiment_dir(method: str, dataset: str, depth: int, n_thresholds: int) -> Path:
    return experiment_root(method, depth, n_thresholds) / dataset


def should_collapse_n_thresholds(df: pd.DataFrame) -> bool:
    if "n_thresholds" not in df.columns or "threshold_label" not in df.columns:
        return False
    labels = df["threshold_label"].dropna().astype(str).unique()
    return len(labels) == 1 and labels[0] == "full" and df["n_thresholds"].nunique(dropna=True) > 1


def group_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(METRIC_COLS) | {"outer"}
    if should_collapse_n_thresholds(df):
        excluded.add("n_thresholds")
    return [col for col in df.columns if col not in excluded]


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in METRIC_COLS if col in df.columns]


def read_outer_rows(
    method: str,
    dataset: str,
    depth: int,
    n_thresholds: int,
    require_complete: bool,
) -> pd.DataFrame | None:
    base_dir = experiment_dir(method, dataset, depth, n_thresholds)
    file_stem = dataset_file_stem(dataset)
    dfs = []
    missing = []
    for outer in range(5):
        nested_path = base_dir / f"outer{outer}" / f"{file_stem}_outer{outer}_d{depth}.csv"
        direct_path = base_dir / f"{file_stem}_outer{outer}_d{depth}.csv"
        path = next((candidate for candidate in [nested_path, direct_path] if candidate.exists()), None)
        if path is None:
            missing.append(nested_path)
        else:
            dfs.append(pd.read_csv(path))

    if missing:
        for path in missing:
            print(f"[WARN] Missing: {path}")
        if require_complete:
            print(f"[SKIP] Incomplete: {method} {dataset}")
            return None
    if not dfs:
        print(f"[SKIP] No outer files: {method} {dataset}")
        return None
    return pd.concat(dfs, ignore_index=True)


def summarize_outer_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = group_columns(df)
    metric_cols = metric_columns(df)
    if not group_cols or not metric_cols:
        raise ValueError("Could not determine group/metric columns")

    mean_df = df.groupby(group_cols, as_index=False, dropna=False)[metric_cols].mean()
    mean_df["outer"] = "mean"

    std_df = df.groupby(group_cols, as_index=False, dropna=False)[metric_cols].std()
    std_df["outer"] = "std"

    if "val_r2" not in mean_df.columns:
        raise ValueError("val_r2 is required to choose best row")
    best_df = (
        mean_df.sort_values("val_r2", ascending=False)
        .groupby(["dataset", "method"], as_index=False, dropna=False)
        .head(1)
        .copy()
    )
    best_df["outer"] = "best_by_mean_val_r2"
    return mean_df, std_df, best_df


def merge_one(
    method: str,
    dataset: str,
    depth: int,
    n_thresholds: int,
    require_complete: bool,
    write: bool,
) -> pd.DataFrame | None:
    df = read_outer_rows(method, dataset, depth, n_thresholds, require_complete)
    if df is None:
        return None

    mean_df, std_df, best_df = summarize_outer_rows(df)
    final_df = pd.concat([df, mean_df, std_df, best_df], ignore_index=True, sort=False)

    if write:
        file_stem = dataset_file_stem(dataset)
        out_csv = experiment_dir(method, dataset, depth, n_thresholds) / f"{file_stem}_outer0-4_d{depth}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_csv, index=False)
        best = best_df.iloc[0]
        print(
            f"{method} {dataset}: mean_val_r2={best['val_r2']:.6f}, "
            f"mean_test_r2={best['test_r2']:.6f}, mean_leaves={best['n_leaves']:.2f}"
        )
        print(f"Saved to {out_csv}")
    return final_df


def final_output_path(openml: bool, n_thresholds: int) -> Path:
    base = "final_constant_openml" if openml else "final_constant"
    if n_thresholds == 20:
        return Path("results") / f"{base}.csv"
    return Path("results") / f"{base}_{n_thresholds}.csv"


def merge_all(
    depth: int,
    n_thresholds: int,
    require_complete: bool,
    openml: bool,
    method_filter: list[str] | None,
    dataset_filter: list[str] | None,
) -> pd.DataFrame:
    datasets = discover_openml_datasets() if openml else DATASETS
    if dataset_filter is not None:
        datasets = dataset_filter
    methods = method_filter if method_filter is not None else METHODS

    parts = []
    for method in methods:
        for dataset in datasets:
            merged = merge_one(
                method,
                dataset,
                depth=depth,
                n_thresholds=n_thresholds,
                require_complete=require_complete,
                write=False,
            )
            if merged is None:
                continue
            raw = merged[~merged["outer"].isin(["mean", "std", "best_by_mean_val_r2"])].copy()
            mean_df, std_df, best_df = summarize_outer_rows(raw)
            parts.extend([mean_df, std_df, best_df])

    if not parts:
        raise RuntimeError("No merged rows found")
    final_df = pd.concat(parts, ignore_index=True, sort=False)
    final_df["threshold_label"] = str(n_thresholds)
    out_csv = final_output_path(openml, n_thresholds)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv} ({len(final_df)} rows)")
    return final_df


def parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--openml", action="store_true")
    parser.add_argument("--method", default=None, help="Comma-separated subset of methods.")
    parser.add_argument("--name", default=None, help="Comma-separated subset of datasets.")
    parser.add_argument("--write_per_dataset", action="store_true")
    args = parser.parse_args()

    if args.n_thresholds != 20:
        parser.error("constant experiments are fixed to --n_thresholds 20")

    method_filter = parse_csv_arg(args.method)
    if method_filter is not None:
        invalid = sorted(set(method_filter) - set(METHODS))
        if invalid:
            parser.error(f"Unknown method(s): {invalid}. Valid methods: {METHODS}")
    dataset_filter = parse_csv_arg(args.name)

    if args.write_per_dataset:
        datasets = dataset_filter if dataset_filter is not None else (discover_openml_datasets() if args.openml else DATASETS)
        methods = method_filter if method_filter is not None else METHODS
        for method in methods:
            for dataset in datasets:
                merge_one(method, dataset, args.depth, args.n_thresholds, args.require_complete, write=True)
    else:
        merge_all(
            depth=args.depth,
            n_thresholds=args.n_thresholds,
            require_complete=args.require_complete,
            openml=args.openml,
            method_filter=method_filter,
            dataset_filter=dataset_filter,
        )


if __name__ == "__main__":
    main()

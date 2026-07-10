# example: python scripts/merge_single_method_outer.py --method claritree
# large full example: python scripts/merge_single_method_outer.py --method streed --large_full
# openml example: python scripts/merge_single_method_outer.py --method claritree --openml

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


BASELINE_METHODS = {"streed", "streed_s", "m5", "pilot", "guide"}
OURS_METHODS = {"claritree", "greedy"}
METHODS = sorted(BASELINE_METHODS | OURS_METHODS)
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
LARGE_FULL_DIRNAME = "large_full"
LARGE_FULL_DATASETS = [
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
PILOT_GROUP_COL = "pilot_grid_index"
PILOT_PARAM_COLS = ["pilot_min_sample_leaf", "pilot_min_sample_split"]
PILOT_REPRESENTATIVE_COLS = ["cost_complexity", "pilot_target_complexity", *PILOT_PARAM_COLS]
SUMMARY_COLS = {"n_outer_merged"}


def discover_openml_datasets() -> list[str]:
    if not OPENML_DATA_ROOT.exists():
        return []
    return [f"openml/{path.name}" for path in sorted(OPENML_DATA_ROOT.iterdir()) if path.is_dir()]


def normalize_dataset_name(dataset: str, openml: bool) -> str:
    if openml and not dataset.startswith("openml/"):
        return f"openml/{dataset}"
    return dataset


def dataset_file_stem(dataset: str) -> str:
    return Path(dataset).name


def data_dir(dataset: str) -> Path:
    return Path("data") / dataset


def outer_id(value: object) -> int | None:
    text = str(value)
    if text.startswith("outer_"):
        text = text.removeprefix("outer_")
    elif text.startswith("outer"):
        text = text.removeprefix("outer")
    try:
        return int(text)
    except ValueError:
        return None


def count_csv_data_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        n_lines = sum(1 for _ in handle)
    return max(0, n_lines - 1)


def pilot_min_leaf(n_train: int, target: float) -> int | None:
    if not pd.notna(target) or target <= 0 or n_train <= 0:
        return None
    return int(max(1, min(max(1, n_train // 2), math.ceil(n_train / float(target)))))


def add_pilot_grouping_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty or "cost_complexity" not in df.columns or "outer" not in df.columns:
        return df

    train_sizes: dict[int, int] = {}
    split_root = data_dir(dataset) / "splits"
    for outer in sorted({outer_id(value) for value in df["outer"].dropna().unique()}):
        if outer is None:
            continue
        n_train = count_csv_data_rows(split_root / f"outer_{outer}" / "train.csv")
        if n_train is not None:
            train_sizes[outer] = n_train

    if not train_sizes:
        return df

    df = df.copy()
    if "pilot_target_complexity" not in df.columns:
        df["pilot_target_complexity"] = df["cost_complexity"]
    if PILOT_GROUP_COL not in df.columns:
        df[PILOT_GROUP_COL] = df.groupby("outer", sort=False).cumcount()
    leaves = []
    splits = []
    for _, row in df.iterrows():
        outer = outer_id(row.get("outer"))
        n_train = train_sizes.get(outer)
        if all(col in df.columns for col in PILOT_PARAM_COLS) and pd.notna(row.get("pilot_min_sample_leaf")):
            leaf = int(row.get("pilot_min_sample_leaf"))
        else:
            leaf = pilot_min_leaf(n_train, row.get("pilot_target_complexity")) if n_train is not None else None
        leaves.append(leaf)
        splits.append(max(2, 2 * leaf) if leaf is not None else None)
    df["pilot_min_sample_leaf"] = leaves
    df["pilot_min_sample_split"] = splits
    return df


def method_root(method: str) -> Path:
    root = "baseline" if method in BASELINE_METHODS else "ours"
    return Path("results") / root / method


def threshold_dir_label(n_thresholds: int, threshold_label: str | None = None) -> str:
    return threshold_label if threshold_label else str(n_thresholds)


def experiment_root(
    method: str,
    depth: int,
    n_thresholds: int,
    threshold_label: str | None = None,
) -> Path:
    return method_root(method) / (
        f"linear_regression_tree_depth{depth}_threshold_{threshold_dir_label(n_thresholds, threshold_label)}"
    )


def experiment_dir(
    method: str,
    dataset: str,
    depth: int,
    n_thresholds: int,
    large_full: bool = False,
    threshold_label: str | None = None,
) -> Path:
    root = experiment_root(method, depth, n_thresholds, threshold_label)
    if large_full:
        return root / LARGE_FULL_DIRNAME / dataset
    return root / dataset


def discover_large_full_datasets(
    method: str,
    depth: int,
    n_thresholds: int,
    threshold_label: str | None = None,
) -> list[str]:
    root = experiment_root(method, depth, n_thresholds, threshold_label) / LARGE_FULL_DIRNAME
    if not root.exists():
        return []
    return [path.name for path in sorted(root.iterdir()) if path.is_dir()]


def should_collapse_n_thresholds(df: pd.DataFrame) -> bool:
    if "n_thresholds" not in df.columns or "threshold_label" not in df.columns:
        return False
    labels = df["threshold_label"].dropna().astype(str).unique()
    return len(labels) == 1 and labels[0] == "full" and df["n_thresholds"].nunique(dropna=True) > 1


def should_collapse_pilot_complexity(df: pd.DataFrame) -> bool:
    if "method" not in df.columns or "cost_complexity" not in df.columns:
        return False
    if PILOT_GROUP_COL not in df.columns:
        return False
    methods = df["method"].dropna().astype(str).unique()
    return len(methods) == 1 and methods[0] == "pilot"


def group_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(METRIC_COLS) | {"outer"} | SUMMARY_COLS
    if should_collapse_n_thresholds(df):
        excluded.add("n_thresholds")
    if should_collapse_pilot_complexity(df):
        excluded.update(PILOT_REPRESENTATIVE_COLS)
    return [col for col in df.columns if col not in excluded]


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in METRIC_COLS if col in df.columns]


def read_outer_rows(
    method: str,
    dataset: str,
    depth: int = 4,
    n_thresholds: int = 20,
    require_complete: bool = False,
    large_full: bool = False,
    threshold_label: str | None = None,
) -> pd.DataFrame | None:
    base_dir = experiment_dir(
        method,
        dataset,
        depth,
        n_thresholds,
        large_full=large_full,
        threshold_label=threshold_label,
    )
    file_stem = dataset_file_stem(dataset)
    dfs = []
    missing = []
    for outer in range(5):
        direct_path = base_dir / f"{file_stem}_outer{outer}_d{depth}.csv"
        nested_path = base_dir / f"outer{outer}" / f"{file_stem}_outer{outer}_d{depth}.csv"
        candidates = [direct_path] if large_full else [nested_path, direct_path]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is not None:
            dfs.append(pd.read_csv(path))
        else:
            missing.append(candidates[0])

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

    outer_counts = (
        df.groupby(group_cols, as_index=False, dropna=False)["outer"]
        .nunique()
        .rename(columns={"outer": "n_outer_merged"})
    )
    mean_df = mean_df.merge(outer_counts, on=group_cols, how="left")
    std_df = std_df.merge(outer_counts, on=group_cols, how="left")

    representative_cols = [
        col
        for col in PILOT_REPRESENTATIVE_COLS
        if col in df.columns and col not in group_cols and should_collapse_pilot_complexity(df)
    ]
    if representative_cols:
        representative_df = df.groupby(group_cols, as_index=False, dropna=False)[representative_cols].mean()
        mean_df = mean_df.merge(representative_df, on=group_cols, how="left")
        std_df = std_df.merge(representative_df, on=group_cols, how="left")

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
    depth: int = 4,
    n_thresholds: int = 20,
    require_complete: bool = False,
    write: bool = True,
    openml: bool = False,
    large_full: bool = False,
    threshold_label: str | None = None,
) -> pd.DataFrame | None:
    dataset = normalize_dataset_name(dataset, openml)
    df = read_outer_rows(
        method,
        dataset,
        depth,
        n_thresholds,
        require_complete,
        large_full=large_full,
        threshold_label=threshold_label,
    )
    if df is None:
        return None
    if threshold_label is not None:
        df["threshold_label"] = threshold_label
    if method == "pilot":
        df = add_pilot_grouping_columns(df, dataset)

    mean_df, std_df, best_df = summarize_outer_rows(df)
    final_df = pd.concat([df, mean_df, std_df, best_df], ignore_index=True, sort=False)

    if write:
        file_stem = dataset_file_stem(dataset)
        suffix = "_openml" if openml else ""
        out_csv = experiment_dir(
            method,
            dataset,
            depth,
            n_thresholds,
            large_full=large_full,
            threshold_label=threshold_label,
        ) / f"{file_stem}_outer0-4_d{depth}{suffix}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_csv, index=False)

        best = best_df.iloc[0]
        print(
            f"{method}{' large_full' if large_full else ''} {dataset}: "
            f"mean_val_r2={best['val_r2']:.6f}, "
            f"mean_test_r2={best['test_r2']:.6f}, "
            f"mean_leaves={best['n_leaves']:.2f}"
        )
        print(f"Saved to {out_csv}")
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--name", default=None, help="Dataset name. Omit to merge all default datasets.")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label, e.g. full.",
    )
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--openml", action="store_true", help="Merge datasets under data/openml.")
    parser.add_argument(
        "--large_full",
        action="store_true",
        help="Merge 64h large_full StreeD files under the large_full subdirectory.",
    )
    args = parser.parse_args()

    if args.large_full and args.openml:
        parser.error("--large_full cannot be combined with --openml")
    if args.large_full and args.method != "streed":
        parser.error("--large_full is currently only supported for --method streed")
    if args.threshold_label and ("/" in args.threshold_label or "\\" in args.threshold_label):
        parser.error("--threshold_label must not contain path separators")

    if args.name:
        datasets = [normalize_dataset_name(args.name, args.openml)]
    elif args.large_full:
        datasets = discover_large_full_datasets(
            args.method,
            args.depth,
            args.n_thresholds,
            args.threshold_label,
        )
        if not datasets:
            datasets = LARGE_FULL_DATASETS
    else:
        datasets = discover_openml_datasets() if args.openml else DATASETS

    if args.openml and not datasets:
        raise RuntimeError("No OpenML datasets found under data/openml")
    if args.large_full and not datasets:
        raise RuntimeError("No large_full datasets found")

    for dataset in datasets:
        merge_one(
            args.method,
            dataset,
            depth=args.depth,
            n_thresholds=args.n_thresholds,
            require_complete=args.require_complete,
            openml=args.openml,
            large_full=args.large_full,
            threshold_label=args.threshold_label,
        )


if __name__ == "__main__":
    main()

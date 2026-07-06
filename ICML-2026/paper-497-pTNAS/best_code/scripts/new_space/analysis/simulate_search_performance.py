#!/usr/bin/env python3
"""Offline pTNAS search simulator for the added search spaces.

The simulator does not read old search-run CSVs and does not retrain models.
It reconstructs the search pipeline from compact artifacts:

  1. randomly sample M architectures from the candidate pool;
  2. rank sampled architectures by the saved pTProxy score;
  3. keep topK = ceil(M / 30);
  4. simulate SH refinement with saved validation metrics;
  5. report the selected architecture's saved test metric.

For each dataset, M grows as 5, 10, 20, 40, ... until the pool maximum.
Sampling is nested for each seed: larger M contains all architectures from
smaller M, which matches the interpretation of an increasing search budget.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASETS = [
    "avito-ad-ctr",
    "event-user-attendance",
    "avito-user-clicks",
    "hm-user-churn",
]

REGRESSION_DATASETS = {"avito-ad-ctr", "event-user-attendance"}

RESDNN_TRAIN_CSV = ROOT / "datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv"
RESDNN_PROXY_DIR = ROOT / "datasets/nas_bench_tabular/space_resdnn/proxy_score/ptproxy"
RESDNN_REGR_SPACE = ROOT / "datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_regression.txt"
RESDNN_CLS_SPACE = ROOT / "datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_classification.txt"

BLOCKMIXED_TRAIN_CSV = ROOT / "datasets/nas_bench_tabular/space_blockmixed/training/block_mixed_diverse_results.csv"
BLOCKMIXED_PROXY_DIR = ROOT / "datasets/nas_bench_tabular/space_blockmixed/proxy_score/ptproxy"

# Historical pTProxy v1 score direction on BlockMixed. Positive means larger
# proxy score is better; negative means smaller proxy score is better.
BLOCKMIXED_PROXY_DIRECTION = {
    "event-user-attendance": "negative",
    "avito-ad-ctr": "negative",
    "hm-user-churn": "positive",
    "avito-user-clicks": "negative",
}


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_datasets(value: str) -> list[str]:
    if not value.strip():
        return DEFAULT_DATASETS
    return [x.strip() for x in value.split(",") if x.strip()]


def read_space(path: Path) -> set[str]:
    with path.open() as f:
        return {line.strip() for line in f if line.strip()}


def is_regression_metric(metric: str) -> bool:
    return metric.lower() in {"mae", "mean_absolute_error"}


def m_schedule(pool_size: int, start: int = 5) -> list[int]:
    values = []
    m = start
    while m <= pool_size:
        values.append(m)
        m *= 2
    if values[-1] != pool_size:
        values.append(pool_size)
    return values


def nested_sample_order(n: int, seed: int) -> list[int]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def simulate_sh(candidates: pd.DataFrame, regression: bool, eta: int) -> pd.Series:
    """Simulate SH selection with saved validation metrics.

    Since this is an offline simulator, each SH round uses the saved
    full-training validation metric instead of retraining for early epochs.
    The final winner is equivalent to repeatedly keeping the best ceil(n/eta).
    """
    current = candidates.copy()
    while len(current) > 1:
        current = current.sort_values("best_val_metric", ascending=regression)
        keep = max(1, math.ceil(len(current) / eta))
        current = current.head(keep)
    return current.iloc[0]


def load_resdnn_dataset(dataset: str) -> tuple[pd.DataFrame, bool, bool]:
    train_df = pd.read_csv(RESDNN_TRAIN_CSV)
    train_df = train_df[train_df["dataset"] == dataset].copy()

    proxy_df = pd.read_csv(RESDNN_PROXY_DIR / f"score_{dataset}_v1.csv")
    space_file = RESDNN_REGR_SPACE if dataset in REGRESSION_DATASETS else RESDNN_CLS_SPACE
    space = read_space(space_file)

    df = proxy_df.merge(
        train_df[
            [
                "dataset",
                "architecture",
                "best_val_metric",
                "test_metric",
                "train_time_seconds",
                "test_time_seconds",
                "metric",
            ]
        ],
        on=["dataset", "architecture"],
        how="inner",
    )
    df = df[df["architecture"].isin(space)].copy()
    df["arch_key"] = df["architecture"]
    df = df.sort_values("arch_key").reset_index(drop=True)

    metric = str(df["metric"].iloc[0])
    regression = is_regression_metric(metric)
    proxy_higher_is_better = True
    return df, regression, proxy_higher_is_better


def load_blockmixed_dataset(dataset: str) -> tuple[pd.DataFrame, bool, bool]:
    train_df = pd.read_csv(BLOCKMIXED_TRAIN_CSV)
    train_df = train_df[train_df["dataset"] == dataset].copy()

    proxy_df = pd.read_csv(BLOCKMIXED_PROXY_DIR / f"score_{dataset}_v1.csv")
    df = proxy_df.merge(
        train_df[
            [
                "dataset",
                "block_specs",
                "best_val_metric",
                "test_metric",
                "train_time_seconds",
                "test_time_seconds",
                "metric",
            ]
        ],
        on=["dataset", "block_specs"],
        how="inner",
    )
    df["arch_key"] = df["block_specs"]
    df = df.sort_values("arch_key").reset_index(drop=True)

    metric = str(df["metric"].iloc[0])
    regression = is_regression_metric(metric)
    proxy_higher_is_better = BLOCKMIXED_PROXY_DIRECTION.get(dataset, "negative") == "positive"
    return df, regression, proxy_higher_is_better


def load_pool(space: str, dataset: str) -> tuple[pd.DataFrame, bool, bool]:
    if space == "ResDNN":
        return load_resdnn_dataset(dataset)
    if space == "BlockMixed":
        return load_blockmixed_dataset(dataset)
    raise ValueError(f"Unknown space: {space}")


def simulate_dataset(
    space: str,
    dataset: str,
    seed: int,
    mk_ratio: int,
    eta: int,
) -> pd.DataFrame:
    pool, regression, proxy_higher_is_better = load_pool(space, dataset)
    order = nested_sample_order(len(pool), seed)
    rows = []
    best_so_far = math.inf if regression else -math.inf

    for m in m_schedule(len(pool)):
        sampled = pool.iloc[order[:m]].copy()
        top_k = max(1, math.ceil(m / mk_ratio))
        proxy_sorted = sampled.sort_values(
            "proxy_score",
            ascending=not proxy_higher_is_better,
        )
        top_candidates = proxy_sorted.head(top_k)
        selected = simulate_sh(top_candidates, regression=regression, eta=eta)

        test_metric = float(selected["test_metric"])
        if regression:
            best_so_far = min(best_so_far, test_metric)
        else:
            best_so_far = max(best_so_far, test_metric)

        rows.append(
            {
                "space": space,
                "dataset": dataset,
                "seed": seed,
                "pool_size": len(pool),
                "M": m,
                "top_k": top_k,
                "metric": "MAE" if regression else "AUC",
                "selected_test_metric": test_metric,
                "best_so_far": best_so_far,
                "selected_val_metric": float(selected["best_val_metric"]),
                "proxy_score": float(selected["proxy_score"]),
                "proxy_time_seconds": float(sampled["proxy_time_seconds"].sum()),
                "selected_train_time_seconds": float(selected["train_time_seconds"]),
                "selected_test_time_seconds": float(selected["test_time_seconds"]),
                "best_params": selected["arch_key"],
            }
        )

    return pd.DataFrame(rows)


def simulate_all(datasets: list[str], seeds: list[int], mk_ratio: int, eta: int) -> pd.DataFrame:
    pieces = []
    for space in ["ResDNN", "BlockMixed"]:
        for dataset in datasets:
            for seed in seeds:
                pieces.append(simulate_dataset(space, dataset, seed, mk_ratio, eta))
    return pd.concat(pieces, ignore_index=True)


def summarize_for_print(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["space", "dataset", "metric", "M"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_selected_test_metric=("selected_test_metric", "mean"),
            std_selected_test_metric=("selected_test_metric", "std"),
            mean_best_so_far=("best_so_far", "mean"),
            std_best_so_far=("best_so_far", "std"),
            mean_top_k=("top_k", "mean"),
            mean_proxy_time_seconds=("proxy_time_seconds", "mean"),
            mean_selected_train_time_seconds=("selected_train_time_seconds", "mean"),
            mean_selected_test_time_seconds=("selected_test_time_seconds", "mean"),
        )
    )


def add_display_column(grouped: pd.DataFrame) -> pd.DataFrame:
    grouped = grouped.copy()
    grouped["display"] = grouped.apply(
        lambda r: f"{r['mean_best_so_far']:.4f}"
        if pd.isna(r["std_best_so_far"])
        else f"{r['mean_best_so_far']:.4f}+/-{r['std_best_so_far']:.4f}",
        axis=1,
    )
    return grouped


def print_tables(df: pd.DataFrame) -> None:
    summary = add_display_column(summarize_for_print(df))
    for space in ["ResDNN", "BlockMixed"]:
        sub = summary[summary["space"] == space].copy()
        sub["M_col"] = sub["M"].map(lambda x: f"M={x}")
        table = sub.pivot_table(
            index=["dataset", "metric"],
            columns="M_col",
            values="display",
            aggfunc="first",
        )
        ordered_cols = [f"M={m}" for m in sorted(sub["M"].unique())]
        ordered_cols = [c for c in ordered_cols if c in table.columns]
        table = table[ordered_cols].reset_index()

        print(f"\n[{space}] best-so-far test metric")
        print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42 or 42,43,44.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated datasets. Defaults to all four new-space datasets.",
    )
    parser.add_argument("--mk_ratio", type=int, default=30, help="Use topK=ceil(M/mk_ratio).")
    parser.add_argument("--eta", type=int, default=3, help="SH keep ratio denominator.")
    parser.add_argument("--output_csv", type=str, default="", help="Optional path to save detailed simulated rows.")
    parser.add_argument("--details", action="store_true", help="Print detailed selected rows.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    datasets = parse_datasets(args.datasets)
    df = simulate_all(datasets=datasets, seeds=seeds, mk_ratio=args.mk_ratio, eta=args.eta)

    print(
        f"Simulated pTNAS search: datasets={datasets}, seeds={seeds}, topK=ceil(M/{args.mk_ratio}), "
        f"eta={args.eta}, M=5,10,20,40,...,max."
    )
    print("SH refinement is simulated with saved best_val_metric; no models are retrained.")
    print_tables(df)

    if args.details:
        cols = [
            "space",
            "dataset",
            "seed",
            "M",
            "top_k",
            "metric",
            "selected_test_metric",
            "best_so_far",
            "best_params",
        ]
        print("\nDetailed selected rows")
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.output_csv:
        output_path = ROOT / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved detailed simulated rows to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

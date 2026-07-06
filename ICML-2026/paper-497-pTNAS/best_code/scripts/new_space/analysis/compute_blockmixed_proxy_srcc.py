from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FINAL_PTPROXY_VARIANT = "v1"
CLASSIC_PROXIES = [
    "SynFlow",
    "SNIP",
    "NTKCond",
    "NASWOT",
    "NTKTrace",
    "NTKTrAppx",
    "Fisher",
    "GraSP",
    "GradNorm",
]


def metric_to_perf(metric_name: str, test_metric: pd.Series) -> pd.Series:
    if metric_name == "mean_absolute_error":
        return -test_metric
    if metric_name == "roc_auc_score":
        return test_metric
    raise ValueError(f"Unsupported metric: {metric_name}")


def load_selected_groups(space_json: Path) -> dict[str, set[str]]:
    with space_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    selected: dict[str, set[str]] = {}
    for dataset, configs in payload.items():
        if not configs:
            continue
        # The response experiments use the final selected group config.
        selected[dataset] = set(configs[-1]["groups"])
    return selected


def compute_srcc(score_df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[float, int]:
    merged = score_df.merge(
        train_df[["dataset", "block_specs", "target_perf"]],
        on=["dataset", "block_specs"],
        how="inner",
    )
    merged = merged[np.isfinite(merged["proxy_score"])].copy()
    merged = merged[np.isfinite(merged["target_perf"])].copy()
    if len(merged) < 2:
        return float("nan"), len(merged)
    return float(spearmanr(merged["proxy_score"], merged["target_perf"]).statistic), len(merged)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    benchmark_root = project_root / "datasets" / "nas_bench_tabular" / "space_blockmixed"
    architecture_root = benchmark_root / "architecture"
    paper_root = project_root / "run_outputs" / "data" / "new_space"

    parser = argparse.ArgumentParser(
        description="Compute BlockMixed selected-subspace SRCC for pTProxy and classic zero-cost proxies."
    )
    parser.add_argument("--train_csv", type=Path, default=benchmark_root / "training" / "block_mixed_diverse_results.csv")
    parser.add_argument("--space_json", type=Path, default=architecture_root / "random_sampled_arch_blockmixed_metadata.json")
    parser.add_argument(
        "--ptproxy_dir",
        type=Path,
        default=benchmark_root / "proxy_score" / "ptproxy",
        help="Final BlockMixed pTProxy v1 scores used by the paper.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=benchmark_root / "proxy_score" / "baseline",
        help="Merged baseline proxy scores stored as score_<dataset>.csv files.",
    )
    parser.add_argument("--output_dir", type=Path, default=paper_root)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = load_selected_groups(args.space_json)
    train_df = pd.read_csv(args.train_csv)
    train_df["block_specs"] = train_df["block_specs"].astype(str)

    summary_rows: list[dict] = []
    for dataset, groups in selected_groups.items():
        train_ds = train_df[train_df["dataset"] == dataset].copy()
        if train_ds.empty:
            continue
        metric_name = str(train_ds["metric"].iloc[0])
        train_ds["target_perf"] = metric_to_perf(metric_name, train_ds["test_metric"])

        proxy_frames: list[tuple[str, pd.DataFrame]] = []

        ptproxy_path = args.ptproxy_dir / f"score_{dataset}_{FINAL_PTPROXY_VARIANT}.csv"
        if ptproxy_path.exists():
            proxy_frames.append((FINAL_PTPROXY_VARIANT, pd.read_csv(ptproxy_path)))

        baseline_path = args.baseline_dir / f"score_{dataset}.csv"
        if baseline_path.exists():
            classic_df = pd.read_csv(baseline_path)
            for proxy in CLASSIC_PROXIES:
                proxy_frames.append((proxy, classic_df[classic_df["proxy"] == proxy].copy()))

        for proxy, score_df in proxy_frames:
            if score_df.empty:
                continue
            score_df["block_specs"] = score_df["block_specs"].astype(str)
            score_df = score_df[score_df["group"].isin(groups)].copy()
            srcc, n_models = compute_srcc(score_df, train_ds)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "proxy": proxy,
                    "n_models": n_models,
                    "metric": metric_name,
                    "srcc_test_perf": srcc,
                    "abs_srcc": abs(srcc) if np.isfinite(srcc) else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "blockmixed_selected_proxy_srcc_summary.csv", index=False)

    if not summary_df.empty:
        matrix = summary_df.pivot(index="proxy", columns="dataset", values="abs_srcc")
        print(matrix.round(4).to_string())


if __name__ == "__main__":
    main()

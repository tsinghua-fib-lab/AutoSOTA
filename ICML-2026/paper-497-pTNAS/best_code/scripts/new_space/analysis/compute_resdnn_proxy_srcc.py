from __future__ import annotations

import argparse
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    benchmark_root = project_root / "datasets" / "nas_bench_tabular" / "space_resdnn"
    architecture_root = benchmark_root / "architecture"
    parser = argparse.ArgumentParser(description="Compute SRCC for resnet-pool proxy scores.")
    parser.add_argument(
        "--ptproxy_dir",
        type=Path,
        default=benchmark_root / "proxy_score" / "ptproxy",
        help="Final ResDNN pTProxy v1 scores used by the paper.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=benchmark_root / "proxy_score" / "baseline",
        help="Merged baseline proxy scores stored as score_<dataset>.csv files.",
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=benchmark_root / "training" / "resnet_pool_results.csv",
    )
    parser.add_argument(
        "--space_regr",
        type=Path,
        default=architecture_root / "random_sampled_arch_resdnn_regression.txt",
        help="Final regression ResDNN architecture list used by the paper.",
    )
    parser.add_argument(
        "--space_cls",
        type=Path,
        default=architecture_root / "random_sampled_arch_resdnn_classification.txt",
        help="Final classification ResDNN architecture list used by the paper.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "run_outputs" / "data" / "new_space",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_df = train_df[train_df["space_name"] == "resnet"].copy()
    train_df["architecture"] = train_df["architecture"].astype(str)
    datasets = sorted(train_df["dataset"].astype(str).unique().tolist())
    regr_archs = set(args.space_regr.read_text(encoding="utf-8").splitlines())
    cls_archs = set(args.space_cls.read_text(encoding="utf-8").splitlines())

    summary_rows: list[dict] = []

    for dataset in datasets:
        train_ds = train_df[train_df["dataset"] == dataset].copy()
        if train_ds.empty:
            continue

        metric_name = str(train_ds["metric"].iloc[0])
        final_archs = regr_archs if metric_name == "mean_absolute_error" else cls_archs
        train_ds = train_ds[train_ds["architecture"].isin(final_archs)].copy()
        train_ds["target_perf"] = metric_to_perf(metric_name, train_ds["test_metric"])

        proxy_frames: list[tuple[str, pd.DataFrame]] = []
        ptproxy_path = args.ptproxy_dir / f"score_{dataset}_{FINAL_PTPROXY_VARIANT}.csv"
        if ptproxy_path.exists():
            proxy_frames.append((FINAL_PTPROXY_VARIANT, pd.read_csv(ptproxy_path)))

        baseline_path = args.baseline_dir / f"score_{dataset}.csv"
        if baseline_path.exists():
            classic_df = pd.read_csv(baseline_path)
            if "proxy" not in classic_df.columns and "variant" in classic_df.columns:
                classic_df = classic_df.rename(columns={"variant": "proxy"})
            for proxy in CLASSIC_PROXIES:
                proxy_frames.append((proxy, classic_df[classic_df["proxy"] == proxy].copy()))

        for variant, score_df in proxy_frames:
            if score_df.empty:
                continue
            score_df["architecture"] = score_df["architecture"].astype(str)
            score_df = score_df[score_df["architecture"].isin(final_archs)].copy()
            merged = train_ds.merge(
                score_df[["dataset", "architecture", "proxy_score", "error"]],
                on=["dataset", "architecture"],
                how="inner",
            )
            merged = merged[np.isfinite(merged["proxy_score"])].copy()
            merged = merged[np.isfinite(merged["target_perf"])].copy()

            n_models = len(merged)
            srcc = np.nan
            if n_models >= 2:
                srcc = float(spearmanr(merged["proxy_score"], merged["target_perf"]).statistic)

            n_failures = int((score_df["error"].fillna("") != "").sum())

            summary_rows.append(
                {
                    "dataset": dataset,
                    "proxy": variant,
                    "metric": metric_name,
                    "n_scored_models": len(score_df),
                    "n_matched_models": n_models,
                    "n_failures": n_failures,
                    "srcc_test_perf": srcc,
                    "abs_srcc": abs(srcc) if np.isfinite(srcc) else np.nan,
                    "mean_proxy_time_seconds": float(score_df["proxy_time_seconds"].mean()),
                    "median_proxy_time_seconds": float(score_df["proxy_time_seconds"].median()),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "proxy"]).reset_index(drop=True)

    summary_df.to_csv(args.output_dir / "resnet_pool_proxy_srcc_summary.csv", index=False)

    if not summary_df.empty:
        abs_matrix = summary_df.pivot(index="proxy", columns="dataset", values="abs_srcc")
        print(abs_matrix.round(4).to_string())


if __name__ == "__main__":
    main()

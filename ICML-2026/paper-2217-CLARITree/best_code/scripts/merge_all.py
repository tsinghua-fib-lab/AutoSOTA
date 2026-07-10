# example: python scripts/merge_all.py
# openml example: python scripts/merge_all.py --openml
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_single_method_outer import DATASETS, METHODS, discover_openml_datasets, merge_one, summarize_outer_rows


DEFAULT_N_THRESHOLDS = 20


def threshold_output_label(n_thresholds: int, threshold_label: str | None = None) -> str:
    return threshold_label if threshold_label else str(n_thresholds)


def final_output_path(openml: bool, threshold_label: str) -> Path:
    base = "final_openml" if openml else "final"
    if threshold_label == str(DEFAULT_N_THRESHOLDS):
        return Path("results") / f"{base}.csv"
    return Path("results") / f"{base}_{threshold_label}.csv"


def resolve_threshold_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[int, str | None]:
    if args.full and args.threshold_5:
        parser.error("--full and --5 cannot be used together")
    if args.full:
        if args.threshold_label and args.threshold_label != "full":
            parser.error("--full cannot be combined with --threshold_label other than full")
        return args.n_thresholds, "full"
    if args.threshold_5:
        if args.threshold_label:
            parser.error("--5 cannot be combined with --threshold_label")
        return 5, None
    return args.n_thresholds, args.threshold_label


def merge_all(
    depth: int = 4,
    n_thresholds: int = 20,
    require_complete: bool = False,
    openml: bool = False,
    threshold_label: str | None = None,
) -> pd.DataFrame:
    datasets = discover_openml_datasets() if openml else DATASETS
    if openml and not datasets:
        raise RuntimeError("No OpenML datasets found under data/openml")

    parts = []
    for method in METHODS:
        for dataset in datasets:
            merged = merge_one(
                method,
                dataset,
                depth=depth,
                n_thresholds=n_thresholds,
                require_complete=require_complete,
                write=False,
                openml=openml,
                threshold_label=threshold_label,
            )
            if merged is None:
                continue
            mean_df, std_df, best_df = summarize_outer_rows(
                merged[~merged["outer"].isin(["mean", "std", "best_by_mean_val_r2"])].copy()
            )
            parts.extend([mean_df, std_df, best_df])

    if not parts:
        raise RuntimeError("No merged rows found")
    final_df = pd.concat(parts, ignore_index=True, sort=False)
    output_label = threshold_output_label(n_thresholds, threshold_label)
    final_df["threshold_label"] = output_label
    out_csv = final_output_path(openml, output_label)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv} ({len(final_df)} rows)")
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_thresholds", type=int, default=20)
    parser.add_argument("--5", dest="threshold_5", action="store_true", help="Merge threshold=5 results.")
    parser.add_argument("--full", action="store_true", help="Merge threshold=full results.")
    parser.add_argument(
        "--threshold_label",
        default=None,
        help="Optional result directory label, e.g. full.",
    )
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--openml", action="store_true", help="Merge datasets under data/openml.")
    args = parser.parse_args()
    if args.threshold_label and ("/" in args.threshold_label or "\\" in args.threshold_label):
        parser.error("--threshold_label must not contain path separators")
    n_thresholds, threshold_label = resolve_threshold_args(args, parser)
    merge_all(args.depth, n_thresholds, args.require_complete, args.openml, threshold_label)


if __name__ == "__main__":
    main()

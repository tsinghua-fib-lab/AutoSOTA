#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all summary.csv files into one all_summary.csv"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="~/IsoCLIP/exp_img-img_retrieval/results/clip_b32_img_retrieval",
        help="Root folder containing exp_* subfolders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: <root>/all_summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else root / "all_summary.csv"
    )

    csv_files = sorted(root.glob("exp_*/summary.csv"))

    if not csv_files:
        print(f"No summary.csv found in: {root}")
        return

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = str(csv_file)
            dfs.append(df)
            print(f"Loaded: {csv_file}")
        except Exception as e:
            print(f"Skipping {csv_file} due to error: {e}")

    if not dfs:
        print("No valid CSV files could be loaded.")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    keep_cols = [
        "dataset_name",
        "clip_model_name",
        "query_eval_type",
        "gallery_eval_type",
        "no_iso",
        "iso_ktop",
        "iso_kbottom",
        "use_open_clip",
        "open_clip_pretrained",
        "out_path",
        "mAP",
        "timestamp",
        "folder_path",
        "source_file",
    ]

    # Keep only columns that actually exist
    keep_cols = [c for c in keep_cols if c in all_df.columns]
    all_df = all_df[keep_cols]

    all_df = all_df.drop_duplicates()

    # Optional: sort for readability
    sort_cols = [c for c in ["clip_model_name", "dataset_name", "timestamp"] if c in all_df.columns]
    if sort_cols:
        all_df = all_df.sort_values(sort_cols).reset_index(drop=True)

    all_df.to_csv(output_path, index=False)

    print(f"\nSaved aggregated CSV to: {output_path}")
    print(f"Total rows: {len(all_df)}")


if __name__ == "__main__":
    main()
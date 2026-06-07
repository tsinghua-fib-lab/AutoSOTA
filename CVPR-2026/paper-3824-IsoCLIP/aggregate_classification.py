#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all classification_summary.csv files into one all_classification.csv"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="~/IsoCLIP/exp_img_classification/clip_b32_classification",
        help="Root folder containing exp_* subfolders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: <root>/all_classification.csv",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else root / "all_classification.csv"
    )

    csv_files = sorted(root.glob("exp_*/classification_summary.csv"))

    if not csv_files:
        print(f"No classification_summary.csv found in: {root}")
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

    # Optional: remove exact duplicate rows
    all_df = all_df.drop_duplicates()

    all_df.to_csv(output_path, index=False)
    print(f"\nSaved aggregated CSV to: {output_path}")
    print(f"Total rows: {len(all_df)}")


if __name__ == "__main__":
    main()
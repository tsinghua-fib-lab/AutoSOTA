# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Aggregate performance estimation results into a clean pivot table.

Produces a simple table: rows = datasets, columns = methods, cells = MAE.

Usage::

    python -m experiment.aggregate_results --results-dir results/

    # Only aggregate new unified runs (exclude old format)
    python -m experiment.aggregate_results --results-dir results/ --latest

    # Filter by setting
    python -m experiment.aggregate_results --results-dir results/ --setting new_pair
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment.helper import METHOD_ORDER


def parse_filename(filename: str) -> dict:
    """Extract dataset, target_model, budget, setting from summary filename.

    Supported formats:
      New:  summary_{dataset}_{target_model}_budget{N}_{setting}_{timestamp}.csv
      Old:  summary_{dataset}_{target_model}_{setting}_{timestamp}.csv
    """
    basename = os.path.basename(filename).replace("summary_", "").replace(".csv", "")

    for setting in ["new_pair", "new_benchmark", "new_model"]:
        # Try format with budget tag first
        pattern_budget = rf"^(.+?)_(.+?)_budget(\d+)_{setting}_(\d{{8}}_\d{{6}})$"
        m = re.match(pattern_budget, basename)
        if m:
            return {
                "dataset": m.group(1),
                "target_model": m.group(2),
                "budget": int(m.group(3)),
                "setting": setting,
                "timestamp": m.group(4),
            }

        # Fall back to old format (no budget)
        pattern = rf"^(.+?)_(.+?)_{setting}_(\d{{8}}_\d{{6}})$"
        m = re.match(pattern, basename)
        if m:
            return {
                "dataset": m.group(1),
                "target_model": m.group(2),
                "budget": None,
                "setting": setting,
                "timestamp": m.group(3),
            }

    return {"dataset": basename, "target_model": "", "budget": None, "setting": "", "timestamp": ""}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results into a clean dataset × method table"
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--setting", type=str, default=None,
                        help="Filter by setting (new_pair, new_benchmark, new_model)")
    parser.add_argument("--latest", action="store_true",
                        help="Only include the latest run per dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/aggregated_table.csv)")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "summary_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No summary_*.csv files found in {args.results_dir}")
        return

    # Parse and filter files
    records = []
    for f in files:
        info = parse_filename(f)
        info["path"] = f
        records.append(info)

    # Filter by setting if specified
    if args.setting:
        records = [r for r in records if r["setting"] == args.setting]

    # If --latest, keep only the most recent file per dataset
    if args.latest:
        latest = {}
        for r in records:
            key = r["dataset"]
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        records = list(latest.values())

    if not records:
        print("No matching files found after filtering.")
        return

    print(f"Aggregating {len(records)} result files:")
    for r in records:
        print(f"  {r['dataset']:20s} | {r['setting']:15s} | {os.path.basename(r['path'])}")

    # Load and merge
    all_rows = []
    for r in records:
        df = pd.read_csv(r["path"])
        df["Dataset"] = r["dataset"]
        df["Setting"] = r["setting"]
        all_rows.append(df)

    merged = pd.concat(all_rows, ignore_index=True)

    # Build pivot table: Dataset (rows) × Method (columns), cells = Mean MAE
    if "Mean MAE" not in merged.columns:
        print("Error: 'Mean MAE' column not found in result files.")
        print(f"Available columns: {merged.columns.tolist()}")
        return

    pivot = merged.pivot_table(
        index="Dataset",
        columns="Method",
        values="Mean MAE",
        aggfunc="first",  # take first if duplicates
    )

    # Reorder columns by canonical method order (only include methods that exist)
    ordered_cols = [m for m in METHOD_ORDER if m in pivot.columns]
    extra_cols = [m for m in pivot.columns if m not in METHOD_ORDER]
    pivot = pivot[ordered_cols + extra_cols]

    # Sort rows alphabetically
    pivot = pivot.sort_index()

    # Save
    output_path = args.output or os.path.join(args.results_dir, "aggregated_table.csv")
    pivot.to_csv(output_path)
    print(f"\nSaved to: {output_path}")

    # Print clean table
    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS (Mean MAE)")
    print("=" * 120)

    # Format with 4 decimal places, handle NaN
    formatted = pivot.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )

    print(formatted.to_string())
    print()


if __name__ == "__main__":
    main()

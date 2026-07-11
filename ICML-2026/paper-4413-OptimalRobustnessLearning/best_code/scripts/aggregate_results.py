#!/usr/bin/env python
"""Aggregate per-dataset benchmark CSVs into a per-algorithm mean table.

Reads flat CSV files from ``stat/<dataset>_<name>[_<fraction>].csv``,
auto-discovers available predictor types from the stat directory,
computes per-algorithm mean (and std) of Hit Rate, Cost Ratio, and
LRU-normalized Cost Ratio across datasets.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict


CSV_COST_COL = "Cost Ratio"
DISPLAY_COLS = ["Hit Rate", "Cost Ratio", "LRU-norm Cost Ratio"]

SPEC2006_DATASETS = [
    "astar", "bwaves", "bzip", "cactusadm", "gems",
    "lbm", "leslie3d", "libq", "mcf", "milc",
    "omnetpp", "sphinx3", "xalanc",
]


def discover_predictors(results_dir, datasets):
    """Scan stat/ for available predictor types based on CSV filenames."""
    predictors = set()
    for f in os.listdir(results_dir):
        if not f.endswith(".csv"):
            continue
        for ds in datasets:
            if f.startswith(ds + "_"):
                rest = f[len(ds) + 1:]
                rest = rest.removesuffix(".csv")
                # rest is either "<pred>_<fraction>" or "<pred>"
                parts = rest.rsplit("_", 1)
                if len(parts) == 2 and parts[1].replace(".", "", 1).isdigit():
                    predictors.add(parts[0])
                else:
                    predictors.add(rest)
                break
    return sorted(predictors)


def find_csv(results_dir, dataset, name, fraction):
    path = os.path.join(results_dir, f"{dataset}_{name}_{fraction}.csv")
    if os.path.isfile(path):
        return path
    path = os.path.join(results_dir, f"{dataset}_{name}.csv")
    if os.path.isfile(path):
        return path
    return None


def aggregate_one(results_dir, name, fraction, expected):
    """Aggregate results for a single predictor type. Returns (order, per_alg_values, n_found, errors)."""
    per_alg_values = defaultdict(lambda: [[] for _ in DISPLAY_COLS])
    order = []
    errors = []
    n_found = 0

    for dataset in expected:
        path = find_csv(results_dir, dataset, name, fraction)
        if not path:
            continue
        n_found += 1
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                dataset_rows = {}
                for row in reader:
                    alg = row["Name"]
                    hit_rate = float(row["Hit Rate"])
                    cost_ratio = float(row[CSV_COST_COL])
                    if math.isnan(hit_rate) or math.isinf(hit_rate):
                        continue
                    if math.isnan(cost_ratio) or math.isinf(cost_ratio):
                        continue
                    dataset_rows[alg] = (hit_rate, cost_ratio)

                lru_cost = dataset_rows.get("LRU", (None, None))[1]
                opt_cost = 1.0

                for alg, (hit_rate, cost_ratio) in dataset_rows.items():
                    if alg not in per_alg_values or not per_alg_values[alg][0]:
                        order.append(alg)
                    if lru_cost is not None and lru_cost != opt_cost:
                        lru_norm = (cost_ratio - opt_cost) / (lru_cost - opt_cost)
                    else:
                        lru_norm = 0.0
                    per_alg_values[alg][0].append(hit_rate)
                    per_alg_values[alg][1].append(cost_ratio)
                    per_alg_values[alg][2].append(lru_norm)
        except (KeyError, ValueError) as e:
            errors.append((dataset, str(e)))

    # deduplicate order
    seen = set()
    unique = []
    for a in order:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique, per_alg_values, n_found, errors


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def print_table(title, order, per_alg_values, n_found):
    print(f"\n{'=' * 60}")
    print(f"  {title}  ({n_found} datasets)")
    print(f"{'=' * 60}")

    header_parts = ["Name", "N"]
    for col in DISPLAY_COLS:
        header_parts.append(col)
        header_parts.append("std")
    col_widths = []
    for h in header_parts:
        if h == "Name":
            col_widths.append(max(len(h), 40))
        elif h == "std":
            col_widths.append(8)
        else:
            col_widths.append(max(len(h), 10))

    def fmt_row(vals):
        return "| " + " | ".join(
            v.ljust(w) if isinstance(v, str) else v.rjust(w)
            for v, w in zip(vals, col_widths)
        ) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(sep)
    print(fmt_row(header_parts))
    print(sep)
    for name in order:
        vals = per_alg_values[name]
        n = len(vals[0])
        row = [name, str(n)]
        for i in range(len(DISPLAY_COLS)):
            row.append(f"{mean(vals[i]):.4f}")
            row.append(f"{std(vals[i]):.4f}")
        print(fmt_row(row))
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark CSVs from stat/ into per-algorithm mean tables.")
    parser.add_argument("--name", default="pleco",
                        help="predictor name to aggregate (default: auto-discover all from stat/)")
    parser.add_argument("--fraction", default="1",
                        help="model_fraction (default: 1)")
    parser.add_argument("--results_dir", default="stat",
                        help="directory containing CSV result files (default: stat)")
    parser.add_argument("--expected", default="spec2006",
                        choices=["spec2006", "none"],
                        help="expected dataset set (default: spec2006)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    if args.expected == "spec2006":
        expected = SPEC2006_DATASETS
    else:
        expected = sorted(set(
            f.split("_")[0] for f in os.listdir(args.results_dir) if f.endswith(".csv")
        ))

    # Determine which predictors to aggregate
    if args.name:
        predictors = [args.name]
    else:
        predictors = discover_predictors(args.results_dir, expected)
        if not predictors:
            print(f"ERROR: No CSV files found in {args.results_dir}/")
            sys.exit(1)
        print(f"Auto-discovered predictors: {', '.join(predictors)}")

    for pred in predictors:
        order, per_alg_values, n_found, errors = aggregate_one(
            args.results_dir, pred, args.fraction, expected)

        if not order:
            print(f"\nWARNING: No data found for predictor '{pred}', skipping.")
            continue

        if errors:
            print(f"\nWARNING: Errors parsing '{pred}':")
            for ds, err in errors:
                print(f"  {ds}: {err}")

        missing = [ds for ds in expected if find_csv(args.results_dir, ds, pred, args.fraction) is None]
        if missing:
            print(f"\nWARNING: Missing datasets for '{pred}': {', '.join(missing)}")

        print_table(pred, order, per_alg_values, n_found)


if __name__ == "__main__":
    main()

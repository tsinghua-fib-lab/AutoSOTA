#!/usr/bin/env python3
"""Summarize final internal-state parameters from `state_index.csv` traces."""

import argparse
import csv
import os
from collections import defaultdict

from _project import BASE_DIR, repo_relpath

def read_state_index(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_last_state_row(path: str) -> dict | None:
    last = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def safe_float(val: str, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val: str, default: int = -1) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing state_index.csv")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    results_dir = os.path.abspath(args.results_dir)
    state_index_path = os.path.join(results_dir, "state_index.csv")
    if not os.path.isfile(state_index_path):
        raise SystemExit(f"Missing: {state_index_path}")

    rows = read_state_index(state_index_path)
    if not rows:
        raise SystemExit("No rows in state_index.csv")

    out_rows = []
    for r in rows:
        state_file = r.get("state_file", "")
        if not state_file or not os.path.isfile(state_file):
            continue
        last = read_last_state_row(state_file)
        if not last:
            continue
        out_rows.append(
            {
                "algorithm": r.get("algorithm", ""),
                "budget_multiplier": safe_int(r.get("budget_multiplier", "")),
                "function": safe_int(r.get("function", "")),
                "dimension": safe_int(r.get("dimension", "")),
                "instance": safe_int(r.get("instance", "")),
                "noise_model": r.get("noise_model", ""),
                "noise_sigma": safe_float(r.get("noise_sigma", "")),
                "final_evals": safe_int(last.get("evals", "")),
                "final_generation": safe_int(last.get("generation", "")),
                "final_noise_ema": safe_float(last.get("noise_ema", "")),
                "final_temp_scale": safe_float(last.get("temp_scale", "")),
                "final_noise_s0": safe_float(last.get("noise_s0", "")),
                "final_noise_s1": safe_float(last.get("noise_s1", "")),
                "final_noise_z_pool_size": safe_int(last.get("noise_z_pool_size", "")),
                "final_noise_z_abs_median": safe_float(last.get("noise_z_abs_median", "")),
                "state_file": state_file,
            }
        )

    out_path = os.path.join(results_dir, "state_params_final.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)

    # Quick aggregation by algorithm (mean of finite values).
    by_algo = defaultdict(list)
    for row in out_rows:
        by_algo[row["algorithm"]].append(row)

    agg_path = os.path.join(results_dir, "state_params_final_by_algorithm.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "count",
                "mean_final_noise_s0",
                "mean_final_noise_s1",
                "mean_final_noise_z_pool_size",
                "mean_final_noise_z_abs_median",
            ],
        )
        writer.writeheader()
        for algo, items in sorted(by_algo.items()):
            def mean(field: str) -> float:
                vals = [float(it[field]) for it in items if float(it[field]) == float(it[field])]
                return sum(vals) / len(vals) if vals else float("nan")

            writer.writerow(
                {
                    "algorithm": algo,
                    "count": len(items),
                    "mean_final_noise_s0": mean("final_noise_s0"),
                    "mean_final_noise_s1": mean("final_noise_s1"),
                    "mean_final_noise_z_pool_size": mean("final_noise_z_pool_size"),
                    "mean_final_noise_z_abs_median": mean("final_noise_z_abs_median"),
                }
            )

    print("Wrote:", repo_relpath(out_path))
    print("Wrote:", repo_relpath(agg_path))


if __name__ == "__main__":
    main()

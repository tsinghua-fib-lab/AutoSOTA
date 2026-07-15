#!/usr/bin/env python3
"""
Extract a small "Budget Usage vs Performance" table from trace CSVs produced by
`tools/extract_coco_traces.py`.

This is intended for the Hansen fixed-budget evidence package:
- choose a few evals/D checkpoints (e.g., 10,25,50,100),
- report median best-so-far (noise-free delta) for each algorithm and function.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np

from _project import BASE_DIR, repo_relpath

def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def parse_csv_header(header: list[str]) -> dict[str, int]:
    """
    Return mapping: algo -> idx_of_median_column
    """
    out: dict[str, int] = {}
    for idx, name in enumerate(header):
        m = re.match(r"^(.*):median$", str(name).strip())
        if not m:
            continue
        algo = str(m.group(1)).strip()
        out[algo] = int(idx)
    return out


def step_at(xs: np.ndarray, ys: np.ndarray, xq: float) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size <= 0:
        return float("nan")
    idx = int(np.searchsorted(xs, float(xq), side="right") - 1)
    idx = int(np.clip(idx, 0, int(xs.size) - 1))
    return float(ys[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", required=True, help="Directory with trace_noisefree_f*_d*.csv files.")
    parser.add_argument("--functions", required=True, help="Function ids, e.g. '108,110,114,120'.")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--budgets-per-d", default="10,25,50,100", help="evals/D checkpoints.")
    parser.add_argument(
        "--algorithms",
        default="",
        help="Comma-separated algorithms to include (default: all found in the first CSV).",
    )
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    csv_dir = os.path.abspath(str(args.csv_dir))
    dim = int(args.dim)
    functions = parse_int_list(str(args.functions))
    budgets_per_d = parse_float_list(str(args.budgets_per_d))

    if not functions or not budgets_per_d:
        raise SystemExit("Empty functions/budgets-per-d.")

    # Discover algorithms from the first function's CSV if not specified.
    first_csv = os.path.join(csv_dir, f"trace_noisefree_f{int(functions[0])}_d{int(dim)}.csv")
    if not os.path.isfile(first_csv):
        raise FileNotFoundError(f"Missing: {first_csv}")
    with open(first_csv, newline="") as f:
        header = next(csv.reader(f))
    algo_to_idx = parse_csv_header(header)
    if not algo_to_idx:
        raise SystemExit(f"No ':median' columns found in {first_csv}")

    if str(args.algorithms).strip():
        algorithms = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    else:
        algorithms = sorted(algo_to_idx.keys())

    out_rows: list[dict[str, object]] = []

    for func_id in functions:
        csv_path = os.path.join(csv_dir, f"trace_noisefree_f{int(func_id)}_d{int(dim)}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        algo_to_idx = parse_csv_header(header)
        xs = np.asarray([float(r[0]) for r in rows[1:] if r and r[0] != ""], dtype=float)  # evals/D

        for alg in algorithms:
            if alg not in algo_to_idx:
                continue
            ys = np.asarray([float(r[algo_to_idx[alg]]) for r in rows[1:] if len(r) > algo_to_idx[alg] and r[algo_to_idx[alg]] != ""], dtype=float)
            for bp in budgets_per_d:
                out_rows.append(
                    {
                        "function": int(func_id),
                        "function_index": int(func_id) - 100 if int(func_id) >= 101 else int(func_id),
                        "dimension": int(dim),
                        "algorithm": str(alg),
                        "evals_per_D": float(bp),
                        "evals": float(bp) * float(dim),
                        "median_best_delta": step_at(xs, ys, float(bp)),
                    }
                )

    out_csv = os.path.abspath(str(args.output_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "function",
                "function_index",
                "dimension",
                "algorithm",
                "evals_per_D",
                "evals",
                "median_best_delta",
            ],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print("Wrote:", repo_relpath(out_csv))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot best-(noise-free) value vs evaluations from COCO `exdata/` folders.

Why:
- COCO post-processing (cocopp) focuses on ERT/ECDF/targets and does not provide
  a direct "best value vs evaluations" trace plot.
- For bbob-noisy, COCO log files contain both "measured" (noisy) and "noise-free"
  best-so-far values. The latter is often the more informative curve.

This script parses COCO `.dat` files (bbob logger format) and produces:
- per (function, dimension) median trace across instances (with IQR shading),
- CSV dumps of the plotted curves for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)


def parse_int_list(spec: str) -> list[int]:
    spec = str(spec).strip()
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def read_exdata_list(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            p = raw.strip()
            if not p or p.startswith("#"):
                continue
            out.append(p)
    return out

def guess_budget_multiplier_from_dirname(path: str) -> int | None:
    m = re.search(r"_B(\d+)(?:_|$)", os.path.basename(os.path.abspath(path)))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def read_alg_id(exdata_dir: str, function: int) -> str:
    info_path = os.path.join(exdata_dir, f"bbobexp_f{int(function)}.info")
    if not os.path.isfile(info_path):
        candidates = [p for p in os.listdir(exdata_dir) if p.endswith(".info")]
        if not candidates:
            raise FileNotFoundError(f"Could not find any .info file under {exdata_dir}")
        info_path = os.path.join(exdata_dir, sorted(candidates)[0])
    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    m = re.search(r"algId\s*=\s*'([^']+)'", first)
    return m.group(1) if m else os.path.basename(exdata_dir)


@dataclass(frozen=True)
class RunTrace:
    evals: np.ndarray  # strictly increasing
    best_delta: np.ndarray  # best (noise-free) fitness - fopt (may contain 0)


def parse_bbob_dat(path: str) -> list[RunTrace]:
    """
    Parse a COCO bbob `.dat` file into per-instance traces.

    File format:
    - segments separated by lines starting with '%'
    - data rows: `f_evals g_evals best_noise_free_delta measured best_measured ...`
    """

    runs: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] | None = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                if current is not None and current:
                    runs.append(current)
                current = []
                continue
            if current is None:
                current = []
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                fe = int(parts[0])
                best_delta = float(parts[2])
            except ValueError:
                continue
            current.append((fe, best_delta))

    if current is not None and current:
        runs.append(current)

    out: list[RunTrace] = []
    for rows in runs:
        evals = np.array([r[0] for r in rows], dtype=int)
        best = np.array([r[1] for r in rows], dtype=float)
        # Ensure monotonic best-so-far (safety against any format quirks).
        best = np.minimum.accumulate(best)
        # Ensure strictly increasing eval indices (drop duplicates if any).
        if len(evals) > 1:
            keep = np.concatenate([[True], evals[1:] > evals[:-1]])
            evals = evals[keep]
            best = best[keep]
        out.append(RunTrace(evals=evals, best_delta=best))
    return out


def step_value_at(evals: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation: value at each query eval."""
    idx = np.searchsorted(evals, query, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def make_log_grid(max_x: int, points: int) -> np.ndarray:
    max_x = int(max(1, max_x))
    points = int(max(10, points))
    xs = np.unique(np.round(np.logspace(0, math.log10(max_x), num=points)).astype(int))
    xs[0] = 1
    xs[-1] = max_x
    return xs


def safe_positive(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), float(eps))


@dataclass(frozen=True)
class AggregateTrace:
    x_eval: np.ndarray  # evaluations (absolute)
    y_median: np.ndarray
    y_q25: np.ndarray
    y_q75: np.ndarray


def aggregate_traces(runs: list[RunTrace], x_eval: np.ndarray) -> AggregateTrace:
    if not runs:
        raise ValueError("No runs to aggregate.")
    ys = []
    for r in runs:
        ys.append(step_value_at(r.evals, r.best_delta, x_eval))
    Y = np.stack(ys, axis=0)
    return AggregateTrace(
        x_eval=x_eval,
        y_median=np.median(Y, axis=0),
        y_q25=np.quantile(Y, 0.25, axis=0),
        y_q75=np.quantile(Y, 0.75, axis=0),
    )


def write_curve_csv(path: str, x_per_dim: np.ndarray, curves: dict[str, AggregateTrace]) -> None:
    algs = sorted(curves.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["evals_per_dim"]
        for alg in algs:
            header.extend([f"{alg}:median", f"{alg}:q25", f"{alg}:q75"])
        writer.writerow(header)
        for i in range(len(x_per_dim)):
            row = [float(x_per_dim[i])]
            for alg in algs:
                c = curves[alg]
                row.extend([float(c.y_median[i]), float(c.y_q25[i]), float(c.y_q75[i])])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exdata-dirs", nargs="*", default=[], help="List of COCO exdata directories (one per algo).")
    parser.add_argument(
        "--exdata-list",
        default="",
        help="Text file listing exdata directories (one per line). Useful when paths contain spaces.",
    )
    parser.add_argument("--functions", default="101,110,126,128", help="Function ids, e.g. '101,110' or '101-130'.")
    parser.add_argument("--dims", default="10,20,40", help="Dimensions, e.g. '10,20,40'.")
    parser.add_argument("--instances", default="1-5", help="Instances (segments) to aggregate, default '1-5'.")
    parser.add_argument("--grid-points", type=int, default=180, help="Number of x points (log-spaced).")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots and CSVs.")
    args = parser.parse_args()

    exdata_dirs = [os.path.abspath(p) for p in args.exdata_dirs if str(p).strip()]
    if str(args.exdata_list).strip():
        exdata_dirs.extend([os.path.abspath(p) for p in read_exdata_list(str(args.exdata_list))])
    exdata_dirs = [p for p in exdata_dirs if os.path.isdir(p)]
    exdata_dirs = sorted(set(exdata_dirs))
    if not exdata_dirs:
        raise SystemExit("No valid exdata directories provided.")
    functions = parse_int_list(args.functions)
    dims = parse_int_list(args.dims)
    instances = parse_int_list(args.instances)
    if not functions or not dims or not instances:
        raise SystemExit("Empty functions/dims/instances specification.")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    for func in functions:
        for dim in dims:
            curves: dict[str, AggregateTrace] = {}
            budget_mult: int | None = None

            for exdir in exdata_dirs:
                this_b = guess_budget_multiplier_from_dirname(exdir)
                if this_b is not None:
                    budget_mult = this_b if budget_mult is None else min(budget_mult, this_b)

                alg = read_alg_id(exdir, func)
                dat_dir = os.path.join(exdir, f"data_f{int(func)}")
                if not os.path.isdir(dat_dir):
                    raise FileNotFoundError(f"Missing folder {dat_dir}")
                candidates = [p for p in os.listdir(dat_dir) if p.endswith(f"_DIM{int(dim)}.dat") and f"_f{int(func)}_" in p]
                if not candidates:
                    raise FileNotFoundError(f"Could not find .dat for f{func} DIM{dim} under {dat_dir}")
                dat_path = os.path.join(dat_dir, sorted(candidates)[0])

                runs_all = parse_bbob_dat(dat_path)
                runs = []
                for idx, r in enumerate(runs_all, start=1):
                    if idx in instances:
                        runs.append(r)
                if not runs:
                    raise ValueError(f"No runs left after instance filtering in {dat_path}")

                max_eval = None
                if this_b is not None:
                    max_eval = int(this_b) * int(dim)
                else:
                    max_eval = int(max(r.evals[-1] for r in runs))
                x_eval = make_log_grid(max_eval, args.grid_points)
                curves[alg] = aggregate_traces(runs, x_eval)

            if not curves:
                continue

            # Prefer plotting in evals-per-dim space for comparability across dims.
            first = next(iter(curves.values()))
            x_per_dim = first.x_eval.astype(float) / float(dim)

            # Plot
            plt.figure(figsize=(7.8, 5.0))
            for alg in sorted(curves.keys()):
                c = curves[alg]
                y_med = safe_positive(c.y_median)
                y_lo = safe_positive(c.y_q25)
                y_hi = safe_positive(c.y_q75)
                plt.plot(x_per_dim, y_med, linewidth=2.0, label=alg)
                plt.fill_between(x_per_dim, y_lo, y_hi, alpha=0.18)

            plt.xscale("log")
            plt.yscale("log")
            btxt = f", B={budget_mult}×D" if budget_mult is not None else ""
            plt.title(f"Best noise-free (f - fopt) vs evals/D | f{func}, D={dim}{btxt}")
            plt.xlabel("Evaluations / D (log)")
            plt.ylabel("Best noise-free f - fopt (log)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"trace_noisefree_f{func}_d{dim}.png")
            plt.savefig(plot_path, dpi=220)
            plt.close()

            csv_path = os.path.join(csv_dir, f"trace_noisefree_f{func}_d{dim}.csv")
            write_curve_csv(csv_path, x_per_dim, curves)

    print("Wrote plots to:", plots_dir)
    print("Wrote curve CSVs to:", csv_dir)


if __name__ == "__main__":
    main()

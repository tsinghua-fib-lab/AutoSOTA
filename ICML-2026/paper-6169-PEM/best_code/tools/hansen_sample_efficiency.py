#!/usr/bin/env python3
"""
Fixed-budget sample-efficiency analysis for COCO bbob-noisy exdata.

This is designed for the fixed-budget "Hansen test" argument:

  Selection-stage uncertainty integration (BERW) is more sample-efficient than
  evaluation-stage uncertainty reduction (UH-CMA-ES) under a fixed evaluation budget.

We compute (using COCO noise-free deltas stored in `.dat` files):

1) Performance-at-budget points:
   - median log10(best_delta) across (function, instance) runs at evals/D ∈ {…}

2) Relative hitting times:
   - for each run, define targets as a *relative* improvement over the initial delta:
       target = initial_delta * rel_factor
   - report success rate and median evals/D to reach each rel_factor.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

# Local tools modules (import works because tools/ is on sys.path for tools scripts).
from extract_coco_traces import parse_bbob_dat, step_value_at  # type: ignore
from summarize_coco_noisefree_from_exdata import parse_info_file  # type: ignore

from _project import repo_relpath

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


def read_exdata_list(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            p = raw.strip()
            if not p or p.startswith("#"):
                continue
            out.append(p)
    return out


def safe_log10(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, float(eps)))


def first_hit_evals(evals: np.ndarray, best_delta: np.ndarray, target: float) -> int | None:
    target = float(target)
    if not np.isfinite(target):
        return None
    idx = np.where(best_delta <= target)[0]
    if idx.size == 0:
        return None
    return int(evals[int(idx[0])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exdata-list", required=True, help="Text file listing COCO exdata directories (one per line).")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV tables.")
    parser.add_argument("--dim", type=int, default=40)
    parser.add_argument("--functions", required=True, help="Function ids, e.g. '108,110,114,120' or '101-130'.")
    parser.add_argument("--instances", default="1-5", help="Instance/run ids to include (segment indices).")
    parser.add_argument(
        "--budget-points",
        default="10,25,50,100",
        help="Comma-separated evals/D points to summarize performance at.",
    )
    parser.add_argument(
        "--rel-factors",
        default="0.1,0.01",
        help="Comma-separated relative factors for hitting times (target = initial_delta * factor).",
    )
    args = parser.parse_args()

    dim = int(args.dim)
    functions = parse_int_list(str(args.functions))
    instances = parse_int_list(str(args.instances))
    budget_points = parse_float_list(str(args.budget_points))
    rel_factors = parse_float_list(str(args.rel_factors))

    exdata_dirs = [os.path.abspath(p) for p in read_exdata_list(str(args.exdata_list))]
    exdata_dirs = [p for p in exdata_dirs if os.path.isdir(p)]
    exdata_dirs = sorted(set(exdata_dirs))
    if not exdata_dirs:
        raise SystemExit("No valid exdata dirs.")

    out_dir = os.path.abspath(str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-run wide table.
    per_run_rows: list[dict[str, object]] = []

    # Collect values for summaries.
    # perf_by_budget[(algo, evals_per_dim)] -> list[log10(delta)]
    perf_by_budget: dict[tuple[str, float], list[float]] = defaultdict(list)
    # hit_times[(algo, rel_factor)] -> list[evals_per_dim]
    hit_times: dict[tuple[str, float], list[float]] = defaultdict(list)
    # hit_success[(algo, rel_factor)] -> (reached, total)
    hit_counts: dict[tuple[str, float], list[int]] = defaultdict(list)

    # Cache parsed dat segments per (exdir, dat_relpath).
    dat_cache: dict[tuple[str, str], list] = {}

    for exdir in exdata_dirs:
        info_files = [p for p in os.listdir(exdir) if p.startswith("bbobexp_f") and p.endswith(".info")]
        for info_name in sorted(info_files):
            info_path = os.path.join(exdir, info_name)
            refs = parse_info_file(info_path)
            for ref in refs:
                if int(ref.dim) != int(dim):
                    continue
                if int(ref.func_id) not in set(functions):
                    continue

                dat_key = (exdir, ref.dat_relpath)
                if dat_key not in dat_cache:
                    dat_path = os.path.join(exdir, ref.dat_relpath)
                    if not os.path.isfile(dat_path):
                        continue
                    dat_cache[dat_key] = parse_bbob_dat(dat_path)

                segments = dat_cache[dat_key]
                n = min(len(segments), len(ref.run_ids))
                segments = segments[:n]
                run_ids = ref.run_ids[:n]

                for seg, run_id in zip(segments, run_ids):
                    inst = int(run_id)
                    if inst not in set(instances):
                        continue

                    evals = np.asarray(seg.evals, dtype=int)
                    best = np.asarray(seg.best_delta, dtype=float)
                    if evals.size <= 0:
                        continue
                    initial = float(best[0])
                    final = float(best[-1])

                    row: dict[str, object] = {
                        "algorithm": str(ref.alg_id),
                        "function": int(ref.func_id),
                        "dimension": int(ref.dim),
                        "instance": int(inst),
                        "initial_delta": float(initial),
                        "final_delta": float(final),
                    }

                    # Performance at budget points.
                    for bp in budget_points:
                        bp = float(bp)
                        q = np.asarray([max(1, int(round(bp * float(dim))))], dtype=int)
                        val = float(step_value_at(evals, best, q)[0])
                        row[f"delta_at_{bp:g}D"] = float(val)
                        perf_by_budget[(str(ref.alg_id), float(bp))].append(float(safe_log10(np.asarray([val]))[0]))

                    # Relative hitting times.
                    for rf in rel_factors:
                        rf = float(rf)
                        target = float(initial) * float(rf)
                        hit_eval = first_hit_evals(evals, best, target)
                        key = (str(ref.alg_id), float(rf))
                        hit_counts[key].append(1 if hit_eval is not None else 0)
                        if hit_eval is not None:
                            hit_times[key].append(float(hit_eval) / float(dim))
                            row[f"hit_evals_per_D_r{rf:g}"] = float(hit_eval) / float(dim)
                        else:
                            row[f"hit_evals_per_D_r{rf:g}"] = ""

                    per_run_rows.append(row)

    if not per_run_rows:
        raise SystemExit("No runs parsed. Check filters (functions/dim/instances).")

    # Write per-run table.
    per_run_csv = os.path.join(out_dir, "per_run_sample_efficiency.csv")
    # Stable column order.
    base_cols = ["algorithm", "function", "dimension", "instance", "initial_delta", "final_delta"]
    budget_cols = [f"delta_at_{float(bp):g}D" for bp in budget_points]
    hit_cols = [f"hit_evals_per_D_r{float(rf):g}" for rf in rel_factors]
    fieldnames = base_cols + budget_cols + hit_cols
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_run_rows:
            w.writerow(row)

    # Write performance-by-budget summary (median in log10 space).
    perf_csv = os.path.join(out_dir, "performance_by_budget.csv")
    with open(perf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "evals_per_D", "n", "median_log10_delta", "q25_log10_delta", "q75_log10_delta", "geom_median_delta"])
        for (algo, bp), vals in sorted(perf_by_budget.items(), key=lambda x: (x[0][0], x[0][1])):
            arr = np.asarray(vals, dtype=float)
            med = float(np.median(arr))
            q25 = float(np.quantile(arr, 0.25))
            q75 = float(np.quantile(arr, 0.75))
            w.writerow([algo, float(bp), int(arr.size), med, q25, q75, float(10.0**med)])

    # Write hitting-time summary.
    hit_csv = os.path.join(out_dir, "hitting_time_by_relative_factor.csv")
    with open(hit_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "rel_factor", "n_total", "n_reached", "success_rate", "median_hit_evals_per_D"])
        for key in sorted(hit_counts.keys(), key=lambda x: (x[0], x[1])):
            algo, rf = key
            reached_flags = np.asarray(hit_counts[key], dtype=int)
            n_total = int(reached_flags.size)
            n_reached = int(np.sum(reached_flags))
            success = float(n_reached / n_total) if n_total else float("nan")
            if n_reached > 0:
                med_hit = float(np.median(np.asarray(hit_times[key], dtype=float)))
            else:
                med_hit = float("nan")
            w.writerow([algo, float(rf), n_total, n_reached, success, med_hit])

    print("Wrote:", repo_relpath(per_run_csv))
    print("Wrote:", repo_relpath(perf_csv))
    print("Wrote:", repo_relpath(hit_csv))


if __name__ == "__main__":
    main()

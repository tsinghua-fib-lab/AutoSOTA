#!/usr/bin/env python3
"""
Probe ROC / budget sweep on bbob-noisy.

Motivation:
- ProbeSwitch relies on an online probe; this script quantifies how reliable that probe is
  as its evaluation budget changes.
- We treat the probe as a classifier and measure ROC / AUC as a function of probe budget
  (here: the number of candidates λ used by the misranking probe).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import cocoex
import numpy as np

from _project import BASE_DIR, repo_relpath
from algorithms import probe_switch as ms


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def format_filter(dims: list[int], funcs: list[int], instances: list[int]) -> str:
    dims_s = ",".join(str(d) for d in dims)
    funcs_s = ",".join(str(f) for f in funcs)
    inst_s = ",".join(str(i) for i in instances)
    return f"dimensions:{dims_s} function_indices:{funcs_s} instance_indices:{inst_s}"


def read_summary(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "algorithm": str(row["algorithm"]),
                    "budget_multiplier": int(row["budget_multiplier"]),
                    "function": int(row["function"]),
                    "dimension": int(row["dimension"]),
                    "instance": int(row["instance"]),
                    "best_f": float(row["best_f"]),
                }
            )
    return rows


def is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


def auc_trapezoid(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    order = np.argsort(np.asarray(xs, dtype=float))
    xs_s = np.asarray(xs, dtype=float)[order]
    ys_s = np.asarray(ys, dtype=float)[order]
    area = float(np.trapezoid(ys_s, xs_s))
    return float(max(0.0, min(1.0, area)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--functions", default="1-30")
    parser.add_argument("--instances", default="1-5")
    parser.add_argument("--budget", type=int, default=200, help="Budget multiplier (xD) used in the results CSV.")
    parser.add_argument(
        "--results-dir",
        default="Results/bbob_noisy_d40_i1-15_probe_labels_B200/noisefree",
        help="Directory containing bbob_summary.csv for outcome labels (noise-free or measured).",
    )
    parser.add_argument("--algo-cma", default="CMA-ES-sep")
    parser.add_argument("--algo-berw", default="BERW-Hetero")
    parser.add_argument("--atol", type=float, default=0.0, help="Tie tolerance on best_f (absolute).")
    parser.add_argument("--rtol", type=float, default=0.0, help="Tie tolerance on best_f (relative).")

    parser.add_argument("--lam-list", default="4,8,16,32", help="Comma-separated λ values for the misranking probe.")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--tstep", type=float, default=0.005)
    parser.add_argument("--report-threshold", type=float, default=0.12, help="Also report accuracy at this threshold.")
    parser.add_argument("--output-dir", default="evidence/bbob_noisy_probe_budget_roc")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    dim = int(args.dimension)
    funcs = parse_int_list(args.functions)
    inst = parse_int_list(args.instances)
    budget = int(args.budget)
    lam_list = parse_int_list(args.lam_list)

    results_dir = os.path.join(os.path.abspath(BASE_DIR), str(args.results_dir))
    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = read_summary(summary_path)
    rows = [
        r
        for r in rows
        if r["dimension"] == dim
        and r["budget_multiplier"] == budget
        and ((int(r["function"]) - 100) in funcs)
        and (r["instance"] in inst)
    ]

    by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (int(r["function"]), int(r["instance"]))
        by_key[key][str(r["algorithm"])] = float(r["best_f"])

    # Build labels.
    labeled: dict[tuple[int, int], str] = {}
    n_ties = 0
    for key, vals in sorted(by_key.items()):
        if str(args.algo_cma) not in vals or str(args.algo_berw) not in vals:
            continue
        best_cma = float(vals[str(args.algo_cma)])
        best_berw = float(vals[str(args.algo_berw)])
        if is_tie(best_cma, best_berw, atol=float(args.atol), rtol=float(args.rtol)):
            n_ties += 1
            continue
        labeled[key] = "berw" if best_berw < best_cma else "cma"

    if not labeled:
        raise SystemExit("No labeled points found (check --algo-cma/--algo-berw and --results-dir).")

    # Threshold grid.
    tmin = float(args.tmin)
    tmax = float(args.tmax)
    tstep = float(args.tstep)
    if tstep <= 0:
        raise SystemExit("--tstep must be > 0")
    thresholds = np.arange(tmin, tmax + 0.5 * tstep, tstep, dtype=float)
    report_t = float(args.report_threshold)

    out_dir = os.path.join(os.path.abspath(BASE_DIR), str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    roc_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {
        "setup": {
            "suite": "bbob-noisy",
            "dimension": dim,
            "functions": str(args.functions),
            "instances": str(args.instances),
            "budget_multiplier": budget,
            "outcome_results_dir": os.path.relpath(results_dir, os.path.abspath(BASE_DIR)),
            "algo_cma": str(args.algo_cma),
            "algo_berw": str(args.algo_berw),
            "tie_tolerance": {"atol": float(args.atol), "rtol": float(args.rtol)},
            "threshold_grid": {"tmin": tmin, "tmax": tmax, "tstep": tstep},
            "lam_list": lam_list,
        },
        "counts": {"n_labeled": int(len(labeled)), "n_ties_dropped": int(n_ties)},
        "by_lam": {},
    }

    suite_filter = format_filter([dim], funcs, inst)

    for lam in lam_list:
        # Compute probe values for this λ.
        rd_by_key: dict[tuple[int, int], float | None] = {}
        suite = cocoex.Suite("bbob-noisy", "", suite_filter)
        for problem in suite:
            key = (int(problem.id_function), int(problem.id_instance))
            if key not in labeled:
                continue
            rd_by_key[key] = ms._misranking_probe(problem, max_evals=int(10**9), lam_override=int(lam))

        # ROC sweep over thresholds.
        fprs: list[float] = []
        tprs: list[float] = []
        best_acc = -1.0
        best_t = float("nan")
        acc_at_report_t = float("nan")

        for t in thresholds.tolist():
            tp = fp = tn = fn = 0
            for key, y in labeled.items():
                rd = rd_by_key.get(key, None)
                pred = "berw" if (rd is not None and float(rd) >= float(t)) else "cma"
                if pred == "berw" and y == "berw":
                    tp += 1
                elif pred == "berw" and y == "cma":
                    fp += 1
                elif pred == "cma" and y == "cma":
                    tn += 1
                else:
                    fn += 1

            tpr = float(tp) / float(tp + fn) if (tp + fn) else float("nan")
            fpr = float(fp) / float(fp + tn) if (fp + tn) else float("nan")
            acc = float(tp + tn) / float(tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")

            fprs.append(float(fpr))
            tprs.append(float(tpr))

            roc_rows.append(
                {
                    "lam": int(lam),
                    "threshold": float(t),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "accuracy": float(acc),
                }
            )

            if np.isfinite(acc) and float(acc) > float(best_acc):
                best_acc = float(acc)
                best_t = float(t)
            if abs(float(t) - float(report_t)) <= 1e-12:
                acc_at_report_t = float(acc)

        auc = auc_trapezoid(fprs, tprs)
        summaries["by_lam"][str(lam)] = {
            "lam": int(lam),
            "auc": float(auc),
            "best_accuracy": float(best_acc),
            "best_threshold": float(best_t),
            "accuracy_at_report_threshold": float(acc_at_report_t),
            "report_threshold": float(report_t),
        }

    roc_csv = os.path.join(out_dir, "roc.csv")
    with open(roc_csv, "w", newline="") as f:
        fieldnames = list(roc_rows[0].keys()) if roc_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if roc_rows:
            w.writeheader()
            for r in roc_rows:
                w.writerow(r)

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    print("Wrote:", repo_relpath(roc_csv))
    print("Wrote:", repo_relpath(summary_json))


if __name__ == "__main__":
    main()

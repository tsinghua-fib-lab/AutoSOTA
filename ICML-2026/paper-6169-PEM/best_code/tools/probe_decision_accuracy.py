#!/usr/bin/env python3
"""
Quantify how well a probe predicts "which algorithm should we run?" on bbob-noisy.

This is meant to strengthen the evidence for probe-driven switching:
- given a probe value on a problem instance (computed with a tiny evaluation budget),
- does it correctly predict whether CMA-ES-sep or BERW-Hetero will win (noise-free or measured)?

Outputs (in --output-dir):
- decision_points.csv : per (function,instance) probe values + outcomes
- summary.json        : aggregate accuracy / confusion / baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import cocoex

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


def is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


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


def summarize_classifier(points: list[dict], *, pred_key: str) -> dict:
    usable = [p for p in points if p["label"] in {"cma", "berw"} and p[pred_key] in {"cma", "berw"}]
    n = int(len(usable))
    correct = int(sum(1 for p in usable if p[pred_key] == p["label"]))
    confusion = {
        "pred_cma_label_cma": int(sum(1 for p in usable if p[pred_key] == "cma" and p["label"] == "cma")),
        "pred_cma_label_berw": int(sum(1 for p in usable if p[pred_key] == "cma" and p["label"] == "berw")),
        "pred_berw_label_cma": int(sum(1 for p in usable if p[pred_key] == "berw" and p["label"] == "cma")),
        "pred_berw_label_berw": int(sum(1 for p in usable if p[pred_key] == "berw" and p["label"] == "berw")),
    }
    return {
        "n_non_ties": n,
        "accuracy": (float(correct) / float(n)) if n else float("nan"),
        "correct": correct,
        "confusion": confusion,
    }


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
    parser.add_argument("--misranking-threshold", type=float, default=0.12)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--variance-reps", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.0, help="Tie tolerance on best_f (absolute).")
    parser.add_argument("--rtol", type=float, default=0.0, help="Tie tolerance on best_f (relative).")
    parser.add_argument("--output-dir", default="evidence/bbob_noisy_probe_decision_accuracy")
    args = parser.parse_args()

    dim = int(args.dimension)
    funcs = parse_int_list(args.functions)
    inst = parse_int_list(args.instances)
    suite_filter = format_filter([dim], funcs, inst)

    results_dir = os.path.join(os.path.abspath(BASE_DIR), str(args.results_dir))
    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = read_summary(summary_path)
    rows = [
        r
        for r in rows
        if r["budget_multiplier"] == int(args.budget) and r["dimension"] == int(args.dimension)
    ]

    by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["algorithm"] not in {str(args.algo_cma), str(args.algo_berw)}:
            continue
        by_key[(int(r["function"]), int(r["instance"]))][str(r["algorithm"])] = float(r["best_f"])

    rd_by_key: dict[tuple[int, int], float | None] = {}
    suite_mis = cocoex.Suite("bbob-noisy", "", suite_filter)
    for problem in suite_mis:
        key = (int(problem.id_function), int(problem.id_instance))
        rd_by_key[key] = ms._misranking_probe(problem, max_evals=int(10**9))

    var_by_key: dict[tuple[int, int], float | None] = {}
    suite_var = cocoex.Suite("bbob-noisy", "", suite_filter)
    for problem in suite_var:
        key = (int(problem.id_function), int(problem.id_instance))
        var_by_key[key] = ms._variance_probe(problem, max_evals=int(10**9), reps=int(args.variance_reps))

    points: list[dict] = []
    for (func_id, inst_id), vals in sorted(by_key.items()):
        key = (int(func_id), int(inst_id))
        if str(args.algo_cma) not in vals or str(args.algo_berw) not in vals:
            continue
        best_cma = float(vals[str(args.algo_cma)])
        best_berw = float(vals[str(args.algo_berw)])

        if is_tie(best_cma, best_berw, atol=float(args.atol), rtol=float(args.rtol)):
            label = "tie"
        elif best_berw < best_cma:
            label = "berw"
        else:
            label = "cma"

        rd = rd_by_key.get(key, None)
        rel_sd = var_by_key.get(key, None)

        pred_mis = "cma"
        if rd is not None and float(rd) >= float(args.misranking_threshold):
            pred_mis = "berw"
        pred_var = "cma"
        if rel_sd is not None and float(rel_sd) >= float(args.variance_threshold):
            pred_var = "berw"

        points.append(
            {
                "function": int(func_id),
                "function_index": int(func_id) - 100,
                "dimension": dim,
                "instance": int(inst_id),
                "best_f_cma": best_cma,
                "best_f_berw": best_berw,
                "label": str(label),
                "misranking_rd": "" if rd is None else float(rd),
                "variance_rel_sd": "" if rel_sd is None else float(rel_sd),
                "pred_misranking": pred_mis,
                "pred_variance": pred_var,
            }
        )

    out_dir = os.path.join(os.path.abspath(BASE_DIR), str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    points_path = os.path.join(out_dir, "decision_points.csv")
    with open(points_path, "w", newline="") as f:
        fieldnames = list(points[0].keys()) if points else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if points:
            w.writeheader()
            for p in points:
                w.writerow(p)

    # Aggregate summaries
    n_total = int(len(points))
    n_ties = int(sum(1 for p in points if p["label"] == "tie"))
    n_missing_rd = int(sum(1 for p in points if p["misranking_rd"] == ""))
    n_missing_var = int(sum(1 for p in points if p["variance_rel_sd"] == ""))

    always_cma = [{"label": p["label"], "pred": "cma"} for p in points]
    always_berw = [{"label": p["label"], "pred": "berw"} for p in points]
    baseline_points = [
        {"label": p["label"], "pred_misranking": p["pred_misranking"], "pred_variance": p["pred_variance"]}
        for p in points
    ]

    # Reuse summarize_classifier with adapted keys
    summary = {
        "setup": {
            "suite": "bbob-noisy",
            "dimension": int(args.dimension),
            "functions": str(args.functions),
            "instances": str(args.instances),
            "budget_multiplier": int(args.budget),
            "outcome_results_dir": os.path.relpath(results_dir, os.path.abspath(BASE_DIR)),
            "algo_cma": str(args.algo_cma),
            "algo_berw": str(args.algo_berw),
            "tie_tolerance": {"atol": float(args.atol), "rtol": float(args.rtol)},
            "probe_thresholds": {
                "misranking_rd": float(args.misranking_threshold),
                "variance_rel_sd": float(args.variance_threshold),
                "variance_reps": int(args.variance_reps),
            },
        },
        "counts": {
            "n_total": n_total,
            "n_ties_outcome": n_ties,
            "n_missing_misranking_probe": n_missing_rd,
            "n_missing_variance_probe": n_missing_var,
        },
        "misranking_probe": summarize_classifier(baseline_points, pred_key="pred_misranking"),
        "variance_probe": summarize_classifier(baseline_points, pred_key="pred_variance"),
        "baselines": {
            "always_cma": summarize_classifier(
                [{"label": p["label"], "pred_misranking": p["pred"]} for p in always_cma], pred_key="pred_misranking"
            ),
            "always_berw": summarize_classifier(
                [{"label": p["label"], "pred_misranking": p["pred"]} for p in always_berw], pred_key="pred_misranking"
            ),
        },
        "outputs": {
            "decision_points_csv": os.path.relpath(points_path, os.path.abspath(BASE_DIR)),
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Wrote:", repo_relpath(points_path))
    print("Wrote:", repo_relpath(summary_path))
    print("n_total:", n_total, "n_ties:", n_ties)
    print("misranking accuracy:", summary["misranking_probe"]["accuracy"])
    print("variance accuracy:", summary["variance_probe"]["accuracy"])


if __name__ == "__main__":
    main()

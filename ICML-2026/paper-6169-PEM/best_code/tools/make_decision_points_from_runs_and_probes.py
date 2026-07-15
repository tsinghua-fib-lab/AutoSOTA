#!/usr/bin/env python3
"""
Create a `decision_points.csv` (and a small `summary.json`) by merging:
- per-run outcomes from one or more `runs.csv` files (external tasks),
- per-instance probe values from a `probe_values.csv`.

This lets us reuse the same decision-evidence tooling used on COCO:
- `tools/probe_threshold_sweep.py`
- `tools/probe_threshold_train_test.py`
- `tools/probe_threshold_kfold.py`

Expected join pattern (typical):
- key columns: `seed,batch_size` (for mini-batch tasks)
- label: which base optimizer is better on a post-hoc metric (lower is better)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from _project import repo_relpath


def parse_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def parse_list(spec: str) -> list[str]:
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def canonical_key_value(v: object) -> str:
    s = str(v).strip()
    if not s:
        return ""
    try:
        fv = float(s)
    except ValueError:
        return s
    if not np.isfinite(fv):
        return ""
    if float(fv).is_integer():
        return str(int(fv))
    return str(fv)


def key_of(row: dict, key_cols: list[str]) -> tuple[str, ...] | None:
    vals = []
    for c in key_cols:
        if c not in row:
            return None
        v = canonical_key_value(row.get(c, ""))
        if not v:
            return None
        vals.append(v)
    return tuple(vals)


def is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


def confusion(points: list[dict], *, probe_key: str, threshold: float) -> dict:
    usable = [p for p in points if str(p.get("label", "")).strip() in {"cma", "berw"}]
    pred = []
    for p in usable:
        v = str(p.get(probe_key, "")).strip()
        vv = float(v) if v else float("nan")
        if np.isfinite(vv) and float(vv) >= float(threshold):
            pred.append("berw")
        else:
            pred.append("cma")
    lab = [str(p["label"]).strip() for p in usable]
    n = int(len(lab))
    return {
        "n_non_ties": n,
        "accuracy": (float(sum(1 for a, b in zip(pred, lab) if a == b)) / float(n)) if n else float("nan"),
        "pred_cma_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "cma")),
        "pred_cma_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "berw")),
        "pred_berw_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "cma")),
        "pred_berw_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "berw")),
        "pred_berw_rate": (float(sum(1 for p in pred if p == "berw")) / float(n)) if n else float("nan"),
    }


def always_baseline(points: list[dict], *, pred: str) -> dict:
    usable = [p for p in points if str(p.get("label", "")).strip() in {"cma", "berw"}]
    lab = [str(p["label"]).strip() for p in usable]
    n = int(len(lab))
    if pred not in {"cma", "berw"}:
        raise ValueError("pred must be 'cma' or 'berw'")
    return {
        "n_non_ties": n,
        "accuracy": (float(sum(1 for y in lab if y == pred)) / float(n)) if n else float("nan"),
        "pred_cma_label_cma": int(sum(1 for y in lab if pred == "cma" and y == "cma")),
        "pred_cma_label_berw": int(sum(1 for y in lab if pred == "cma" and y == "berw")),
        "pred_berw_label_cma": int(sum(1 for y in lab if pred == "berw" and y == "cma")),
        "pred_berw_label_berw": int(sum(1 for y in lab if pred == "berw" and y == "berw")),
        "pred_berw_rate": 1.0 if pred == "berw" else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True, help="Comma-separated list of runs.csv paths.")
    parser.add_argument("--probe-values-csv", required=True, help="Probe values CSV (must include the join keys).")
    parser.add_argument("--key-cols", default="seed", help="Comma-separated join key columns (e.g., seed,batch_size).")
    parser.add_argument("--algo-cma", required=True, help="Algorithm name for the CMA baseline in runs.csv.")
    parser.add_argument("--algo-berw", required=True, help="Algorithm name for the BERW baseline in runs.csv.")
    parser.add_argument("--metric", required=True, help="Metric column in runs.csv to compare (lower is better by default).")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--tie-atol", type=float, default=0.0)
    parser.add_argument("--tie-rtol", type=float, default=0.0)
    parser.add_argument("--instance-col", default="seed", help="Which key column should be copied into 'instance'.")
    parser.add_argument("--misranking-threshold", type=float, default=0.12)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if bool(args.lower_is_better) and bool(args.higher_is_better):
        raise SystemExit("Choose at most one of --lower-is-better/--higher-is-better.")
    higher_better = bool(args.higher_is_better)

    key_cols = parse_list(str(args.key_cols))
    if not key_cols:
        raise SystemExit("--key-cols is empty.")

    probe_rows = parse_csv(os.path.abspath(str(args.probe_values_csv)))
    probe_by_key: dict[tuple[str, ...], dict] = {}
    for row in probe_rows:
        k = key_of(row, key_cols)
        if k is None:
            continue
        if k in probe_by_key:
            # Keep the first one; duplicates usually indicate a bug in upstream files.
            continue
        probe_by_key[k] = row

    runs_paths = [os.path.abspath(p) for p in parse_list(str(args.runs_csv))]
    runs_rows = []
    for p in runs_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing runs.csv: {p}")
        runs_rows.extend(parse_csv(p))

    algo_cma = str(args.algo_cma)
    algo_berw = str(args.algo_berw)
    metric = str(args.metric)

    metrics_by_key: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in runs_rows:
        algo = str(row.get("algorithm", "")).strip()
        if algo not in {algo_cma, algo_berw}:
            continue
        if metric not in row:
            raise SystemExit(f"Missing metric column '{metric}' in runs.csv rows.")
        k = key_of(row, key_cols)
        if k is None:
            continue
        metrics_by_key[k][algo] = float(row[metric])

    points: list[dict[str, object]] = []
    for k, vals in sorted(metrics_by_key.items()):
        if algo_cma not in vals or algo_berw not in vals:
            continue
        if k not in probe_by_key:
            continue
        best_cma = float(vals[algo_cma])
        best_berw = float(vals[algo_berw])

        if is_tie(best_cma, best_berw, atol=float(args.tie_atol), rtol=float(args.tie_rtol)):
            label = "tie"
        else:
            berw_better = (best_berw > best_cma) if higher_better else (best_berw < best_cma)
            label = "berw" if berw_better else "cma"

        probe_row = probe_by_key[k]
        out_row: dict[str, object] = {}
        for col, val in zip(key_cols, k):
            out_row[str(col)] = val
        out_row["instance"] = out_row.get(str(args.instance_col), out_row.get("seed", ""))
        out_row["best_f_cma"] = float(best_cma)
        out_row["best_f_berw"] = float(best_berw)
        out_row["label"] = str(label)

        # Copy probe values (keep all columns to avoid losing future signals).
        for col, val in probe_row.items():
            if str(col) in key_cols:
                continue
            if str(col) in out_row:
                continue
            out_row[str(col)] = val

        points.append(out_row)

    out_dir = os.path.abspath(str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    decision_points_path = os.path.join(out_dir, "decision_points.csv")
    with open(decision_points_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()) if points else [])
        if points:
            w.writeheader()
            for p in points:
                w.writerow(p)

    n_total = int(len(points))
    n_ties = int(sum(1 for p in points if str(p.get("label", "")).strip() == "tie"))
    n_missing_mis = int(sum(1 for p in points if not str(p.get("misranking_rd", "")).strip()))
    n_missing_var = int(sum(1 for p in points if not str(p.get("variance_rel_sd", "")).strip()))

    summary = {
        "setup": {
            "runs_csv": [repo_relpath(p) for p in runs_paths],
            "probe_values_csv": repo_relpath(str(args.probe_values_csv)),
            "key_cols": key_cols,
            "instance_col": str(args.instance_col),
            "algo_cma": algo_cma,
            "algo_berw": algo_berw,
            "metric": metric,
            "higher_is_better": bool(higher_better),
            "tie_tolerance": {"atol": float(args.tie_atol), "rtol": float(args.tie_rtol)},
            "probe_thresholds": {
                "misranking_rd": float(args.misranking_threshold),
                "variance_rel_sd": float(args.variance_threshold),
            },
        },
        "counts": {
            "n_total": n_total,
            "n_ties_outcome": n_ties,
            "n_missing_misranking_probe": n_missing_mis,
            "n_missing_variance_probe": n_missing_var,
        },
        "misranking_probe": confusion(points, probe_key="misranking_rd", threshold=float(args.misranking_threshold)),
        "variance_probe": confusion(points, probe_key="variance_rel_sd", threshold=float(args.variance_threshold)),
        "baselines": {
            "always_cma": always_baseline(points, pred="cma"),
            "always_berw": always_baseline(points, pred="berw"),
        },
        "outputs": {"decision_points_csv": repo_relpath(decision_points_path)},
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Wrote:", repo_relpath(decision_points_path))
    print("Wrote:", repo_relpath(summary_path))
    print("n_total:", n_total, "n_ties:", n_ties)
    print("misranking accuracy:", summary["misranking_probe"]["accuracy"])
    print("variance accuracy:", summary["variance_probe"]["accuracy"])


if __name__ == "__main__":
    main()

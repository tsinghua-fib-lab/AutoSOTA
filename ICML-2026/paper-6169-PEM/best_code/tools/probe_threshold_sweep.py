#!/usr/bin/env python3
"""
Sweep a probe threshold on an existing `decision_points.csv` (from probe_decision_accuracy.py).

This is for robustness checks:
- is the chosen threshold (e.g. 0.12 for misranking_rd) sitting on a broad plateau?
- how sensitive is the classification accuracy to the threshold?
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from _project import BASE_DIR, repo_relpath

def read_points(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def fr(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def sweep_thresholds(
    rows: list[dict],
    *,
    probe_value_key: str,
    thresholds: np.ndarray,
    label_key: str = "label",
) -> list[dict]:
    usable = [r for r in rows if str(r.get(label_key, "")).strip() in {"cma", "berw"}]
    out: list[dict] = []

    labels = [str(r[label_key]).strip() for r in usable]
    vals = [fr(r.get(probe_value_key, "")) for r in usable]

    for t in thresholds.tolist():
        pred = []
        for v in vals:
            if v is not None and float(v) >= float(t):
                pred.append("berw")
            else:
                pred.append("cma")

        n = int(len(labels))
        correct = int(sum(1 for p, y in zip(pred, labels) if p == y))
        pred_berw = int(sum(1 for p in pred if p == "berw"))

        conf = {
            "pred_cma_label_cma": int(sum(1 for p, y in zip(pred, labels) if p == "cma" and y == "cma")),
            "pred_cma_label_berw": int(sum(1 for p, y in zip(pred, labels) if p == "cma" and y == "berw")),
            "pred_berw_label_cma": int(sum(1 for p, y in zip(pred, labels) if p == "berw" and y == "cma")),
            "pred_berw_label_berw": int(sum(1 for p, y in zip(pred, labels) if p == "berw" and y == "berw")),
        }

        out.append(
            {
                "probe_value_key": str(probe_value_key),
                "threshold": float(t),
                "n_non_ties": n,
                "accuracy": (float(correct) / float(n)) if n else float("nan"),
                "correct": correct,
                "pred_berw_count": pred_berw,
                "pred_berw_rate": (float(pred_berw) / float(n)) if n else float("nan"),
                **conf,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-points", required=True, help="Path to decision_points.csv")
    parser.add_argument(
        "--probe",
        default="both",
        choices=["misranking", "variance", "both"],
        help="Which probe thresholds to sweep.",
    )
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--tstep", type=float, default=0.005)
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path (default: alongside decision_points.csv as threshold_sweep.csv).",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    in_path = os.path.abspath(str(args.decision_points))
    rows = read_points(in_path)

    tmin = float(args.tmin)
    tmax = float(args.tmax)
    tstep = float(args.tstep)
    if tstep <= 0:
        raise SystemExit("--tstep must be > 0")
    thresholds = np.arange(tmin, tmax + 0.5 * tstep, tstep, dtype=float)

    out_rows: list[dict] = []
    if args.probe in {"misranking", "both"}:
        out_rows.extend(sweep_thresholds(rows, probe_value_key="misranking_rd", thresholds=thresholds))
    if args.probe in {"variance", "both"}:
        out_rows.extend(sweep_thresholds(rows, probe_value_key="variance_rel_sd", thresholds=thresholds))

    out_path = str(args.output_csv).strip()
    if not out_path:
        out_path = os.path.join(os.path.dirname(in_path), "threshold_sweep.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if out_rows:
            w.writeheader()
            for r in out_rows:
                w.writerow(r)

    print("Wrote:", repo_relpath(out_path))

    # Print best thresholds for quick use in notes.
    for key in sorted({r["probe_value_key"] for r in out_rows}):
        subset = [r for r in out_rows if r["probe_value_key"] == key]
        subset_sorted = sorted(subset, key=lambda r: (-float(r["accuracy"]), float(r["threshold"])))
        if subset_sorted:
            best = subset_sorted[0]
            print(
                "best",
                key,
                "threshold",
                f"{float(best['threshold']):.6g}",
                "accuracy",
                f"{float(best['accuracy']):.6g}",
                "pred_berw_rate",
                f"{float(best['pred_berw_rate']):.6g}",
            )


if __name__ == "__main__":
    main()

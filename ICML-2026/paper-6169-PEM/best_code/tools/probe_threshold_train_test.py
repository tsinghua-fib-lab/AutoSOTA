#!/usr/bin/env python3
"""
Choose a probe threshold on a training split and evaluate on a test split.

This is a "no leakage" protocol for threshold selection:
- the probe values are computed per (function, instance),
- labels are derived from performance comparisons (CMA vs BERW) on the same instances,
- threshold is selected ONLY on train instances, then frozen and evaluated on test instances.

Input: `decision_points.csv` from `tools/probe_decision_accuracy.py`.
Output: JSON summary + optional per-threshold CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from _project import repo_relpath


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


def fr(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def read_points(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def classify(value: float | None, *, threshold: float) -> str:
    if value is not None and float(value) >= float(threshold):
        return "berw"
    return "cma"


def confusion(points: list[dict], *, probe_key: str, threshold: float) -> dict:
    usable = [p for p in points if str(p.get("label", "")).strip() in {"cma", "berw"}]
    pred = [classify(fr(p.get(probe_key, "")), threshold=float(threshold)) for p in usable]
    lab = [str(p["label"]).strip() for p in usable]
    return {
        "n": int(len(usable)),
        "accuracy": (float(sum(1 for a, b in zip(pred, lab) if a == b)) / float(len(usable))) if usable else float("nan"),
        "pred_cma_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "cma")),
        "pred_cma_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "berw")),
        "pred_berw_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "cma")),
        "pred_berw_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "berw")),
        "pred_berw_rate": (float(sum(1 for p in pred if p == "berw")) / float(len(usable))) if usable else float("nan"),
    }


def regrets_with_transform(
    points: list[dict],
    *,
    probe_key: str,
    threshold: float,
    loss: str,
    eps: float,
) -> dict:
    usable = [p for p in points if str(p.get("label", "")).strip() in {"cma", "berw"}]
    regs = []
    for p in usable:
        v = fr(p.get(probe_key, ""))
        pred = classify(v, threshold=float(threshold))
        bf_cma = float(p["best_f_cma"])
        bf_berw = float(p["best_f_berw"])
        best = min(bf_cma, bf_berw)
        chosen = bf_berw if pred == "berw" else bf_cma
        if loss == "raw":
            regs.append(float(chosen - best))
        else:
            # COCO noise-free best_f is (best noise-free fitness - Fopt) and is non-negative.
            c = float(chosen) + float(eps)
            b = float(best) + float(eps)
            if loss == "log10":
                regs.append(float(np.log10(c) - np.log10(b)))
            elif loss == "log":
                regs.append(float(np.log(c) - np.log(b)))
            else:
                raise ValueError(f"Unknown loss: {loss}")
    arr = np.asarray(regs, dtype=float)
    if arr.size <= 0:
        return {"mean": float("nan"), "median": float("nan"), "q90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-points", required=True)
    parser.add_argument("--probe-key", default="misranking_rd", choices=["misranking_rd", "variance_rel_sd"])
    parser.add_argument("--train-instances", default="1-2")
    parser.add_argument("--test-instances", default="3-5")
    parser.add_argument(
        "--loss",
        default="raw",
        choices=["log10", "log", "raw"],
        help=(
            "Regret/loss scale used for threshold selection and reporting. "
            "'log10' is recommended for COCO-style, scale-robust comparisons, "
            "but 'raw' is kept as default for backward-compatible evidence snapshots."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon used in log-loss regrets.")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.3)
    parser.add_argument("--tstep", type=float, default=0.005)
    parser.add_argument(
        "--selection",
        default="accuracy_then_rate",
        choices=[
            "accuracy_then_rate",
            "accuracy_then_threshold",
            "regret_mean_then_threshold",
            "regret_median_then_threshold",
            "regret_q90_then_threshold",
        ],
        help=(
            "How to select the threshold on the train split.\n"
            "- accuracy_then_rate: maximize train accuracy, tie-break by |pred_berw_rate-label_berw_rate|, then smaller threshold.\n"
            "- accuracy_then_threshold: maximize train accuracy, tie-break by smaller threshold.\n"
            "- regret_*_then_threshold: minimize train regret (mean/median/q90), tie-break by smaller threshold.\n"
            "Regret is defined relative to the best of the two algorithms (CMA vs BERW) on each (function,instance)."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON path (default: alongside decision_points.csv as train_test_threshold_<probe>.json).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional per-threshold CSV (default: alongside decision_points.csv as train_test_threshold_sweep_<probe>.csv).",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(str(args.decision_points))
    rows = read_points(in_path)

    train_set = set(parse_int_list(args.train_instances))
    test_set = set(parse_int_list(args.test_instances))
    if not train_set or not test_set:
        raise SystemExit("Empty train/test instance set.")
    if train_set.intersection(test_set):
        raise SystemExit("Train/test instances overlap.")

    train_points = [r for r in rows if int(float(r["instance"])) in train_set]
    test_points = [r for r in rows if int(float(r["instance"])) in test_set]

    train_labels = [r for r in train_points if str(r.get("label", "")).strip() in {"cma", "berw"}]
    label_berw_rate = (
        float(sum(1 for r in train_labels if str(r["label"]).strip() == "berw")) / float(len(train_labels))
        if train_labels
        else float("nan")
    )

    tmin = float(args.tmin)
    tmax = float(args.tmax)
    tstep = float(args.tstep)
    if tstep <= 0:
        raise SystemExit("--tstep must be > 0")
    thresholds = np.arange(tmin, tmax + 0.5 * tstep, tstep, dtype=float)

    sweep_rows = []
    for t in thresholds.tolist():
        c_tr = confusion(train_points, probe_key=str(args.probe_key), threshold=float(t))
        r_tr = regrets_with_transform(
            train_points,
            probe_key=str(args.probe_key),
            threshold=float(t),
            loss=str(args.loss),
            eps=float(args.eps),
        )
        c_te = confusion(test_points, probe_key=str(args.probe_key), threshold=float(t))
        r_te = regrets_with_transform(
            test_points,
            probe_key=str(args.probe_key),
            threshold=float(t),
            loss=str(args.loss),
            eps=float(args.eps),
        )
        sweep_rows.append(
            {
                "probe_key": str(args.probe_key),
                "threshold": float(t),
                "train_n": int(c_tr["n"]),
                "train_accuracy": float(c_tr["accuracy"]),
                "train_pred_berw_rate": float(c_tr["pred_berw_rate"]),
                "train_regret_mean": float(r_tr["mean"]),
                "train_regret_median": float(r_tr["median"]),
                "train_regret_q90": float(r_tr["q90"]),
                "test_n": int(c_te["n"]),
                "test_accuracy": float(c_te["accuracy"]),
                "test_pred_berw_rate": float(c_te["pred_berw_rate"]),
                "test_regret_mean": float(r_te["mean"]),
                "test_regret_median": float(r_te["median"]),
                "test_regret_q90": float(r_te["q90"]),
            }
        )

    sel = str(args.selection)
    if sel == "accuracy_then_threshold":
        sweep_sorted = sorted(sweep_rows, key=lambda r: (-float(r["train_accuracy"]), float(r["threshold"])))
        selection_criterion = "maximize train_accuracy, tie-break by smaller threshold"
    elif sel == "accuracy_then_rate":
        sweep_sorted = sorted(
            sweep_rows,
            key=lambda r: (
                -float(r["train_accuracy"]),
                abs(float(r["train_pred_berw_rate"]) - float(label_berw_rate)) if np.isfinite(label_berw_rate) else 0.0,
                float(r["threshold"]),
            ),
        )
        selection_criterion = "maximize train_accuracy, tie-break by |pred_berw_rate-label_berw_rate| then smaller threshold"
    elif sel == "regret_median_then_threshold":
        sweep_sorted = sorted(sweep_rows, key=lambda r: (float(r["train_regret_median"]), float(r["threshold"])))
        selection_criterion = "minimize train_regret_median, tie-break by smaller threshold"
    elif sel == "regret_q90_then_threshold":
        sweep_sorted = sorted(sweep_rows, key=lambda r: (float(r["train_regret_q90"]), float(r["threshold"])))
        selection_criterion = "minimize train_regret_q90, tie-break by smaller threshold"
    else:  # regret_mean_then_threshold
        sweep_sorted = sorted(sweep_rows, key=lambda r: (float(r["train_regret_mean"]), float(r["threshold"])))
        selection_criterion = "minimize train_regret_mean, tie-break by smaller threshold"
    best = sweep_sorted[0] if sweep_sorted else None
    if best is None:
        raise SystemExit("No thresholds evaluated.")

    best_t = float(best["threshold"])
    out = {
        "input": repo_relpath(in_path),
        "split": {
            "train_instances": sorted(train_set),
            "test_instances": sorted(test_set),
        },
        "probe_key": str(args.probe_key),
        "loss": {"name": str(args.loss), "eps": float(args.eps)},
        "selected_threshold": best_t,
        "selection": {
            "criterion": selection_criterion,
            "train_label_berw_rate": float(label_berw_rate),
        },
        "train": {
            "confusion": confusion(train_points, probe_key=str(args.probe_key), threshold=best_t),
            "regret": regrets_with_transform(
                train_points,
                probe_key=str(args.probe_key),
                threshold=best_t,
                loss=str(args.loss),
                eps=float(args.eps),
            ),
        },
        "test": {
            "confusion": confusion(test_points, probe_key=str(args.probe_key), threshold=best_t),
            "regret": regrets_with_transform(
                test_points,
                probe_key=str(args.probe_key),
                threshold=best_t,
                loss=str(args.loss),
                eps=float(args.eps),
            ),
        },
    }

    out_json = str(args.output_json).strip()
    if not out_json:
        out_json = os.path.join(os.path.dirname(in_path), f"train_test_threshold_{str(args.probe_key)}.json")
    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    out_csv = str(args.output_csv).strip()
    if not out_csv:
        out_csv = os.path.join(
            os.path.dirname(in_path),
            f"train_test_threshold_sweep_{str(args.probe_key)}.csv",
        )
    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()) if sweep_rows else [])
        if sweep_rows:
            w.writeheader()
            for r in sweep_rows:
                w.writerow(r)

    print("Wrote:", repo_relpath(out_json))
    print("Wrote:", repo_relpath(out_csv))
    print("selected_threshold:", best_t)
    print("train_accuracy:", out["train"]["confusion"]["accuracy"], "test_accuracy:", out["test"]["confusion"]["accuracy"])
    print("test_regret_mean:", out["test"]["regret"]["mean"], "test_regret_median:", out["test"]["regret"]["median"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
K-fold cross-validation for probe-threshold policies on `decision_points.csv`.

This complements `tools/probe_threshold_train_test.py`:
- train/test split by instances is good, but stability across splits is also useful;
- k-fold CV over instance groups gives a compact robustness check and a less brittle threshold estimate.

Design choices:
- Default grouping key is `instance` (so each fold tests on unseen COCO instances for every function).
- Threshold candidates are data-driven (unique probe values on the train split + two sentinels),
  so policies like "always CMA" (threshold > max) are always feasible without hand-picked `tmax`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

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


def is_usable(row: dict) -> bool:
    return str(row.get("label", "")).strip() in {"cma", "berw"}


def confusion(points: list[dict], *, probe_key: str, threshold: float) -> dict:
    usable = [p for p in points if is_usable(p)]
    pred = [classify(fr(p.get(probe_key, "")), threshold=float(threshold)) for p in usable]
    lab = [str(p["label"]).strip() for p in usable]
    n = int(len(lab))
    return {
        "n": n,
        "accuracy": (float(sum(1 for a, b in zip(pred, lab) if a == b)) / float(n)) if n else float("nan"),
        "pred_cma_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "cma")),
        "pred_cma_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "cma" and y == "berw")),
        "pred_berw_label_cma": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "cma")),
        "pred_berw_label_berw": int(sum(1 for p, y in zip(pred, lab) if p == "berw" and y == "berw")),
        "pred_berw_rate": (float(sum(1 for p in pred if p == "berw")) / float(n)) if n else float("nan"),
    }


def regrets(
    points: list[dict],
    *,
    probe_key: str,
    threshold: float,
    loss: str,
    eps: float,
) -> dict:
    usable = [p for p in points if is_usable(p)]
    regs = []
    for p in usable:
        pred = classify(fr(p.get(probe_key, "")), threshold=float(threshold))
        bf_cma = float(p["best_f_cma"])
        bf_berw = float(p["best_f_berw"])
        best = min(bf_cma, bf_berw)
        chosen = bf_berw if pred == "berw" else bf_cma
        if loss == "raw":
            regs.append(float(chosen - best))
        else:
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


def candidate_thresholds(points: list[dict], *, probe_key: str) -> list[float]:
    vals = [fr(p.get(probe_key, "")) for p in points if is_usable(p)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return [0.0]
    uniq = sorted(set(float(v) for v in vals))
    vmin = float(min(uniq))
    vmax = float(max(uniq))
    tiny = 1e-12
    # Include two sentinels so that "always berw" and "always cma" are always feasible.
    return sorted(set([vmin - tiny] + uniq + [vmax + tiny]))


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_groups: list[int]
    test_groups: list[int]


def make_folds(groups: list[int], *, k: int) -> list[FoldSpec]:
    k = int(max(2, k))
    groups = sorted(set(int(g) for g in groups))
    if len(groups) < k:
        raise SystemExit(f"Not enough groups ({len(groups)}) for k={k}.")
    buckets: list[list[int]] = [[] for _ in range(k)]
    for i, g in enumerate(groups):
        buckets[i % k].append(int(g))
    folds: list[FoldSpec] = []
    all_groups = set(groups)
    for fold_idx in range(k):
        test = buckets[fold_idx]
        train = sorted(all_groups.difference(test))
        folds.append(FoldSpec(fold=fold_idx, train_groups=train, test_groups=sorted(test)))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-points", required=True)
    parser.add_argument("--probe-key", default="misranking_rd", choices=["misranking_rd", "variance_rel_sd"])
    parser.add_argument("--group-by", default="instance", choices=["instance", "function", "function_index"])
    parser.add_argument("--k", type=int, default=5, help="Number of folds (grouped by --group-by).")
    parser.add_argument("--loss", default="log10", choices=["log10", "log", "raw"])
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--selection",
        default="regret_mean_then_threshold",
        choices=[
            "accuracy_then_rate",
            "accuracy_then_threshold",
            "regret_mean_then_threshold",
            "regret_median_then_threshold",
            "regret_q90_then_threshold",
        ],
    )
    parser.add_argument("--fixed-threshold", type=float, default=float("nan"), help="Optional fixed threshold to report.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    in_path = os.path.abspath(str(args.decision_points))
    rows = read_points(in_path)

    group_by = str(args.group_by)
    groups = sorted({int(float(r[group_by])) for r in rows if r.get(group_by, "").strip()})
    folds = make_folds(groups, k=int(args.k))

    probe_key = str(args.probe_key)
    loss = str(args.loss)
    eps = float(args.eps)
    selection = str(args.selection)

    per_fold = []
    all_test_points: list[dict] = []

    for spec in folds:
        train_set = set(int(g) for g in spec.train_groups)
        test_set = set(int(g) for g in spec.test_groups)

        train_points = [r for r in rows if int(float(r[group_by])) in train_set]
        test_points = [r for r in rows if int(float(r[group_by])) in test_set]

        # Train label rate for accuracy_then_rate.
        train_labels = [r for r in train_points if is_usable(r)]
        label_berw_rate = (
            float(sum(1 for r in train_labels if str(r["label"]).strip() == "berw")) / float(len(train_labels))
            if train_labels
            else float("nan")
        )

        thrs = candidate_thresholds(train_points, probe_key=probe_key)
        sweep = []
        for t in thrs:
            c_tr = confusion(train_points, probe_key=probe_key, threshold=float(t))
            r_tr = regrets(train_points, probe_key=probe_key, threshold=float(t), loss=loss, eps=eps)
            sweep.append(
                {
                    "threshold": float(t),
                    "train_accuracy": float(c_tr["accuracy"]),
                    "train_pred_berw_rate": float(c_tr["pred_berw_rate"]),
                    "train_regret_mean": float(r_tr["mean"]),
                    "train_regret_median": float(r_tr["median"]),
                    "train_regret_q90": float(r_tr["q90"]),
                }
            )

        if selection == "accuracy_then_threshold":
            sweep_sorted = sorted(sweep, key=lambda r: (-float(r["train_accuracy"]), float(r["threshold"])))
            selection_criterion = "maximize train_accuracy, tie-break by smaller threshold"
        elif selection == "accuracy_then_rate":
            sweep_sorted = sorted(
                sweep,
                key=lambda r: (
                    -float(r["train_accuracy"]),
                    abs(float(r["train_pred_berw_rate"]) - float(label_berw_rate)) if np.isfinite(label_berw_rate) else 0.0,
                    float(r["threshold"]),
                ),
            )
            selection_criterion = "maximize train_accuracy, tie-break by |pred_berw_rate-label_berw_rate| then smaller threshold"
        elif selection == "regret_median_then_threshold":
            sweep_sorted = sorted(sweep, key=lambda r: (float(r["train_regret_median"]), float(r["threshold"])))
            selection_criterion = "minimize train_regret_median, tie-break by smaller threshold"
        elif selection == "regret_q90_then_threshold":
            sweep_sorted = sorted(sweep, key=lambda r: (float(r["train_regret_q90"]), float(r["threshold"])))
            selection_criterion = "minimize train_regret_q90, tie-break by smaller threshold"
        else:  # regret_mean_then_threshold
            sweep_sorted = sorted(sweep, key=lambda r: (float(r["train_regret_mean"]), float(r["threshold"])))
            selection_criterion = "minimize train_regret_mean, tie-break by smaller threshold"

        best = sweep_sorted[0] if sweep_sorted else None
        if best is None:
            raise SystemExit("No thresholds evaluated.")

        best_t = float(best["threshold"])
        c_te = confusion(test_points, probe_key=probe_key, threshold=best_t)
        r_te = regrets(test_points, probe_key=probe_key, threshold=best_t, loss=loss, eps=eps)

        per_fold.append(
            {
                "fold": int(spec.fold),
                "group_by": group_by,
                "train_groups": list(spec.train_groups),
                "test_groups": list(spec.test_groups),
                "selected_threshold": float(best_t),
                "selection_criterion": selection_criterion,
                "train_label_berw_rate": float(label_berw_rate),
                "train_accuracy": float(best["train_accuracy"]),
                "train_regret_mean": float(best["train_regret_mean"]),
                "test_accuracy": float(c_te["accuracy"]),
                "test_regret_mean": float(r_te["mean"]),
                "test_regret_median": float(r_te["median"]),
                "test_regret_q90": float(r_te["q90"]),
            }
        )
        all_test_points.extend(test_points)

    # Aggregate metrics by pooling all test points (each group appears as test exactly once).
    # This matches "CV-out-of-sample" evaluation across all points.
    thresholds = np.asarray([r["selected_threshold"] for r in per_fold], dtype=float)
    agg = {
        "thresholds": {
            "mean": float(np.mean(thresholds)),
            "std": float(np.std(thresholds)),
            "median": float(np.median(thresholds)),
            "min": float(np.min(thresholds)),
            "max": float(np.max(thresholds)),
        }
    }

    # Recompute pooled confusion/regret by applying the fold-specific threshold.
    # We do this by re-deriving per-row fold membership using the same round-robin fold assignment.
    group_to_fold = {}
    for spec in folds:
        for g in spec.test_groups:
            group_to_fold[int(g)] = int(spec.fold)
    fold_to_threshold = {int(r["fold"]): float(r["selected_threshold"]) for r in per_fold}

    usable = [r for r in rows if is_usable(r)]
    preds = []
    labs = []
    regs = []
    for row in usable:
        g = int(float(row[group_by]))
        fold = group_to_fold.get(int(g), None)
        if fold is None:
            continue
        t = fold_to_threshold[int(fold)]
        pred = classify(fr(row.get(probe_key, "")), threshold=float(t))
        lab = str(row["label"]).strip()
        preds.append(pred)
        labs.append(lab)

        bf_cma = float(row["best_f_cma"])
        bf_berw = float(row["best_f_berw"])
        best = min(bf_cma, bf_berw)
        chosen = bf_berw if pred == "berw" else bf_cma
        if loss == "raw":
            regs.append(float(chosen - best))
        else:
            c = float(chosen) + float(eps)
            b = float(best) + float(eps)
            regs.append(float(np.log10(c) - np.log10(b)) if loss == "log10" else float(np.log(c) - np.log(b)))

    preds_arr = np.asarray(preds, dtype=object)
    labs_arr = np.asarray(labs, dtype=object)
    regs_arr = np.asarray(regs, dtype=float)

    def _confusion_from_arrays(p: np.ndarray, y: np.ndarray) -> dict:
        n = int(y.size)
        correct = int(np.sum(p == y))
        return {
            "n_non_ties": n,
            "accuracy": (float(correct) / float(n)) if n else float("nan"),
            "pred_cma_label_cma": int(np.sum((p == "cma") & (y == "cma"))),
            "pred_cma_label_berw": int(np.sum((p == "cma") & (y == "berw"))),
            "pred_berw_label_cma": int(np.sum((p == "berw") & (y == "cma"))),
            "pred_berw_label_berw": int(np.sum((p == "berw") & (y == "berw"))),
            "pred_berw_rate": (float(np.mean(p == "berw")) if n else float("nan")),
        }

    agg["cv"] = {
        "confusion": _confusion_from_arrays(preds_arr, labs_arr),
        "regret": {
            "mean": float(np.mean(regs_arr)) if regs_arr.size else float("nan"),
            "median": float(np.median(regs_arr)) if regs_arr.size else float("nan"),
            "q90": float(np.quantile(regs_arr, 0.9)) if regs_arr.size else float("nan"),
        },
    }

    # Baselines on the same set of usable points.
    def _baseline(pred: str) -> dict:
        p = np.asarray([pred] * labs_arr.size, dtype=object)
        y = labs_arr
        c = _confusion_from_arrays(p, y)
        # regret
        regs2 = []
        for row in usable:
            bf_cma = float(row["best_f_cma"])
            bf_berw = float(row["best_f_berw"])
            best = min(bf_cma, bf_berw)
            chosen = bf_berw if pred == "berw" else bf_cma
            if loss == "raw":
                regs2.append(float(chosen - best))
            else:
                c_ = float(chosen) + float(eps)
                b_ = float(best) + float(eps)
                regs2.append(float(np.log10(c_) - np.log10(b_)) if loss == "log10" else float(np.log(c_) - np.log(b_)))
        regs2 = np.asarray(regs2, dtype=float)
        return {
            "confusion": c,
            "regret": {
                "mean": float(np.mean(regs2)) if regs2.size else float("nan"),
                "median": float(np.median(regs2)) if regs2.size else float("nan"),
                "q90": float(np.quantile(regs2, 0.9)) if regs2.size else float("nan"),
            },
        }

    agg["baselines"] = {"always_cma": _baseline("cma"), "always_berw": _baseline("berw")}

    fixed = float(args.fixed_threshold)
    if np.isfinite(fixed):
        agg["fixed_threshold"] = {
            "threshold": float(fixed),
            "confusion": confusion(rows, probe_key=probe_key, threshold=float(fixed)),
            "regret": regrets(rows, probe_key=probe_key, threshold=float(fixed), loss=loss, eps=eps),
        }

    out = {
        "input": repo_relpath(in_path),
        "probe_key": probe_key,
        "group_by": group_by,
        "k": int(args.k),
        "loss": {"name": loss, "eps": float(eps)},
        "selection": selection,
        "folds": [spec.__dict__ for spec in folds],
        "per_fold": per_fold,
        "aggregate": agg,
    }

    out_json = str(args.output_json).strip()
    if not out_json:
        tag = f"k{int(args.k)}_{probe_key}_{loss}_{selection}"
        out_json = os.path.join(os.path.dirname(in_path), f"threshold_kfold_{tag}.json")
    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    out_csv = str(args.output_csv).strip()
    if out_csv:
        out_csv = os.path.abspath(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()) if per_fold else [])
            if per_fold:
                w.writeheader()
                for r in per_fold:
                    w.writerow(r)

    print("Wrote:", repo_relpath(out_json))
    if out_csv:
        print("Wrote:", repo_relpath(out_csv))
    print("cv_accuracy:", out["aggregate"]["cv"]["confusion"]["accuracy"])
    print("cv_regret_mean:", out["aggregate"]["cv"]["regret"]["mean"])


if __name__ == "__main__":
    main()

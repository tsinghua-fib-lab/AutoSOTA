#!/usr/bin/env python3
"""
Evaluate *zero-tuning* probe-threshold transfer across datasets/tasks/budgets.

Given:
  - one or more "source" thresholds (e.g., learned on COCO train split),
  - multiple target `decision_points.csv` files (same schema as produced by tools),
this script reports how well a fixed threshold transfers without re-tuning.

Outputs:
  - CSV summary with per-(target,method) confusion + regret stats.

Notes:
  - We intentionally keep this lightweight:
    a fixed threshold learned on one setting should generalize if the probe is meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

from _project import repo_relpath


def fr(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def read_csv_dicts(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_selected_threshold(path: str) -> float:
    with open(path) as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "selected_threshold" not in obj:
        raise ValueError(f"Not a train_test threshold JSON: {path}")
    return float(obj["selected_threshold"])


def classify(v: float | None, *, threshold: float) -> str:
    return "berw" if (v is not None and float(v) >= float(threshold)) else "cma"


def _log_regret(chosen: float, best: float, *, eps: float) -> float:
    c = float(chosen) + float(eps)
    b = float(best) + float(eps)
    if c <= 0.0 or b <= 0.0:
        return float("nan")
    return float(np.log10(c) - np.log10(b))


def evaluate_points(
    points: list[dict],
    *,
    probe_key: str,
    threshold: float | None,
    force_pred: str | None,
    loss: str,
    eps: float,
) -> dict[str, float]:
    usable = [p for p in points if str(p.get("label", "")).strip() in {"cma", "berw"}]
    if not usable:
        return {
            "n": 0,
            "label_berw_rate": float("nan"),
            "accuracy": float("nan"),
            "pred_berw_rate": float("nan"),
            "regret_mean": float("nan"),
            "regret_median": float("nan"),
            "regret_q90": float("nan"),
        }

    labels = [str(p["label"]).strip() for p in usable]
    label_berw_rate = float(sum(1 for y in labels if y == "berw")) / float(len(labels))

    preds: list[str] = []
    regs: list[float] = []
    for p in usable:
        if force_pred is not None:
            pred = str(force_pred)
        else:
            pred = classify(fr(p.get(probe_key, "")), threshold=float(threshold))
        preds.append(pred)

        bf_cma = float(p["best_f_cma"])
        bf_berw = float(p["best_f_berw"])
        best = min(bf_cma, bf_berw)
        chosen = bf_berw if pred == "berw" else bf_cma
        if loss == "raw":
            regs.append(float(chosen - best))
        elif loss == "log10":
            regs.append(_log_regret(chosen, best, eps=float(eps)))
        elif loss == "rel":
            denom = abs(float(best)) + float(eps)
            regs.append(float(chosen - best) / float(denom))
        else:
            raise ValueError(f"Unknown loss: {loss}")

    pred_berw_rate = float(sum(1 for p in preds if p == "berw")) / float(len(preds))
    acc = float(sum(1 for p, y in zip(preds, labels) if p == y)) / float(len(labels))
    arr = np.asarray([v for v in regs if np.isfinite(v)], dtype=float)
    if arr.size <= 0:
        r_mean = r_median = r_q90 = float("nan")
    else:
        r_mean = float(np.mean(arr))
        r_median = float(np.median(arr))
        r_q90 = float(np.quantile(arr, 0.9))
    return {
        "n": int(len(labels)),
        "label_berw_rate": float(label_berw_rate),
        "accuracy": float(acc),
        "pred_berw_rate": float(pred_berw_rate),
        "regret_mean": float(r_mean),
        "regret_median": float(r_median),
        "regret_q90": float(r_q90),
    }


@dataclass(frozen=True)
class ThresholdMethod:
    name: str
    threshold: float | None  # None if force_pred is used
    force_pred: str | None


def maybe_load_target_tuned_threshold(decision_points: str, probe_key: str, *, loss: str) -> float | None:
    d = os.path.dirname(os.path.abspath(decision_points))
    suffix = ""
    if loss == "log10":
        suffix = "_log10_regret_mean"
    elif loss == "raw":
        suffix = ""
    elif loss == "rel":
        # no canonical file; leave None
        return None
    candidates = [
        os.path.join(d, f"train_test_threshold_{probe_key}{suffix}.json"),
        os.path.join(d, f"train_test_threshold_{probe_key}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return load_selected_threshold(p)
            except Exception:
                continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--probe-key", default="misranking_rd")
    parser.add_argument("--loss", default="log10", choices=["log10", "raw", "rel"])
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source threshold spec: name:/path/to/train_test_threshold_*.json",
    )
    parser.add_argument(
        "--fixed-threshold",
        action="append",
        default=[],
        help="Additional fixed threshold: name:value (e.g., fixed0p12:0.12)",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Target decision points: name:/path/to/decision_points.csv",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    methods: list[ThresholdMethod] = []

    for spec in list(args.source or []):
        if ":" not in spec:
            raise SystemExit(f"--source must be name:path, got: {spec}")
        name, path = spec.split(":", 1)
        t = load_selected_threshold(path)
        methods.append(ThresholdMethod(name=str(name), threshold=float(t), force_pred=None))

    for spec in list(args.fixed_threshold or []):
        if ":" not in spec:
            raise SystemExit(f"--fixed-threshold must be name:value, got: {spec}")
        name, val = spec.split(":", 1)
        methods.append(ThresholdMethod(name=str(name), threshold=float(val), force_pred=None))

    # Baselines (no probe).
    methods.extend(
        [
            ThresholdMethod(name="always_cma", threshold=None, force_pred="cma"),
            ThresholdMethod(name="always_berw", threshold=None, force_pred="berw"),
        ]
    )

    targets: list[tuple[str, str]] = []
    for spec in list(args.target or []):
        if ":" not in spec:
            raise SystemExit(f"--target must be name:path, got: {spec}")
        name, path = spec.split(":", 1)
        targets.append((str(name), str(path)))
    if not targets:
        raise SystemExit("No --target specified.")

    rows_out: list[dict[str, object]] = []
    for target_name, decision_points in targets:
        points = read_csv_dicts(decision_points)

        # Target-tuned threshold (for reference only; not used in transfer).
        tuned = maybe_load_target_tuned_threshold(
            decision_points,
            str(args.probe_key),
            loss=str(args.loss),
        )
        if tuned is not None:
            methods_with_tuned = methods + [ThresholdMethod(name="target_tuned", threshold=float(tuned), force_pred=None)]
        else:
            methods_with_tuned = list(methods)

        for m in methods_with_tuned:
            stats = evaluate_points(
                points,
                probe_key=str(args.probe_key),
                threshold=m.threshold,
                force_pred=m.force_pred,
                loss=str(args.loss),
                eps=float(args.eps),
            )
            rows_out.append(
                {
                    "target": str(target_name),
                    "decision_points": repo_relpath(decision_points),
                    "probe_key": str(args.probe_key),
                    "loss": str(args.loss),
                    "method": str(m.name),
                    "threshold": "" if m.threshold is None else float(m.threshold),
                    **stats,
                }
            )

    out_csv = os.path.join(out_dir, "transfer_summary.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows_out[0].keys()) if rows_out else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if rows_out:
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    # Also emit a compact markdown table for quick copy/paste into docs.
    md_path = os.path.join(out_dir, "transfer_summary.md")
    with open(md_path, "w") as f:
        f.write("# Probe threshold transfer summary (auto-generated)\n\n")
        f.write(f"- probe_key: `{args.probe_key}`\n")
        f.write(f"- loss: `{args.loss}`\n\n")
        f.write("| target | method | threshold | n | acc | regret_mean |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows_out:
            f.write(
                "| {target} | {method} | {threshold} | {n} | {accuracy:.3f} | {regret_mean:.4f} |\n".format(
                    target=str(r["target"]),
                    method=str(r["method"]),
                    threshold=str(r["threshold"]),
                    n=int(r["n"]),
                    accuracy=float(r["accuracy"]) if np.isfinite(float(r["accuracy"])) else float("nan"),
                    regret_mean=float(r["regret_mean"]) if np.isfinite(float(r["regret_mean"])) else float("nan"),
                )
            )

    print("Wrote:", repo_relpath(out_csv))
    print("Wrote:", repo_relpath(md_path))


if __name__ == "__main__":
    main()

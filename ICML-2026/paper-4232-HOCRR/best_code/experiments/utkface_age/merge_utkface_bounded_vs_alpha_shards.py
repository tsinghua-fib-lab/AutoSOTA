#!/usr/bin/env python3
"""Merge sharded UTKFace bounded-vs-alpha JSON outputs into one file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "fraction_positive": float(np.mean(arr > 0.0)),
    }


def summarize_abs_error(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mae": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "medae": float(np.median(arr)),
        "p90ae": float(np.percentile(arr, 90)),
        "maxae": float(np.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded UTKFace comparison outputs.")
    p.add_argument("--inputs", nargs="+", required=True, help="Shard JSON files.")
    p.add_argument("--output", required=True, help="Merged output JSON file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = [Path(x) for x in args.inputs]
    shard_data = []
    for p in shard_paths:
        with p.open("r", encoding="utf-8") as f:
            shard_data.append(json.load(f))

    base = shard_data[0]
    all_samples = []
    for d in shard_data:
        all_samples.extend(d.get("samples", []))

    all_samples.sort(key=lambda x: int(x.get("sample_global_idx", x.get("sample_local_idx", -1))))

    seen = set()
    deduped = []
    for s in all_samples:
        key = int(s.get("sample_global_idx", s.get("sample_local_idx", -1)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    clean_abs = [float(s["clean_abs_error"]) for s in deduped]
    smoothed_abs = [
        float(s["smoothed_abs_error_mean_over_trials"])
        for s in deduped
        if s.get("smoothed_abs_error_mean_over_trials") is not None
    ]
    bounded = [
        float(s["ecg_radius_mean_over_trials"])
        for s in deduped
        if s.get("ecg_radius_mean_over_trials") is not None
    ]
    alpha = [
        float(s["alpha_result"]["radius_alpha"])
        for s in deduped
        if s.get("alpha_result") is not None
    ]
    unbounded = [
        float(s["unbounded_radius_mean_over_trials"])
        for s in deduped
        if s.get("unbounded_radius_mean_over_trials") is not None
    ]

    summary = {
        "clean_abs_error": summarize_abs_error(clean_abs),
        "clean_mae": float(np.mean(clean_abs)),
        "clean_rmse": float(np.sqrt(np.mean(np.asarray(clean_abs) ** 2))),
        "bounded_ecg_radius": summarize(bounded) if bounded else None,
        "alpha_radius": summarize(alpha) if alpha else None,
        "unbounded_vg_radius": summarize(unbounded) if unbounded else None,
    }
    if smoothed_abs:
        smoothed_summary = summarize_abs_error(smoothed_abs)
        summary["smoothed_abs_error"] = smoothed_summary
        summary["smoothed_minus_clean_mae"] = float(
            smoothed_summary["mae"] - summary["clean_abs_error"]["mae"]
        )

    merged = {
        **base,
        "timestamp": datetime.now().isoformat(),
        "merged_from_shards": [str(p) for p in shard_paths],
        "samples": deduped,
        "summary": summary,
    }
    if "dataset" in merged and isinstance(merged["dataset"], dict):
        merged["dataset"]["n_points"] = int(len(deduped))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(shard_paths)} shards -> {out}")


if __name__ == "__main__":
    main()


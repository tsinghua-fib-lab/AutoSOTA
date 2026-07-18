#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, Tuple

import numpy as np


def load_meta(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_tensor(bin_path: str, meta: Dict) -> np.ndarray:
    expected = 1
    for d in meta["shape"]:
        expected *= int(d)
    arr = np.fromfile(bin_path, dtype=np.int64)
    if arr.size != expected:
        raise ValueError(f"{bin_path}: expected {expected} elements, got {arr.size}")
    return arr.reshape(meta["shape"])


def to_float(arr: np.ndarray, scale: int) -> np.ndarray:
    return arr.astype(np.float64) / float(1 << scale)


def compute_metrics(ref: np.ndarray, test: np.ndarray, scale: int) -> Dict[str, float]:
    ref_f = to_float(ref, scale)
    test_f = to_float(test, scale)
    diff = test_f - ref_f
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(math.sqrt(np.mean(ref_f * ref_f)) + 1e-12)
    rel_rmse = float(rmse / denom)

    if ref_f.ndim >= 2:
        ref_arg = np.argmax(ref_f, axis=-1)
        test_arg = np.argmax(test_f, axis=-1)
        top1 = float(np.mean(ref_arg == test_arg))
        if ref_f.shape[-1] >= 5:
            ref_top5 = np.argsort(ref_f, axis=-1)[..., -5:]
            test_top1 = test_arg[..., None]
            top5 = float(np.mean(np.any(ref_top5 == test_top1, axis=-1)))
        else:
            top5 = float("nan")
    else:
        top1 = float(np.argmax(ref_f) == np.argmax(test_f))
        top5 = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_rmse": rel_rmse,
        "top1": top1,
        "top5": top5,
    }


def parse_stats(dir_path: str) -> Dict[str, float]:
    stats = {}
    dealer = os.path.join(dir_path, "dealer.txt")
    evaluator = os.path.join(dir_path, "evaluator.txt")
    if os.path.exists(dealer):
        with open(dealer, "r") as f:
            txt = f.read()
        m = re.search(r"Total time=(\d+) us", txt)
        if m:
            stats["gen_us"] = float(m.group(1))
        m = re.search(r"Key size=(\d+) B", txt)
        if m:
            stats["key_bytes"] = float(m.group(1))
    if os.path.exists(evaluator):
        with open(evaluator, "r") as f:
            txt = f.read()
        m = re.search(r"Total time=(\d+) us", txt)
        if m:
            stats["eval_us"] = float(m.group(1))
        m = re.search(r"Total Comm=(\d+) B", txt)
        if m:
            stats["comm_bytes"] = float(m.group(1))
    return stats


def load_output(dir_path: str, prefix: str = "output") -> Tuple[np.ndarray, Dict]:
    bin_path = os.path.join(dir_path, f"{prefix}.bin")
    meta_path = os.path.join(dir_path, f"{prefix}_meta.json")
    meta = load_meta(meta_path)
    tensor = load_tensor(bin_path, meta)
    return tensor, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="Baseline sigma output dir")
    ap.add_argument("--sweep-dir", required=True, help="Directory containing sweep subdirs")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    base_out, base_meta = load_output(args.base_dir, "output")
    clear_meta = None
    clear_out = None
    clear_path = os.path.join(args.base_dir, "cleartext.bin")
    if os.path.exists(clear_path):
        clear_out, clear_meta = load_output(args.base_dir, "cleartext")

    rows = []
    for tag in sorted(os.listdir(args.sweep_dir)):
        run_dir = os.path.join(args.sweep_dir, tag)
        out_path = os.path.join(run_dir, "output.bin")
        if not os.path.exists(out_path):
            continue
        out, meta = load_output(run_dir, "output")
        if meta["scale"] != base_meta["scale"]:
            raise ValueError(f"scale mismatch for {tag}: {meta['scale']} vs {base_meta['scale']}")

        stats = parse_stats(run_dir)
        row = {
            "tag": tag,
            **stats,
        }

        sigma_metrics = compute_metrics(base_out, out, int(meta["scale"]))
        for k, v in sigma_metrics.items():
            row[f"sigma_{k}"] = v

        if clear_out is not None:
            clear_metrics = compute_metrics(clear_out, out, int(meta["scale"]))
            for k, v in clear_metrics.items():
                row[f"clear_{k}"] = v

        rows.append(row)

    if not rows:
        print("No runs found.", file=sys.stderr)
        return 1

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

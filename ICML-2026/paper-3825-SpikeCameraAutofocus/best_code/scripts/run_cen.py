#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from autofocus_cen import CENConfig, estimate_focus_from_npy_files
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from autofocus_cen import CENConfig, estimate_focus_from_npy_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CEN autofocus on a manifest JSON file.")
    parser.add_argument("--manifest", required=True, help="Dataset manifest JSON.")
    parser.add_argument("--output", default="cen_results.csv", help="Output CSV path.")
    parser.add_argument("--r2-min", type=float, default=0.05)
    parser.add_argument("--r2-max", type=float, default=0.30)
    parser.add_argument("--r2-step", type=float, default=0.01)
    return parser.parse_args()


def _r2_grid(r2_min: float, r2_max: float, r2_step: float) -> tuple[float, ...]:
    values = []
    value = r2_min
    while value <= r2_max + 1e-12:
        values.append(round(value, 4))
        value += r2_step
    return tuple(values)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    config = CENConfig(r2_list=_r2_grid(args.r2_min, args.r2_max, args.r2_step))
    rows = []

    for scene in manifest.get("scenes", []):
        name = str(scene["name"])
        dt = int(scene["dt"])
        spike_files = [base_dir / path for path in scene["spike_files"]]
        result = estimate_focus_from_npy_files(spike_files, dt=dt, config=config)

        row = {
            "scene": name,
            "dt": dt,
            "pred_block": int(result.focus_block),
            "r2": float(result.r2),
            "num_blocks": int(result.block_ids.size),
        }
        if "gt_block" in scene:
            row["gt_block"] = int(scene["gt_block"])
            row["abs_err"] = abs(row["pred_block"] - row["gt_block"])
        rows.append(row)

        gt_text = f" gt={row['gt_block']} err={row['abs_err']}" if "gt_block" in row else ""
        print(f"{name}: pred={row['pred_block']} r2={row['r2']:.4f}{gt_text}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "dt", "num_blocks", "gt_block", "pred_block", "abs_err", "r2"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np

DATASET_URL = "https://drive.google.com/file/d/1a0eDgwk-6tjjZQCigShkX_S3cgHybZHT/view?usp=drive_link"

try:
    from autofocus_cen import CENConfig
    from autofocus_cen import estimate_focus_from_npy_files
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from autofocus_cen import CENConfig
    from autofocus_cen import estimate_focus_from_npy_files


GT_IMAGE_MAP = {
    "simu01": "Im37",
    "simu02": "Im37",
    "simu03": "Im13",
}


def natural_key(path: str | Path) -> list[object]:
    name = Path(path).name
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def image_name_to_index(image_name: str) -> int:
    match = re.search(r"Im(\d+)", image_name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse image name: {image_name}")
    return int(match.group(1)) - 1


def list_scene_dirs(generated_root: Path, scene_dirs: list[str] | None) -> list[Path]:
    if scene_dirs:
        return [Path(path) for path in scene_dirs]
    return sorted([path for path in generated_root.glob("simu*") if path.is_dir()], key=natural_key)


def list_spike_files(scene_dir: Path) -> list[Path]:
    spikes_dir = scene_dir / "spikes_npy"
    files = sorted(glob.glob(str(spikes_dir / "Im*_spikes.npy")), key=natural_key)
    spike_files = [Path(path) for path in files]
    if not spike_files:
        raise FileNotFoundError(f"cannot find Im*_spikes.npy under {spikes_dir}")
    return spike_files


def load_focus_distances(scene_dir: Path, expected_len: int) -> list[float]:
    csv_path = scene_dir / "focus_distances.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"focus_distances.csv not found: {csv_path}")

    values: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["focus_distance"]))

    if len(values) != expected_len:
        raise ValueError(f"{csv_path} has {len(values)} focus distances, expected {expected_len}")
    return values


def evaluate_scene(scene_dir: Path, dt: int, config: CENConfig) -> dict[str, object]:
    scene = scene_dir.name
    gt_image = GT_IMAGE_MAP.get(scene)
    if gt_image is None:
        raise ValueError(f"no default GT image configured for {scene}")

    spike_files = list_spike_files(scene_dir)
    focus_distances = load_focus_distances(scene_dir, expected_len=len(spike_files))
    spikes_per_image = int(np.load(spike_files[0], mmap_mode="r").shape[0])

    gt_image_idx = image_name_to_index(gt_image)
    gt_focus = float(focus_distances[gt_image_idx])
    gt_block = int(round((gt_image_idx * spikes_per_image) / float(dt)))

    result = estimate_focus_from_npy_files(spike_files, dt=dt, config=config)
    pred_block = int(result.focus_block)
    pred_frame_center = pred_block * float(dt) + 0.5 * float(max(dt - 1, 0))
    pred_image_idx = int(round(pred_frame_center / float(spikes_per_image)))
    pred_image_idx = max(0, min(pred_image_idx, len(spike_files) - 1))
    pred_image = f"Im{pred_image_idx + 1:02d}"
    pred_focus = float(focus_distances[pred_image_idx])

    return {
        "scene": scene,
        "dt": dt,
        "gt_image": gt_image,
        "gt_block": gt_block,
        "gt_focus": gt_focus,
        "pred_block": pred_block,
        "pred_image": pred_image,
        "pred_focus": pred_focus,
        "abs_err": abs(pred_focus - gt_focus),
        "r2": float(result.r2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CEN autofocus on generated spike-camera data")
    parser.add_argument(
        "--scene_dirs",
        nargs="*",
        default=None,
        help="Optional scene directories. If omitted, read simu* scenes under --generated_root.",
    )
    parser.add_argument(
        "--generated_root",
        default="./simulate_moderate_light",
        help="Root directory of generated spikes.",
    )
    parser.add_argument("--dt", type=int, default=10, help="Spike frames accumulated into one block.")
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Optional output directory. Default: ./results_dt{dt}",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("dt must be positive")

    generated_root = Path(args.generated_root)
    if not generated_root.exists():
        raise FileNotFoundError(
            f"generated_root not found: {generated_root}\n"
            f"Download the public dataset from {DATASET_URL} and pass the extracted "
            "simulate_moderate_light or simulate_low_light directory via --generated_root."
        )

    scene_dirs = list_scene_dirs(generated_root, args.scene_dirs)
    save_dir = Path(args.save_dir or f"./results_dt{args.dt}")
    config = CENConfig()

    result_rows: list[dict[str, object]] = []

    print("========== SPIKE CAMERA AUTOFOCUS WITH CEN ==========")
    print(f"generated_root = {args.generated_root}")
    print(f"save_dir       = {save_dir}")

    for scene_dir in scene_dirs:
        scene_name = Path(scene_dir).resolve().name
        if scene_name not in GT_IMAGE_MAP:
            print(f"[warn] no default GT image for {scene_name}; skipped")
            continue

        try:
            row = evaluate_scene(scene_dir=scene_dir, dt=args.dt, config=config)
        except Exception as exc:
            print(f"[ERROR] scene={scene_name}, dt={args.dt}, error={exc}")
            row = {"scene": scene_name, "dt": args.dt, "error": str(exc)}

        result_rows.append(row)
        if "pred_block" in row:
            print(
                f"{scene_name}: pred_block={row['pred_block']} "
                f"pred_focus={row['pred_focus']:.8f} "
                f"gt_focus={row['gt_focus']:.8f} "
                f"abs_err={row['abs_err']:.8f} "
                f"r2={row['r2']:.4f}"
            )

    results_csv = save_dir / "results.csv"
    write_csv(results_csv, result_rows)
    print(f"[save] {results_csv}")
    print("========== DONE ==========")


if __name__ == "__main__":
    main()

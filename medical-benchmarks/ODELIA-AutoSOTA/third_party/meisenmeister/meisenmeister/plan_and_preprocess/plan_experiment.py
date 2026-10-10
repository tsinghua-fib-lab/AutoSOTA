import json
from math import ceil
from pathlib import Path

import numpy as np

from meisenmeister.utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)

DEFAULT_MARGIN_MM = [10.0, 10.0, 10.0]
DEFAULT_TARGET_SHAPE_PERCENTILE = 95.0


def _load_dataset_fingerprint(fingerprint_path: Path) -> dict:
    if not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset_fingerprint.json in {fingerprint_path.parent}. "
            "Run mm_extract_dataset_fingerprint first."
        )

    with fingerprint_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _compute_target_shape(
    spacings: list[list[float]],
    shapes_after_crop: list[list[int]],
    target_spacing: list[float],
    margin_mm: list[float],
    percentile: float,
) -> list[int]:
    resampled_shapes: list[list[int]] = []
    for spacing, shape in zip(spacings, shapes_after_crop, strict=True):
        physical_size = [
            float(shape_axis) * float(spacing_axis) + 2.0 * float(margin_axis)
            for shape_axis, spacing_axis, margin_axis in zip(
                shape, spacing, margin_mm, strict=True
            )
        ]
        resampled_shape = [
            int(ceil(size_mm / spacing_axis))
            for size_mm, spacing_axis in zip(physical_size, target_spacing, strict=True)
        ]
        resampled_shapes.append(resampled_shape)

    percentile_shape = np.percentile(
        np.asarray(resampled_shapes, dtype=np.float64), percentile, axis=0
    )
    return [int(ceil(axis_value)) for axis_value in percentile_shape]


@require_global_paths_set
def plan_experiment(d: int) -> dict:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]

    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    preprocessed_dataset_dir = mm_preprocessed / dataset_dir.name
    fingerprint_path = preprocessed_dataset_dir / "dataset_fingerprint.json"
    dataset_fingerprint = _load_dataset_fingerprint(fingerprint_path)

    target_spacing = [float(i) for i in dataset_fingerprint["median_spacing"]]
    target_shape = _compute_target_shape(
        spacings=dataset_fingerprint["spacings"],
        shapes_after_crop=dataset_fingerprint["shapes_after_crop"],
        target_spacing=target_spacing,
        margin_mm=DEFAULT_MARGIN_MM,
        percentile=DEFAULT_TARGET_SHAPE_PERCENTILE,
    )

    plans = {
        "dataset_name": dataset_dir.name,
        "normalization": "per_case_zscore",
        "roi_labels": {"left": 1, "right": 2},
        "margin_mm": DEFAULT_MARGIN_MM,
        "target_spacing": target_spacing,
        "target_shape": target_shape,
        "target_shape_percentile": DEFAULT_TARGET_SHAPE_PERCENTILE,
        "output_format": "b2nd",
        "output_folder_name": "mm_b2nd",
    }

    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
    plans_path = preprocessed_dataset_dir / "mmPlans.json"
    with plans_path.open("w", encoding="utf-8") as file:
        json.dump(plans, file, indent=2)

    print(plans_path)
    return plans

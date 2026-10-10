from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_mm_plans(plans_path: Path) -> dict:
    if not plans_path.is_file():
        raise FileNotFoundError(
            f"Missing mmPlans.json in {plans_path.parent}. Run mm_plan_experiment first."
        )

    with plans_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_case_channel_files(case_files: list[Path], dataset_json: dict) -> list[Path]:
    file_ending = dataset_json["file_ending"]
    sorted_channel_ids = sorted(
        int(channel_id) for channel_id in dataset_json["channel_names"]
    )
    ordered_files: list[Path] = []
    for channel_id in sorted_channel_ids:
        suffix = f"_{channel_id:04d}{file_ending}"
        matched = next(
            (path for path in case_files if path.name.endswith(suffix)),
            None,
        )
        if matched is None:
            raise FileNotFoundError(f"Missing channel file with suffix {suffix}")
        ordered_files.append(matched)
    return ordered_files


def load_case_image_data(case_file_paths: list[Path], dataset_json: dict) -> np.ndarray:
    channel_files = get_case_channel_files(case_file_paths, dataset_json)
    channel_arrays = [
        sitk.GetArrayFromImage(sitk.ReadImage(str(channel_file)))
        for channel_file in channel_files
    ]
    return np.stack(channel_arrays, axis=0)


def compute_bbox(
    mask_array: np.ndarray,
    roi_label: int,
    margin_mm: list[float],
    spacing: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    roi_voxels = np.argwhere(mask_array == roi_label)
    if roi_voxels.size == 0:
        raise ValueError(f"ROI label {roi_label} not present in mask")

    min_corner = roi_voxels.min(axis=0)
    max_corner = roi_voxels.max(axis=0)
    margin_voxels = np.ceil(
        np.asarray(margin_mm, dtype=np.float64) / np.asarray(spacing, dtype=np.float64)
    ).astype(int)
    lower = np.maximum(min_corner - margin_voxels, 0)
    upper = np.minimum(
        max_corner + margin_voxels + 1,
        np.asarray(mask_array.shape, dtype=int),
    )
    return lower, upper


def resample_array(
    array: np.ndarray,
    input_spacing: list[float],
    target_spacing: list[float],
    is_mask: bool,
) -> np.ndarray:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(reversed(input_spacing)))

    input_size = image.GetSize()
    output_size = [
        max(1, int(round(size * spacing / target)))
        for size, spacing, target in zip(
            input_size,
            reversed(input_spacing),
            reversed(target_spacing),
            strict=True,
        )
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(reversed(target_spacing)))
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    result = resampler.Execute(image)
    return sitk.GetArrayFromImage(result)


def fit_to_target_shape(array: np.ndarray, target_shape: list[int]) -> np.ndarray:
    fitted = array
    for axis, target in enumerate(target_shape):
        current = fitted.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * fitted.ndim
            slices[axis] = slice(start, end)
            fitted = fitted[tuple(slices)]

    pad_width = []
    for current, target in zip(fitted.shape, target_shape, strict=True):
        total_pad = max(0, target - current)
        before = total_pad // 2
        after = total_pad - before
        pad_width.append((before, after))
    return np.pad(fitted, pad_width, mode="constant", constant_values=0)


def zscore_per_channel(data: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    normalized = data.astype(np.float32, copy=True)
    for channel_idx in range(normalized.shape[0]):
        channel = normalized[channel_idx]
        foreground_values = channel[foreground_mask]
        if foreground_values.size == 0:
            raise ValueError("Foreground mask is empty during z-score normalization")
        mean = float(foreground_values.mean())
        std = float(foreground_values.std())
        normalized[channel_idx, foreground_mask] = (foreground_values - mean) / max(
            std,
            1e-8,
        )
        normalized[channel_idx, ~foreground_mask] = 0.0
    return normalized


def preprocess_roi_array(
    image_data: np.ndarray,
    mask_array: np.ndarray,
    spacing: list[float],
    *,
    roi_label: int,
    target_spacing: list[float],
    target_shape: list[int],
    margin_mm: list[float],
) -> np.ndarray:
    lower, upper = compute_bbox(mask_array, roi_label, margin_mm, spacing)
    slices = tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True)
    )

    cropped_channels = np.stack([channel[slices] for channel in image_data], axis=0)
    cropped_mask = (mask_array[slices] == roi_label).astype(np.uint8)
    cropped_channels = np.stack(
        [
            resample_array(channel, spacing, target_spacing, is_mask=False)
            for channel in cropped_channels
        ],
        axis=0,
    )
    cropped_mask = resample_array(
        cropped_mask,
        spacing,
        target_spacing,
        is_mask=True,
    )
    fitted_channels = np.stack(
        [fit_to_target_shape(channel, target_shape) for channel in cropped_channels],
        axis=0,
    )
    fitted_mask = fit_to_target_shape(cropped_mask, target_shape) > 0
    normalized_channels = zscore_per_channel(fitted_channels, fitted_mask)
    return normalized_channels.astype(np.float32, copy=False)

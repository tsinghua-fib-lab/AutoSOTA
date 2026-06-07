"""Shared helpers for dataset-specific MASS preprocessing scripts.

The dataset scripts in this folder are intentionally thin release wrappers:
users edit paths and a few parameters at the top of each file, then the
script calls the maintained implementations under ``preprocessing/mass_preprocessing``.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PREPROCESSING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PREPROCESSING_ROOT / "mass_preprocessing"
PYTHON = sys.executable
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _path(value: str | Path) -> str:
    return str(Path(value).expanduser())


def _str_values(values: Iterable[object]) -> list[str]:
    return [str(v) for v in values]


def run_command(cmd: Sequence[object], dry_run: bool = False) -> None:
    cmd = [str(x) for x in cmd]
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_dataset_preprocess(
    *,
    dataset: str,
    input_dir: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> None:
    run_command(
        [
            PYTHON,
            PIPELINE_DIR / "preprocess.py",
            "--dataset",
            dataset,
            "--input_dir",
            _path(input_dir),
            "--output_dir",
            _path(output_dir),
        ],
        dry_run=dry_run,
    )


def run_totalsegmentator_preprocess(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    modality: str,
    no_crop: bool = False,
    cases: Sequence[str] | None = None,
    max_cases: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    cmd: list[object] = [
        PYTHON,
        PIPELINE_DIR / "preprocess.py",
        "--dataset",
        "totalsegmentator",
        "--input_dir",
        _path(input_dir),
        "--output_dir",
        _path(output_dir),
        "--modality",
        modality.upper(),
    ]
    if no_crop:
        cmd.append("--no-crop")
    if cases:
        cmd.extend(["--cases", *cases])
    if max_cases is not None:
        cmd.extend(["--max-cases", max_cases])
    if verbose:
        cmd.append("--verbose")
    run_command(cmd, dry_run=dry_run)


def run_sam2_masks(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    sam2_checkpoint: str | Path,
    gpu_ids: Sequence[int],
    model_cfg: str = DEFAULT_SAM2_CFG,
    images: Sequence[str] | None = None,
    auto_select_axes: bool = True,
    process_all_axes_if_isotropic: bool = False,
    isotropic_threshold: float = 1.3,
    min_slices_per_sample: int = 10,
    physical_interval_mm: float = 30.0,
    max_masks_per_slice: int = 70,
    torch_compile: bool = False,
    dry_run: bool = False,
) -> None:
    cmd: list[object] = [
        PYTHON,
        PIPELINE_DIR / "generate_masks.py",
        "--image_dir",
        _path(image_dir),
        "--output_dir",
        _path(output_dir),
        "--dataset_name",
        dataset_name,
        "--sam2_checkpoint",
        _path(sam2_checkpoint),
        "--model_cfg",
        model_cfg,
        "--gpu_ids",
        *_str_values(gpu_ids),
        "--isotropic_threshold",
        isotropic_threshold,
        "--min_slices_per_sample",
        min_slices_per_sample,
        "--physical_interval_mm",
        physical_interval_mm,
        "--max_masks_per_slice",
        max_masks_per_slice,
    ]
    if images:
        cmd.extend(["--images", *images])
    if auto_select_axes:
        cmd.append("--auto_select_axes")
    if process_all_axes_if_isotropic:
        cmd.append("--process_all_axes_if_isotropic")
    if torch_compile:
        cmd.append("--torch_compile")
    run_command(cmd, dry_run=dry_run)


def run_nms(
    *,
    mask_dir: str | Path,
    output_dir: str | Path,
    gpu_ids: Sequence[int],
    images: Sequence[str] | None = None,
    iou_threshold: float = 0.95,
    downsample_factor: int = 4,
    dry_run: bool = False,
) -> None:
    cmd: list[object] = [
        PYTHON,
        PIPELINE_DIR / "postprocess_masks.py",
        "--mask_dir",
        _path(mask_dir),
        "--output_dir",
        _path(output_dir),
        "--iou_threshold",
        iou_threshold,
        "--gpu_ids",
        *_str_values(gpu_ids),
        "--downsample_factor",
        downsample_factor,
    ]
    if images:
        cmd.extend(["--images", *images])
    run_command(cmd, dry_run=dry_run)


def run_resampling_to_training_format(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    auto_label_dir: str | Path | None,
    target_spacing: Sequence[float],
    modality: str,
    gpu_id: int,
    num_workers: int = 1,
    min_size: int = 135,
    images: Sequence[str] | None = None,
    skip_auto_labels: bool = False,
    skip_gt_labels: bool = False,
    ct_clip: Sequence[float] = (-991.0, 500.0),
    non_ct_percentiles: Sequence[float] = (2.0, 98.0),
    dry_run: bool = False,
) -> None:
    cmd: list[object] = [
        PYTHON,
        PIPELINE_DIR / "prepare_training_data.py",
        "--data_dir",
        _path(data_dir),
        "--output_dir",
        _path(output_dir),
        # prepare_training_data expects spacing in array order: z y x.
        "--target_spacing",
        *_str_values(target_spacing),
        "--modality",
        modality,
        "--ct_clip",
        *_str_values(ct_clip),
        "--non_ct_percentiles",
        *_str_values(non_ct_percentiles),
        "--min_size",
        min_size,
        "--gpu_id",
        gpu_id,
        "--num_workers",
        num_workers,
    ]
    if auto_label_dir is not None and not skip_auto_labels:
        cmd.extend(["--auto_label_dir", _path(auto_label_dir)])
    else:
        cmd.append("--skip_auto_labels")
    if skip_gt_labels:
        cmd.append("--skip_gt_labels")
    if images:
        cmd.extend(["--images", *images])
    run_command(cmd, dry_run=dry_run)


def run_sam_to_training(
    *,
    standardized_dir: str | Path,
    sam_mask_dir: str | Path,
    nms_mask_dir: str | Path,
    training_dir: str | Path,
    sam_dataset_name: str,
    modality: str,
    sam2_checkpoint: str | Path,
    gpu_ids: Sequence[int],
    target_spacing: Sequence[float],
    model_cfg: str = DEFAULT_SAM2_CFG,
    images: Sequence[str] | None = None,
    run_sam: bool = True,
    run_nms_stage: bool = True,
    run_resample: bool = True,
    auto_select_axes: bool = True,
    process_all_axes_if_isotropic: bool = False,
    isotropic_threshold: float = 1.3,
    min_slices_per_sample: int = 10,
    physical_interval_mm: float = 30.0,
    max_masks_per_slice: int = 70,
    nms_iou_threshold: float = 0.95,
    nms_downsample_factor: int = 4,
    num_workers: int = 1,
    min_size: int = 135,
    skip_gt_labels: bool = False,
    ct_clip: Sequence[float] = (-991.0, 500.0),
    non_ct_percentiles: Sequence[float] = (2.0, 98.0),
    dry_run: bool = False,
) -> None:
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id")

    if run_sam:
        # Generate SAM2 masks on standardized images before any training-spacing resampling.
        run_sam2_masks(
            image_dir=standardized_dir,
            output_dir=sam_mask_dir,
            dataset_name=sam_dataset_name,
            sam2_checkpoint=sam2_checkpoint,
            gpu_ids=gpu_ids,
            model_cfg=model_cfg,
            images=images,
            auto_select_axes=auto_select_axes,
            process_all_axes_if_isotropic=process_all_axes_if_isotropic,
            isotropic_threshold=isotropic_threshold,
            min_slices_per_sample=min_slices_per_sample,
            physical_interval_mm=physical_interval_mm,
            max_masks_per_slice=max_masks_per_slice,
            dry_run=dry_run,
        )

    final_auto_label_dir = nms_mask_dir if run_nms_stage else sam_mask_dir
    if run_nms_stage:
        # NMS removes highly overlapping SAM objects but keeps masks in image space.
        run_nms(
            mask_dir=sam_mask_dir,
            output_dir=nms_mask_dir,
            gpu_ids=gpu_ids,
            images=images,
            iou_threshold=nms_iou_threshold,
            downsample_factor=nms_downsample_factor,
            dry_run=dry_run,
        )

    if run_resample:
        # The final export writes *_image.npy / *_gt.npy plus compressed auto masks.
        run_resampling_to_training_format(
            data_dir=standardized_dir,
            auto_label_dir=final_auto_label_dir,
            output_dir=training_dir,
            target_spacing=target_spacing,
            modality=modality,
            gpu_id=int(gpu_ids[0]),
            num_workers=num_workers,
            min_size=min_size,
            images=images,
            skip_gt_labels=skip_gt_labels,
            ct_clip=ct_clip,
            non_ct_percentiles=non_ct_percentiles,
            dry_run=dry_run,
        )

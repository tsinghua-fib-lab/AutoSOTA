"""Example preprocessing pipeline for unlabeled imaging data.

This is the release path for pure SSL pretraining data: only images are
required. The script standardizes raw NIfTI images, generates SAM2 auto masks,
and writes a training-ready MASS dataset folder.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

try:
    from .common import run_sam_to_training
except ImportError:
    from common import run_sam_to_training


# Edit this block.
RAW_IMAGE_DIR = Path("/path/to/unlabeled/images")
WORK_DIR = Path("/path/to/MASS_preprocessed/unlabeled_abdomen_ct")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
DATASET_NAME = "unlabeled_abdomen_ct"

SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

# Choose according to the image collection.
MODALITY = "ct"       # "ct", "mr", or "pet"
ANATOMY = "abdomen"   # "abdomen", "chest", "brain", "cardiac", "whole_body"

# ROI_MODE:
# - "threshold": fast body/foreground crop from image intensities.
# - "body": optional TotalSegmentator body crop.
# - "none": no crop.
ROI_MODE = "threshold"
CROP_MARGIN_MM = (20.0, 20.0, 20.0)  # x, y, z
KEEP_LARGEST_COMPONENT = True

# Thresholds are in native image intensity. Set either bound to None to disable.
# For CT abdomen/chest, lower=-500 is a decent body foreground starting point.
THRESHOLD_LOWER = -500.0
THRESHOLD_UPPER = None

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None  # e.g. ["case001", "case002"] after filename stripping


SAM_DATASET_BY_REGION = {
    ("abdomen", "ct"): "abdomen_ct",
    ("abdomen", "mr"): "abdomen_mr",
    ("chest", "ct"): "chest_ct",
    ("brain", "mr"): "brain_mr",
    ("cardiac", "mr"): "cardiac_mr",
    ("whole_body", "ct"): "autopet_ct",
    ("whole_body", "pet"): "autopet_suv",
}


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def iter_nii_images(image_dir: Path) -> list[Path]:
    files = sorted(image_dir.glob("*.nii")) + sorted(image_dir.glob("*.nii.gz"))
    return [p for p in files if not strip_nii_suffix(p).endswith("_gt")]


def reorient_to_ras(image: sitk.Image) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    return orienter.Execute(image)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def threshold_roi_mask(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image)
    mask = np.ones(arr.shape, dtype=bool)
    if THRESHOLD_LOWER is not None:
        mask &= arr >= float(THRESHOLD_LOWER)
    if THRESHOLD_UPPER is not None:
        mask &= arr <= float(THRESHOLD_UPPER)

    mask = ndimage.binary_fill_holes(mask)
    if KEEP_LARGEST_COMPONENT:
        mask = largest_component(mask)
    return mask


def totalseg_body_roi_mask(image: sitk.Image) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_path = tmpdir_path / "image.nii.gz"
        output_dir = tmpdir_path / "totalseg"
        sitk.WriteImage(image, str(image_path))

        subprocess.run(
            [
                "TotalSegmentator",
                "-i",
                str(image_path),
                "-o",
                str(output_dir),
                "--task",
                "body",
                "--fast",
            ],
            check=True,
        )

        body_path = output_dir / "body.nii.gz"
        if not body_path.exists():
            body_path = output_dir / "body_trunc.nii.gz"
        if not body_path.exists():
            raise FileNotFoundError("TotalSegmentator did not create body.nii.gz")

        body = sitk.ReadImage(str(body_path))
        if body.GetSize() != image.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            body = resampler.Execute(body)

        mask = sitk.GetArrayFromImage(body) > 0
        if KEEP_LARGEST_COMPONENT:
            mask = largest_component(mask)
        return mask


def margin_voxels(image: sitk.Image) -> tuple[int, int, int]:
    spacing = image.GetSpacing()  # x, y, z
    return tuple(int(np.ceil(CROP_MARGIN_MM[i] / spacing[i])) for i in range(3))


def crop_to_mask(image: sitk.Image, mask: np.ndarray) -> sitk.Image:
    if not np.any(mask):
        print("Warning: empty ROI mask; returning uncropped image")
        return image

    z_idx, y_idx, x_idx = np.where(mask)
    x_margin, y_margin, z_margin = margin_voxels(image)
    size_xyz = image.GetSize()

    x0 = max(0, int(x_idx.min()) - x_margin)
    x1 = min(size_xyz[0] - 1, int(x_idx.max()) + x_margin)
    y0 = max(0, int(y_idx.min()) - y_margin)
    y1 = min(size_xyz[1] - 1, int(y_idx.max()) + y_margin)
    z0 = max(0, int(z_idx.min()) - z_margin)
    z1 = min(size_xyz[2] - 1, int(z_idx.max()) + z_margin)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([x0, y0, z0])
    roi.SetSize([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1])
    return roi.Execute(image)


def standardize_unlabeled_images(image_paths: Iterable[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        name = strip_nii_suffix(image_path)
        output_path = output_dir / f"{name}.nii.gz"
        if output_path.exists():
            print(f"Skip existing standardized image: {output_path}")
            continue

        print(f"Standardizing {image_path}")
        image = sitk.ReadImage(str(image_path))
        image = reorient_to_ras(image)

        if ROI_MODE == "threshold":
            image = crop_to_mask(image, threshold_roi_mask(image))
        elif ROI_MODE == "body":
            image = crop_to_mask(image, totalseg_body_roi_mask(image))
        elif ROI_MODE == "none":
            pass
        else:
            raise ValueError(f"Unsupported ROI_MODE: {ROI_MODE}")

        sitk.WriteImage(image, str(output_path))


def main() -> None:
    standardized_dir = WORK_DIR / "nii"
    sam_dataset_name = SAM_DATASET_BY_REGION.get((ANATOMY.lower(), MODALITY.lower()))
    if sam_dataset_name is None:
        raise ValueError(
            f"No SAM dataset config for ANATOMY={ANATOMY!r}, MODALITY={MODALITY!r}. "
            "Set one in SAM_DATASET_BY_REGION or choose a supported pair."
        )

    image_paths = iter_nii_images(RAW_IMAGE_DIR)
    if IMAGES:
        keep = set(IMAGES)
        image_paths = [p for p in image_paths if strip_nii_suffix(p) in keep]

    if RUN_STANDARDIZE:
        if DRY_RUN:
            print(f"Would standardize {len(image_paths)} images into {standardized_dir}")
        else:
            standardize_unlabeled_images(image_paths, standardized_dir)

    run_sam_to_training(
        standardized_dir=standardized_dir,
        sam_mask_dir=WORK_DIR / "sam_masks",
        nms_mask_dir=WORK_DIR / "sam_masks_nms",
        training_dir=TRAINING_DATA_ROOT / DATASET_NAME,
        sam_dataset_name=sam_dataset_name,
        modality=MODALITY,
        sam2_checkpoint=SAM2_CHECKPOINT,
        gpu_ids=GPU_IDS,
        target_spacing=TARGET_SPACING,
        physical_interval_mm=PHYSICAL_INTERVAL_MM,
        images=IMAGES,
        run_sam=RUN_SAM,
        run_nms_stage=RUN_NMS,
        run_resample=RUN_RESAMPLE,
        skip_gt_labels=True,
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()

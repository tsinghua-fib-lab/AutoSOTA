"""BraTS preprocessing entry point.

Edit the paths and parameters below, then run this script to standardize BraTS
images/labels, generate SAM2 auto masks, postprocess them, and export MASS
training arrays plus ``dataset.h5``.
"""

from pathlib import Path

try:
    from .common import run_dataset_preprocess, run_sam_to_training
except ImportError:
    from common import run_dataset_preprocess, run_sam_to_training


# Edit this block.
RAW_DATA_DIR = Path("/path/to/BraTS2018")
WORK_DIR = Path("/path/to/MASS_preprocessed/brats")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

# Choose any subset of ["flair", "t1", "t2", "t1ce"].
MODALITIES = ["flair", "t1", "t2", "t1ce"]

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None


def main() -> None:
    standardized_root = WORK_DIR / "nii"

    if RUN_STANDARDIZE:
        run_dataset_preprocess(
            dataset="brats",
            input_dir=RAW_DATA_DIR,
            output_dir=standardized_root,
            dry_run=DRY_RUN,
        )

    for modality_name in MODALITIES:
        standardized_dir = standardized_root / f"brats18_{modality_name}"
        run_sam_to_training(
            standardized_dir=standardized_dir,
            sam_mask_dir=WORK_DIR / "sam_masks" / modality_name,
            nms_mask_dir=WORK_DIR / "sam_masks_nms" / modality_name,
            training_dir=TRAINING_DATA_ROOT / f"brats18_{modality_name}",
            sam_dataset_name="brain_mr",
            modality="mr",
            sam2_checkpoint=SAM2_CHECKPOINT,
            gpu_ids=GPU_IDS,
            target_spacing=TARGET_SPACING,
            physical_interval_mm=PHYSICAL_INTERVAL_MM,
            images=IMAGES,
            run_sam=RUN_SAM,
            run_nms_stage=RUN_NMS,
            run_resample=RUN_RESAMPLE,
            dry_run=DRY_RUN,
        )


if __name__ == "__main__":
    main()

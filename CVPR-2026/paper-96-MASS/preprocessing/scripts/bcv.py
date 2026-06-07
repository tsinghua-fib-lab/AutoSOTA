"""BCV preprocessing entry point.

Edit the paths and parameters below, then run this script to standardize BCV
images/labels, generate SAM2 auto masks, postprocess them, and export MASS
training arrays plus ``dataset.h5``.
"""

from pathlib import Path

try:
    from .common import run_dataset_preprocess, run_sam_to_training
except ImportError:
    from common import run_dataset_preprocess, run_sam_to_training


# Edit this block.
RAW_DATA_DIR = Path("/path/to/BCV/RawData/Training")
WORK_DIR = Path("/path/to/MASS_preprocessed/bcv")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None  # e.g. ["case001", "case002"] for a subset


STANDARDIZED_DIR = WORK_DIR / "nii"
SAM_MASK_DIR = WORK_DIR / "sam_masks"
NMS_MASK_DIR = WORK_DIR / "sam_masks_nms"
TRAINING_DIR = TRAINING_DATA_ROOT / "bcv"


def main() -> None:
    if RUN_STANDARDIZE:
        run_dataset_preprocess(
            dataset="bcv",
            input_dir=RAW_DATA_DIR,
            output_dir=STANDARDIZED_DIR,
            dry_run=DRY_RUN,
        )

    run_sam_to_training(
        standardized_dir=STANDARDIZED_DIR,
        sam_mask_dir=SAM_MASK_DIR,
        nms_mask_dir=NMS_MASK_DIR,
        training_dir=TRAINING_DIR,
        sam_dataset_name="abdomen_ct",
        modality="ct",
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

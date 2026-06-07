"""TotalSegmentator preprocessing entry point.

Edit the paths and parameters below, then run this script to map
TotalSegmentator labels, generate SAM2 auto masks, postprocess them, and export
MASS training arrays plus ``dataset.h5``.
"""

from pathlib import Path

try:
    from .common import run_sam_to_training, run_totalsegmentator_preprocess
except ImportError:
    from common import run_sam_to_training, run_totalsegmentator_preprocess


# Edit this block.
RAW_DATA_DIR = Path("/path/to/TotalSegmentator")
WORK_DIR = Path("/path/to/MASS_preprocessed/totalsegmentator")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

# Choose "CT" or "MRI".
MODALITY = "CT"
NO_CROP = False
CASES = None  # e.g. ["s0001", "s0002"]
MAX_CASES = None

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None


def main() -> None:
    modality_key = MODALITY.lower()
    standardized_dir = WORK_DIR / "nii" / modality_key
    dataset_name = "totalseg_ct" if MODALITY.upper() == "CT" else "totalseg_mr"
    sam_dataset_name = dataset_name
    training_modality = "ct" if MODALITY.upper() == "CT" else "mr"

    if RUN_STANDARDIZE:
        run_totalsegmentator_preprocess(
            input_dir=RAW_DATA_DIR,
            output_dir=standardized_dir,
            modality=MODALITY,
            no_crop=NO_CROP,
            cases=CASES,
            max_cases=MAX_CASES,
            dry_run=DRY_RUN,
        )

    run_sam_to_training(
        standardized_dir=standardized_dir,
        sam_mask_dir=WORK_DIR / "sam_masks" / modality_key,
        nms_mask_dir=WORK_DIR / "sam_masks_nms" / modality_key,
        training_dir=TRAINING_DATA_ROOT / dataset_name,
        sam_dataset_name=sam_dataset_name,
        modality=training_modality,
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

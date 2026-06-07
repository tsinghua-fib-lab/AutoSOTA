"""AMOS preprocessing entry point.

Edit the paths and parameters below, then run this script to standardize AMOS
images/labels, generate SAM2 auto masks, postprocess them, and export MASS
training arrays plus ``dataset.h5``.
"""

from pathlib import Path

try:
    from .common import run_dataset_preprocess, run_sam_to_training
except ImportError:
    from common import run_dataset_preprocess, run_sam_to_training


# Edit this block.
RAW_DATA_DIR = Path("/path/to/AMOS")
WORK_DIR = Path("/path/to/MASS_preprocessed/amos")
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
IMAGES = None

# AMOS standardization creates both "<prefix>_ct" and "<prefix>_mr".
STANDARDIZED_PREFIX = WORK_DIR / "nii" / "amos"
VARIANTS = {
    "ct": {
        "dataset_name": "amos_ct",
        "standardized_dir": Path(f"{STANDARDIZED_PREFIX}_ct"),
        "sam_dataset_name": "abdomen_ct",
        "modality": "ct",
    },
    "mr": {
        "dataset_name": "amos_mr",
        "standardized_dir": Path(f"{STANDARDIZED_PREFIX}_mr"),
        "sam_dataset_name": "abdomen_mr",
        "modality": "mr",
    },
}


def main() -> None:
    if RUN_STANDARDIZE:
        run_dataset_preprocess(
            dataset="amos",
            input_dir=RAW_DATA_DIR,
            output_dir=STANDARDIZED_PREFIX,
            dry_run=DRY_RUN,
        )

    for variant_name, variant in VARIANTS.items():
        run_sam_to_training(
            standardized_dir=variant["standardized_dir"],
            sam_mask_dir=WORK_DIR / "sam_masks" / variant_name,
            nms_mask_dir=WORK_DIR / "sam_masks_nms" / variant_name,
            training_dir=TRAINING_DATA_ROOT / variant["dataset_name"],
            sam_dataset_name=variant["sam_dataset_name"],
            modality=variant["modality"],
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

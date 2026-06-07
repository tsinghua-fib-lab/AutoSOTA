"""StructSeg preprocessing entry point.

Edit the paths and parameters below, then run this script to standardize
StructSeg images/labels, generate SAM2 auto masks, postprocess them, and export
MASS training arrays plus ``dataset.h5``.
"""

from pathlib import Path

try:
    from .common import run_dataset_preprocess, run_sam_to_training
except ImportError:
    from common import run_dataset_preprocess, run_sam_to_training


# Edit this block.
RAW_DATA_DIR = Path("/path/to/StructSeg")
WORK_DIR = Path("/path/to/MASS_preprocessed/structseg")
TRAINING_DATA_ROOT = Path("/path/to/mass_h5")
SAM2_CHECKPOINT = Path("/path/to/sam2.1_hiera_large.pt")
GPU_IDS = [0]
TARGET_SPACING = (1.5, 1.5, 1.5)
PHYSICAL_INTERVAL_MM = 30.0
DRY_RUN = False

# Choose "head_oar" or "thoracic".
STRUCTSEG_TASK = "head_oar"

RUN_STANDARDIZE = True
RUN_SAM = True
RUN_NMS = True
RUN_RESAMPLE = True
IMAGES = None


TASKS = {
    "head_oar": {
        "dataset_name": "structseg_head_oar",
        "preprocess_dataset": "structseg_head_oar",
        "sam_dataset_name": "structseg_head_oar",
    },
    "thoracic": {
        "dataset_name": "structseg_thoracic_oar",
        "preprocess_dataset": "structseg_thoracic",
        "sam_dataset_name": "chest_ct",
    },
}


def main() -> None:
    task = TASKS[STRUCTSEG_TASK]
    standardized_dir = WORK_DIR / "nii" / STRUCTSEG_TASK

    if RUN_STANDARDIZE:
        run_dataset_preprocess(
            dataset=task["preprocess_dataset"],
            input_dir=RAW_DATA_DIR,
            output_dir=standardized_dir,
            dry_run=DRY_RUN,
        )

    run_sam_to_training(
        standardized_dir=standardized_dir,
        sam_mask_dir=WORK_DIR / "sam_masks" / STRUCTSEG_TASK,
        nms_mask_dir=WORK_DIR / "sam_masks_nms" / STRUCTSEG_TASK,
        training_dir=TRAINING_DATA_ROOT / task["dataset_name"],
        sam_dataset_name=task["sam_dataset_name"],
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

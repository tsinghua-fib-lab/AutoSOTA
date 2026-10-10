from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from breastdivider import BreastDividerPredictor

from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def get_breast_segmentation_predictor():
    return BreastDividerPredictor(device="cuda")


def predict_breast_segmentation(predictor, input_path, output_path):
    predictor.predict(
        input_path=input_path,
        output_path=output_path,
    )


def _link_or_copy_file(source_path: Path, target_path: Path) -> None:
    try:
        os.symlink(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _stage_primary_inputs(
    case_files: dict[str, list[Path]],
    *,
    staging_input_dir: Path,
    file_ending: str,
) -> list[str]:
    input_suffix = f"_0000{file_ending}"
    staged_case_ids: list[str] = []

    for case_id, files in sorted(case_files.items()):
        input_file = next(
            (path for path in files if path.name.endswith(input_suffix)),
            None,
        )
        if input_file is None:
            raise FileNotFoundError(
                f"Missing input file ending with '{input_suffix}' for case {case_id}"
            )

        _link_or_copy_file(input_file, staging_input_dir / input_file.name)
        staged_case_ids.append(case_id)

    return staged_case_ids


def _filter_cases_missing_masks(
    case_files: dict[str, list[Path]],
    *,
    masks_tr_dir: Path,
    file_ending: str,
) -> dict[str, list[Path]]:
    pending_case_files: dict[str, list[Path]] = {}
    for case_id, files in sorted(case_files.items()):
        mask_path = masks_tr_dir / f"{case_id}{file_ending}"
        if mask_path.is_file():
            continue
        pending_case_files[case_id] = files
    return pending_case_files


def _verify_masks_written(
    *,
    staged_case_ids: list[str],
    masks_tr_dir: Path,
    file_ending: str,
) -> None:
    for case_id in staged_case_ids:
        mask_path = masks_tr_dir / f"{case_id}{file_ending}"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Breast segmentation output missing for case {case_id}: {mask_path}"
            )


def create_breast_segmentations(d: int) -> tuple[Path, Path]:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    mm_raw = verify_required_global_paths_set()["mm_raw"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    images_tr_dir = dataset_dir / "imagesTr"
    masks_tr_dir = dataset_dir / "masksTr"

    masks_tr_dir.mkdir(parents=True, exist_ok=True)
    file_ending = dataset_json["file_ending"]
    pending_case_files = _filter_cases_missing_masks(
        case_files,
        masks_tr_dir=masks_tr_dir,
        file_ending=file_ending,
    )
    if not pending_case_files:
        return images_tr_dir, masks_tr_dir

    predictor = get_breast_segmentation_predictor()

    with tempfile.TemporaryDirectory(
        dir=dataset_dir,
        prefix=".mm_breastdivider_",
    ) as temp_root:
        temp_root_path = Path(temp_root)
        staging_input_dir = temp_root_path / "input"
        staging_input_dir.mkdir()

        staged_case_ids = _stage_primary_inputs(
            pending_case_files,
            staging_input_dir=staging_input_dir,
            file_ending=file_ending,
        )
        predict_breast_segmentation(
            predictor=predictor,
            input_path=str(staging_input_dir),
            output_path=str(masks_tr_dir),
        )
        _verify_masks_written(
            staged_case_ids=staged_case_ids,
            masks_tr_dir=masks_tr_dir,
            file_ending=file_ending,
        )

    return images_tr_dir, masks_tr_dir

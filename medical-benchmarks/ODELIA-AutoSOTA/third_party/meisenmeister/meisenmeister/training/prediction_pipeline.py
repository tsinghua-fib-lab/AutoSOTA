from __future__ import annotations

import json
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from meisenmeister.plan_and_preprocess.create_breast_seg import (
    _stage_primary_inputs,
    get_breast_segmentation_predictor,
    predict_breast_segmentation,
)
from meisenmeister.plan_and_preprocess.preprocessing_utils import (
    get_case_channel_files,
    load_case_image_data,
    preprocess_roi_array,
)
from meisenmeister.utils.file_utils import discover_case_files
from meisenmeister.utils.prediction_inference import (
    load_checkpoint_payload,
    load_fold_predictors_from_experiment_dir,
    resolve_checkpoint_path,
)
from meisenmeister.utils.prediction_utils import (
    average_probabilities,
    build_prediction_payload,
    predict_roi_with_tta,
    save_binary_mask,
    write_concise_prediction_output,
)
from meisenmeister.utils.training.artifacts import build_experiment_paths


def resolve_trainer_architecture_name(trainer_name: str) -> str:
    from meisenmeister.training.registry import get_trainer_class

    trainer_class = get_trainer_class(trainer_name)
    return getattr(trainer_class, "ARCHITECTURE_NAME", "ResNet3D18")


def validate_folds(folds: list[int]) -> list[int]:
    if not folds:
        raise ValueError("At least one fold must be provided")
    if any(fold < 0 for fold in folds):
        raise ValueError(f"All folds must be non-negative, got {folds}")
    unique_folds = sorted(set(int(fold) for fold in folds))
    if len(unique_folds) != len(folds):
        raise ValueError(f"Folds must be unique, got {folds}")
    return unique_folds


def normalize_prediction_folds(
    folds: list[int | str], *, experiment_dir: Path
) -> list[int | str]:
    if not folds:
        raise ValueError("At least one fold must be provided")
    if "all" in folds:
        if len(folds) != 1:
            raise ValueError("'all' cannot be combined with explicit fold indices")
        if not (experiment_dir / "fold_all").is_dir():
            raise ValueError(f"fold_all does not exist in {experiment_dir}")
        return ["all"]
    return validate_folds([int(fold) for fold in folds])


def stage_breastdivider_primary_inputs(
    case_files_by_case_id: dict[str, list[Path]],
    *,
    staging_input_dir: Path,
    file_ending: str,
) -> list[str]:
    if file_ending == ".nii.gz":
        return _stage_primary_inputs(
            case_files_by_case_id,
            staging_input_dir=staging_input_dir,
            file_ending=file_ending,
        )
    if file_ending != ".mha":
        raise ValueError(
            f"Unsupported prediction file ending for staging: {file_ending}"
        )

    staged_case_ids: list[str] = []
    input_suffix = f"_0000{file_ending}"
    for case_id, files in sorted(case_files_by_case_id.items()):
        input_file = next(
            (path for path in files if path.name.endswith(input_suffix)),
            None,
        )
        if input_file is None:
            raise FileNotFoundError(
                f"Missing input file ending with '{input_suffix}' for case {case_id}"
            )
        image = sitk.ReadImage(str(input_file))
        staged_input_path = staging_input_dir / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(image, str(staged_input_path))
        staged_case_ids.append(case_id)
    return staged_case_ids


def generate_breast_masks_for_cases(
    *,
    case_files_by_case_id: dict[str, list[Path]],
    dataset_json: dict,
    predictor,
    output_dir: Path,
) -> dict[str, Path]:
    file_ending = dataset_json["file_ending"]
    mask_paths = {
        case_id: output_dir / f"{case_id}_breast_mask{file_ending}"
        for case_id in sorted(case_files_by_case_id)
    }
    pending_case_files = {
        case_id: case_files
        for case_id, case_files in sorted(case_files_by_case_id.items())
        if not mask_paths[case_id].is_file()
    }
    if not pending_case_files:
        return mask_paths

    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=".mm_breastdivider_",
    ) as temp_root:
        temp_root_path = Path(temp_root)
        staging_input_dir = temp_root_path / "input"
        staging_input_dir.mkdir()

        staged_case_ids = stage_breastdivider_primary_inputs(
            pending_case_files,
            staging_input_dir=staging_input_dir,
            file_ending=file_ending,
        )
        predict_breast_segmentation(
            predictor=predictor,
            input_path=str(staging_input_dir),
            output_path=str(output_dir),
        )

        for case_id in staged_case_ids:
            generated_mask_path = output_dir / f"{case_id}.nii.gz"
            if not generated_mask_path.is_file():
                raise FileNotFoundError(
                    f"Breast segmentation output missing for case {case_id}: {generated_mask_path}"
                )
            breast_mask_path = output_dir / f"{case_id}_breast_mask{file_ending}"
            if file_ending == ".nii.gz":
                generated_mask_path.replace(breast_mask_path)
            else:
                mask_image = sitk.ReadImage(str(generated_mask_path))
                sitk.WriteImage(mask_image, str(breast_mask_path))
                generated_mask_path.unlink()
            mask_paths[case_id] = breast_mask_path
        return mask_paths


def prepare_case_prediction_inputs(
    *,
    case_id: str,
    case_files: list[Path],
    breast_mask_path: Path,
    dataset_json: dict,
    plans: dict,
    output_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    get_case_channel_files(case_files, dataset_json)
    file_ending = dataset_json["file_ending"]
    mask_image = sitk.ReadImage(str(breast_mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing = [float(i) for i in mask_image.GetSpacing()[::-1]]
    image_data = load_case_image_data(case_files, dataset_json)
    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    zero_margin = [0.0, 0.0, 0.0]

    roi_tensors: dict[str, torch.Tensor] = {}
    artifact_paths = {"breast_mask": str(breast_mask_path)}
    for roi_name, roi_label in plans["roi_labels"].items():
        if not np.any(mask_array == roi_label):
            raise ValueError(
                f"ROI label {roi_label} ({roi_name}) not present for {case_id}"
            )
        roi_output_path = output_dir / f"{case_id}_{roi_name}_mask{file_ending}"
        save_binary_mask(mask_image, mask_array == roi_label, roi_output_path)
        artifact_paths[f"{roi_name}_mask"] = str(roi_output_path)
        roi_array = preprocess_roi_array(
            image_data,
            mask_array,
            spacing,
            roi_label=int(roi_label),
            target_spacing=target_spacing,
            target_shape=target_shape,
            margin_mm=zero_margin,
        )
        roi_tensors[roi_name] = torch.from_numpy(roi_array)

    return roi_tensors, artifact_paths


def iter_prepared_case_prediction_inputs(
    *,
    case_files_by_case_id: dict[str, list[Path]],
    breast_mask_paths: dict[str, Path],
    dataset_json: dict,
    plans: dict,
    output_dir: Path,
    num_workers: int,
):
    case_items = sorted(case_files_by_case_id.items())
    if not case_items:
        return

    worker_count = max(1, int(num_workers))
    max_prefetch = max(worker_count, 2)

    def _prepare_case(case_id: str, case_files: list[Path]):
        roi_tensors, artifact_paths = prepare_case_prediction_inputs(
            case_id=case_id,
            case_files=case_files,
            breast_mask_path=breast_mask_paths[case_id],
            dataset_json=dataset_json,
            plans=plans,
            output_dir=output_dir,
        )
        return case_id, roi_tensors, artifact_paths

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="mm_predict_prepare",
    ) as executor:
        in_flight: dict[int, Future] = {}
        submit_index = 0
        yield_index = 0

        while yield_index < len(case_items):
            while submit_index < len(case_items) and len(in_flight) < max_prefetch:
                case_id, case_files = case_items[submit_index]
                in_flight[submit_index] = executor.submit(
                    _prepare_case,
                    case_id,
                    case_files,
                )
                submit_index += 1

            future = in_flight.pop(yield_index)
            yield future.result()
            yield_index += 1


def load_fold_predictors(
    *,
    dataset_dir: Path,
    results_dir: Path,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int | str],
    checkpoint: str,
    compile_model: bool,
) -> list[dict]:
    experiment_dir = build_experiment_paths(
        results_dir=results_dir,
        dataset_name=dataset_dir.name,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        fold=folds[0],
    )["experiment_dir"]
    return load_fold_predictors_from_experiment_dir(
        experiment_dir=experiment_dir,
        architecture_name=architecture_name,
        folds=folds,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )


def get_experiment_metadata(
    experiment_dir: Path, folds: list[int | str], checkpoint: str
) -> dict:
    checkpoint_path = resolve_checkpoint_path(
        fold_dir=experiment_dir / f"fold_{folds[0]}",
        checkpoint=checkpoint,
    )
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    trainer_config = checkpoint_payload.get("trainer_config", {})
    dataset_name = trainer_config.get("dataset_name", experiment_dir.parent.name)
    trainer_name = trainer_config.get("trainer_name")
    architecture_name = trainer_config.get("architecture_name")
    if trainer_name is None or architecture_name is None:
        raise ValueError(
            "Checkpoint is missing trainer_name or architecture_name required for portable inference"
        )
    return {
        "dataset_id": trainer_config.get("dataset_id"),
        "dataset_name": dataset_name,
        "trainer_name": trainer_name,
        "architecture_name": architecture_name,
        "experiment_postfix": trainer_config.get("experiment_postfix"),
    }


def run_prediction(
    *,
    dataset_id: str | None,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    dataset_json: dict,
    plans: dict,
    fold_predictors: list[dict],
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int | str],
    checkpoint: str,
    use_tta: bool,
    num_workers: int,
    concise_output_path: str | None,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    case_files_by_case_id = discover_case_files(input_path, dataset_json)
    predictor = get_breast_segmentation_predictor()
    breast_mask_paths = generate_breast_masks_for_cases(
        case_files_by_case_id=case_files_by_case_id,
        dataset_json=dataset_json,
        predictor=predictor,
        output_dir=output_path,
    )
    payload = build_prediction_payload(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        input_path=input_path,
        output_path=output_path,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=folds,
        checkpoint=checkpoint,
        use_tta=use_tta,
        compile_enabled=all(
            fold_predictor.get("compile_applied", False)
            for fold_predictor in fold_predictors
        ),
    )

    for case_id, roi_tensors, artifact_paths in tqdm(
        iter_prepared_case_prediction_inputs(
            case_files_by_case_id=case_files_by_case_id,
            breast_mask_paths=breast_mask_paths,
            dataset_json=dataset_json,
            plans=plans,
            output_dir=output_path,
            num_workers=num_workers,
        ),
        total=len(case_files_by_case_id),
        desc="Classifying cases",
        unit="case",
    ):
        roi_results = {}
        for roi_name, roi_tensor in roi_tensors.items():
            per_fold = {}
            fold_probabilities: list[np.ndarray] = []
            for fold_predictor in fold_predictors:
                probabilities = predict_roi_with_tta(
                    fold_predictor["model"],
                    roi_tensor,
                    device=fold_predictor["device"],
                    use_tta=use_tta,
                )
                per_fold[str(fold_predictor["fold"])] = {
                    "checkpoint_path": fold_predictor["checkpoint_path"],
                    "probabilities": [float(value) for value in probabilities.tolist()],
                    "prediction": int(np.argmax(probabilities)),
                }
                fold_probabilities.append(probabilities)
            final_probabilities = average_probabilities(fold_probabilities)
            roi_results[roi_name] = {
                "per_fold": per_fold,
                "probabilities": [
                    float(value) for value in final_probabilities.tolist()
                ],
                "prediction": int(np.argmax(final_probabilities)),
            }

        payload["cases"][case_id] = {
            "artifacts": artifact_paths,
            "rois": roi_results,
        }

    predictions_path = output_path / "predictions.json"
    predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if concise_output_path is not None:
        write_concise_prediction_output(
            predictions_path,
            dataset_json=dataset_json,
            concise_output_path=concise_output_path,
        )
    return predictions_path

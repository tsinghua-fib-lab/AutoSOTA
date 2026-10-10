from __future__ import annotations

import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

SUPPORTED_PREDICTION_FILE_ENDINGS = (".nii.gz", ".mha")


def resolve_prediction_file_ending(input_path: Path, dataset_json: dict) -> str:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_path}")

    channel_ids = sorted(
        int(channel_id) for channel_id in dataset_json["channel_names"]
    )
    candidate_suffixes = {
        file_ending: {f"_{channel_id:04d}{file_ending}" for channel_id in channel_ids}
        for file_ending in SUPPORTED_PREDICTION_FILE_ENDINGS
    }
    detected_suffixes: set[str] = set()

    for path in input_path.iterdir():
        if not path.is_file():
            continue
        for file_ending, suffixes in candidate_suffixes.items():
            if any(path.name.endswith(suffix) for suffix in suffixes):
                detected_suffixes.add(file_ending)

    supported_suffixes = ", ".join(SUPPORTED_PREDICTION_FILE_ENDINGS)
    if len(detected_suffixes) > 1:
        found_suffixes = ", ".join(sorted(detected_suffixes))
        raise ValueError(
            f"Mixed prediction input suffixes detected in {input_path}: "
            f"{found_suffixes}. Prediction supports only one consistent suffix per run "
            f"from: {supported_suffixes}"
        )
    if not detected_suffixes:
        raise ValueError(
            f"Could not detect a supported prediction input suffix in {input_path}. "
            f"Supported suffixes: {supported_suffixes}"
        )
    return next(iter(detected_suffixes))


def resolve_prediction_file_ending_from_paths(paths: list[Path]) -> str:
    detected_suffixes = {
        file_ending
        for path in paths
        for file_ending in SUPPORTED_PREDICTION_FILE_ENDINGS
        if path.name.endswith(file_ending)
    }
    supported_suffixes = ", ".join(SUPPORTED_PREDICTION_FILE_ENDINGS)
    if len(detected_suffixes) > 1:
        found_suffixes = ", ".join(sorted(detected_suffixes))
        raise ValueError(
            "Mixed prediction input suffixes detected for single-case prediction: "
            f"{found_suffixes}. Supported suffixes: {supported_suffixes}"
        )
    if not detected_suffixes:
        raise ValueError(
            "Could not detect a supported prediction input suffix for single-case "
            f"prediction. Supported suffixes: {supported_suffixes}"
        )
    return next(iter(detected_suffixes))


def build_prediction_dataset_json(input_path: Path, dataset_json: dict) -> dict:
    runtime_dataset_json = dict(dataset_json)
    runtime_dataset_json["file_ending"] = resolve_prediction_file_ending(
        input_path,
        dataset_json,
    )
    return runtime_dataset_json


def get_flip_axes(use_tta: bool) -> list[tuple[int, ...]]:
    if not use_tta:
        return [()]
    spatial_axes = (1, 2, 3)
    variants = [()]
    for subset_size in range(1, len(spatial_axes) + 1):
        variants.extend(combinations(spatial_axes, subset_size))
    return variants


def average_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("Cannot average empty probability list")
    stacked = np.stack(probabilities, axis=0).astype(np.float64, copy=False)
    return stacked.mean(axis=0).astype(np.float32, copy=False)


def predict_roi_with_tta(
    model: torch.nn.Module,
    roi_tensor: torch.Tensor,
    *,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    probability_vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for axes in get_flip_axes(use_tta):
            augmented = torch.flip(roi_tensor, dims=axes) if axes else roi_tensor
            logits = model(
                augmented.unsqueeze(0).to(
                    device, dtype=torch.float32, non_blocking=True
                )
            )
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            probability_vectors.append(probabilities)
    return average_probabilities(probability_vectors)


def save_binary_mask(
    mask_image: sitk.Image, binary_mask: np.ndarray, output_path: Path
) -> None:
    binary_image = sitk.GetImageFromArray(binary_mask.astype(np.uint8, copy=False))
    binary_image.CopyInformation(mask_image)
    sitk.WriteImage(binary_image, str(output_path))


def build_prediction_payload(
    *,
    dataset_id: str | None,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    folds: list[int | str],
    checkpoint: str,
    use_tta: bool,
    compile_enabled: bool,
) -> dict:
    return {
        "config": {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "trainer_name": trainer_name,
            "architecture_name": architecture_name,
            "experiment_postfix": experiment_postfix,
            "folds": folds,
            "checkpoint": checkpoint,
            "tta_enabled": bool(use_tta),
            "compile_enabled": bool(compile_enabled),
        },
        "cases": {},
    }


def build_concise_prediction_payload(
    predictions_path: Path,
    *,
    dataset_json: dict,
) -> dict:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict) or len(cases) != 1:
        raise ValueError(
            "Concise output requires predictions.json to contain exactly one case"
        )

    label_names = dataset_json["labels"]
    sorted_label_ids = sorted(int(label_id) for label_id in label_names)
    ordered_label_names = [label_names[str(label_id)] for label_id in sorted_label_ids]

    _, case_payload = next(iter(cases.items()))
    rois = case_payload.get("rois")
    if not isinstance(rois, dict):
        raise ValueError("Prediction case payload must define a 'rois' object")

    concise_payload = {}
    for roi_name, roi_payload in sorted(rois.items()):
        probabilities = roi_payload.get("probabilities")
        if not isinstance(probabilities, list):
            raise ValueError(
                f"Prediction ROI '{roi_name}' must define a 'probabilities' list"
            )
        if len(probabilities) != len(ordered_label_names):
            raise ValueError(
                f"Prediction ROI '{roi_name}' has {len(probabilities)} probabilities, "
                f"expected {len(ordered_label_names)}"
            )
        concise_payload[roi_name] = {
            label_name: float(probability)
            for label_name, probability in zip(
                ordered_label_names,
                probabilities,
                strict=True,
            )
        }
    return concise_payload


def write_concise_prediction_output(
    predictions_path: Path,
    *,
    dataset_json: dict,
    concise_output_path: str,
) -> Path:
    output_path = Path(concise_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concise_payload = build_concise_prediction_payload(
        predictions_path,
        dataset_json=dataset_json,
    )
    output_path.write_text(json.dumps(concise_payload, indent=2), encoding="utf-8")
    return output_path


def stage_prediction_case_file(source_path: Path, output_path: Path) -> None:
    try:
        output_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, output_path)

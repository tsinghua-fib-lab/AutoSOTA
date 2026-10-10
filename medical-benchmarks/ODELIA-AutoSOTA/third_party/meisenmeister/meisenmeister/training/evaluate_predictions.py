from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from meisenmeister.utils import (
    build_final_validation_evaluation,
    save_confusion_matrix_plot,
    save_final_validation_evaluation,
    save_macro_auc_curve_plot,
)

_EXPECTED_NUM_CLASSES = 3
_OPERATING_POINT_TARGET = 0.9


def _normalize_label_value(label_value) -> int:
    if isinstance(label_value, bool):
        raise TypeError("Boolean labels are not supported")
    if isinstance(label_value, int):
        resolved = int(label_value)
    elif isinstance(label_value, list):
        if not label_value:
            raise ValueError("Encountered empty label list")
        resolved = int(np.argmax(np.asarray(label_value, dtype=np.float64)))
    else:
        raise TypeError(f"Unsupported label value type: {type(label_value).__name__}")

    if resolved not in (0, 1, 2):
        raise ValueError(f"Label value must be one of 0, 1, 2, got {resolved}")
    return resolved


def _load_json_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_targets(targets_path: Path) -> dict[str, int]:
    payload = _load_json_file(targets_path)
    normalized_targets: dict[str, int] = {}
    for sample_id, label_value in payload.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Target JSON keys must be non-empty sample ids")
        normalized_targets[sample_id] = _normalize_label_value(label_value)
    if not normalized_targets:
        raise ValueError("Target JSON must not be empty")
    return normalized_targets


def _validate_probability_vector(probabilities, *, sample_id: str) -> list[float]:
    if not isinstance(probabilities, list):
        raise ValueError(f"Prediction probabilities for '{sample_id}' must be a list")
    if len(probabilities) != _EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Prediction probabilities for '{sample_id}' must contain exactly 3 values"
        )
    probability_vector = [float(value) for value in probabilities]
    if not np.all(np.isfinite(probability_vector)):
        raise ValueError(
            f"Prediction probabilities for '{sample_id}' must be finite numbers"
        )
    return probability_vector


def _load_prediction_entries(predictions_path: Path) -> dict[str, dict]:
    payload = _load_json_file(predictions_path)
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("predictions.json must define a 'cases' object")

    entries: dict[str, dict] = {}
    for case_id, case_payload in sorted(cases.items()):
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Prediction case ids must be non-empty strings")
        if not isinstance(case_payload, dict):
            raise ValueError(f"Prediction entry for case '{case_id}' must be an object")
        rois = case_payload.get("rois")
        if not isinstance(rois, dict):
            raise ValueError(f"Prediction case '{case_id}' must define a 'rois' object")

        for roi_name, roi_payload in sorted(rois.items()):
            if not isinstance(roi_name, str) or not roi_name:
                raise ValueError(f"Prediction ROI name for case '{case_id}' is invalid")
            if not isinstance(roi_payload, dict):
                raise ValueError(
                    f"Prediction ROI '{case_id}_{roi_name}' must be a JSON object"
                )
            sample_id = f"{case_id}_{roi_name}"
            if sample_id in entries:
                raise ValueError(f"Duplicate prediction entry found for '{sample_id}'")
            probabilities = _validate_probability_vector(
                roi_payload.get("probabilities"),
                sample_id=sample_id,
            )
            entries[sample_id] = {
                "sample_id": sample_id,
                "case_id": case_id,
                "roi_name": roi_name,
                "probabilities": probabilities,
                "prediction": int(
                    np.argmax(np.asarray(probabilities, dtype=np.float64))
                ),
            }

    if not entries:
        raise ValueError("predictions.json does not contain any ROI predictions")
    return entries


def _build_metric_payload(
    *,
    targets: dict[str, int],
    prediction_entries: dict[str, dict],
) -> dict:
    sample_ids = sorted(prediction_entries)
    missing_targets = [
        sample_id for sample_id in sample_ids if sample_id not in targets
    ]
    if missing_targets:
        missing_str = ", ".join(missing_targets[:5])
        raise ValueError(f"Missing targets for predicted samples: {missing_str}")

    labels = np.asarray(
        [targets[sample_id] for sample_id in sample_ids], dtype=np.int64
    )
    probabilities = np.asarray(
        [prediction_entries[sample_id]["probabilities"] for sample_id in sample_ids],
        dtype=np.float32,
    )
    predictions = np.asarray(
        [prediction_entries[sample_id]["prediction"] for sample_id in sample_ids],
        dtype=np.int64,
    )
    case_ids = [prediction_entries[sample_id]["case_id"] for sample_id in sample_ids]
    roi_names = [prediction_entries[sample_id]["roi_name"] for sample_id in sample_ids]

    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "sample_ids": sample_ids,
        "case_ids": case_ids,
        "roi_names": roi_names,
    }


def _compute_extended_summary(labels: np.ndarray, predictions: np.ndarray) -> dict:
    confusion = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    per_class = {
        str(class_index): {
            "support": int(support[class_index]),
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
        }
        for class_index in range(_EXPECTED_NUM_CLASSES)
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _compute_operating_point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    per_class: dict[str, dict[str, float | bool | None]] = {}
    auc_values: list[float] = []
    specificity_at_target_values: list[float] = []
    sensitivity_at_target_values: list[float] = []

    for class_index in range(_EXPECTED_NUM_CLASSES):
        binary_labels = (labels == class_index).astype(np.int64)
        if np.unique(binary_labels).size < 2:
            per_class[str(class_index)] = {
                "defined": False,
                "auc": None,
                "specificity_at_90_sensitivity": None,
                "sensitivity_at_90_specificity": None,
            }
            continue

        fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, class_index])
        specificity = 1.0 - fpr
        class_auc = float(auc(fpr, tpr))

        valid_specificity = specificity[tpr >= _OPERATING_POINT_TARGET]
        specificity_at_target = (
            float(np.max(valid_specificity)) if valid_specificity.size else 0.0
        )

        valid_sensitivity = tpr[specificity >= _OPERATING_POINT_TARGET]
        sensitivity_at_target = (
            float(np.max(valid_sensitivity)) if valid_sensitivity.size else 0.0
        )

        per_class[str(class_index)] = {
            "defined": True,
            "auc": class_auc,
            "specificity_at_90_sensitivity": specificity_at_target,
            "sensitivity_at_90_specificity": sensitivity_at_target,
        }
        auc_values.append(class_auc)
        specificity_at_target_values.append(specificity_at_target)
        sensitivity_at_target_values.append(sensitivity_at_target)

    macro_auc = float(np.mean(auc_values)) if auc_values else None
    macro_specificity_at_90_sensitivity = (
        float(np.mean(specificity_at_target_values))
        if specificity_at_target_values
        else None
    )
    macro_sensitivity_at_90_specificity = (
        float(np.mean(sensitivity_at_target_values))
        if sensitivity_at_target_values
        else None
    )

    return {
        "per_class": per_class,
        "macro_auc": macro_auc,
        "macro_specificity_at_90_sensitivity": macro_specificity_at_90_sensitivity,
        "macro_sensitivity_at_90_specificity": macro_sensitivity_at_90_specificity,
    }


def _format_interval(interval: dict[str, float | int | bool | None]) -> str:
    if not bool(interval.get("defined")):
        return "undefined"
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return "undefined"
    return f"[{float(lower):.4f}, {float(upper):.4f}]"


def _print_report(payload: dict) -> None:
    summary = payload["summary"]
    extended = payload["extended_summary"]
    challenge_metrics = payload["challenge_metrics"]
    confusion = extended["confusion_matrix"]

    print("Confusion matrix (rows=true, cols=pred):")
    print("      pred_0 pred_1 pred_2")
    for row_index, row in enumerate(confusion):
        print(f"true_{row_index} {row[0]:6d} {row[1]:6d} {row[2]:6d}")
    print("")
    print(f"num_samples: {summary['num_samples']}")
    print(f"accuracy: {extended['accuracy']:.4f}")
    print(f"balanced_accuracy: {summary['balanced_accuracy']:.4f}")
    print(
        "balanced_accuracy_ci: " f"{_format_interval(summary['balanced_accuracy_ci'])}"
    )
    macro_auc_value = summary["macro_auc"]
    if macro_auc_value is None:
        print("macro_auc: undefined")
    else:
        print(f"macro_auc: {float(macro_auc_value):.4f}")
    print(f"macro_auc_ci: {_format_interval(summary['macro_auc_ci'])}")
    challenge_auc = challenge_metrics["macro_auc"]
    if challenge_auc is None:
        print("auc: undefined")
    else:
        print(f"auc: {challenge_auc:.4f}")
    specificity_at_target = challenge_metrics["macro_specificity_at_90_sensitivity"]
    if specificity_at_target is None:
        print("specificity_at_90_sensitivity: undefined")
    else:
        print(f"specificity_at_90_sensitivity: {specificity_at_target:.4f}")
    sensitivity_at_target = challenge_metrics["macro_sensitivity_at_90_specificity"]
    if sensitivity_at_target is None:
        print("sensitivity_at_90_specificity: undefined")
    else:
        print(f"sensitivity_at_90_specificity: {sensitivity_at_target:.4f}")
    print(f"macro_f1: {extended['macro_f1']:.4f}")
    print("per_class:")
    for class_index in range(_EXPECTED_NUM_CLASSES):
        class_payload = extended["per_class"][str(class_index)]
        print(
            f"  class_{class_index}: support={class_payload['support']} "
            f"precision={class_payload['precision']:.4f} "
            f"recall={class_payload['recall']:.4f} "
            f"f1={class_payload['f1']:.4f}"
        )


def _resolve_output_paths(
    predictions_path: Path,
    output_path: str | None,
) -> tuple[Path, Path, Path]:
    if output_path is None:
        output_dir = predictions_path.parent
        evaluation_json_path = output_dir / "evaluation.json"
    else:
        candidate = Path(output_path)
        if candidate.suffix.lower() == ".json":
            evaluation_json_path = candidate
            output_dir = candidate.parent
        else:
            output_dir = candidate
            evaluation_json_path = output_dir / "evaluation.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        evaluation_json_path,
        output_dir / "confusion_matrix.png",
        output_dir / "macro_auc_curve.png",
    )


def evaluate_predictions(
    *,
    targets_path: str,
    predictions_path: str,
    output_path: str | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Path:
    targets = _load_targets(Path(targets_path))
    prediction_entries = _load_prediction_entries(Path(predictions_path))
    metrics = _build_metric_payload(
        targets=targets,
        prediction_entries=prediction_entries,
    )

    payload = build_final_validation_evaluation(
        [metrics],
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    extended_summary = _compute_extended_summary(
        metrics["labels"],
        metrics["predictions"],
    )
    payload["extended_summary"] = extended_summary
    payload["challenge_metrics"] = _compute_operating_point_metrics(
        metrics["labels"],
        metrics["probabilities"],
    )

    evaluation_json_path, confusion_matrix_path, macro_auc_curve_path = (
        _resolve_output_paths(Path(predictions_path), output_path)
    )
    save_final_validation_evaluation(evaluation_json_path, payload)
    save_confusion_matrix_plot(
        np.asarray(extended_summary["confusion_matrix"], dtype=np.int64),
        confusion_matrix_path,
    )
    save_macro_auc_curve_plot(
        metrics["labels"],
        metrics["probabilities"],
        macro_auc_curve_path,
    )

    payload["artifacts"] = {
        "evaluation_json": str(evaluation_json_path),
        "confusion_matrix_png": str(confusion_matrix_path),
        "macro_auc_curve_png": str(macro_auc_curve_path),
    }
    save_final_validation_evaluation(evaluation_json_path, payload)
    _print_report(payload)
    print(f"Saved evaluation JSON to {evaluation_json_path}")
    print(f"Saved confusion matrix plot to {confusion_matrix_path}")
    print(f"Saved macro AUC curve plot to {macro_auc_curve_path}")
    return evaluation_json_path

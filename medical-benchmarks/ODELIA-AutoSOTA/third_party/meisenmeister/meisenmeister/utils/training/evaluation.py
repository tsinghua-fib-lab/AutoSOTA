from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .grad_cam import export_validation_grad_cam
from .metrics import (
    compute_classification_metrics,
    compute_stratified_bootstrap_interval,
)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _macro_auc_from_arrays(
    labels: np.ndarray,
    _predictions: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(
        roc_auc_score(
            labels,
            probabilities,
            average="macro",
            multi_class="ovr",
            labels=list(range(probabilities.shape[1])),
        )
    )


def _balanced_accuracy_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    _probabilities: np.ndarray,
) -> float:
    return float(balanced_accuracy_score(labels, predictions))


def _format_metric_interval(
    metric_value: float | None,
    interval: dict[str, float | int | bool | None],
) -> str | None:
    if metric_value is None or not bool(interval["defined"]):
        return None
    lower = interval["lower"]
    upper = interval["upper"]
    if lower is None or upper is None:
        return None
    return f"{metric_value:.4f} [{float(lower):.4f}, {float(upper):.4f}]"


def build_final_validation_evaluation(
    metrics: list[dict],
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict:
    if not metrics:
        raise ValueError(
            "Cannot build a final validation evaluation from empty metrics"
        )

    labels = np.concatenate([_to_numpy(metric["labels"]) for metric in metrics], axis=0)
    predictions = np.concatenate(
        [_to_numpy(metric["predictions"]) for metric in metrics],
        axis=0,
    )
    probabilities = np.concatenate(
        [_to_numpy(metric["probabilities"]) for metric in metrics],
        axis=0,
    )
    sample_ids = [
        sample_id for metric in metrics for sample_id in list(metric["sample_ids"])
    ]
    case_ids = [case_id for metric in metrics for case_id in list(metric["case_ids"])]
    roi_names = [
        roi_name for metric in metrics for roi_name in list(metric["roi_names"])
    ]

    if not (
        len(sample_ids)
        == len(case_ids)
        == len(roi_names)
        == labels.shape[0]
        == predictions.shape[0]
        == probabilities.shape[0]
    ):
        raise ValueError(
            "Final validation evaluation inputs must have matching lengths"
        )

    balanced_accuracy, macro_auc, macro_auc_defined = compute_classification_metrics(
        labels, predictions, probabilities
    )
    balanced_accuracy_ci = compute_stratified_bootstrap_interval(
        labels,
        predictions,
        probabilities,
        metric_fn=_balanced_accuracy_from_arrays,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    macro_auc_value: float | None = macro_auc
    if not macro_auc_defined:
        macro_auc_value = None
        macro_auc_ci = {
            "defined": False,
            "confidence_level": float(confidence_level),
            "lower": None,
            "upper": None,
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 0,
        }
    else:
        macro_auc_ci = compute_stratified_bootstrap_interval(
            labels,
            predictions,
            probabilities,
            metric_fn=_macro_auc_from_arrays,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed + 1,
        )
        if not bool(macro_auc_ci["defined"]):
            macro_auc_value = None

    predictions_payload = {}
    for index, sample_id in enumerate(sample_ids):
        predictions_payload[sample_id] = {
            "label": int(labels[index]),
            "prediction": int(predictions[index]),
            "probabilities": [float(value) for value in probabilities[index].tolist()],
            "case_id": str(case_ids[index]),
            "roi_name": str(roi_names[index]),
        }

    return {
        "summary": {
            "num_samples": int(labels.shape[0]),
            "balanced_accuracy": float(balanced_accuracy),
            "balanced_accuracy_ci": balanced_accuracy_ci,
            "balanced_accuracy_paper": _format_metric_interval(
                float(balanced_accuracy),
                balanced_accuracy_ci,
            ),
            "macro_auc": macro_auc_value,
            "macro_auc_defined": bool(macro_auc_defined),
            "macro_auc_ci": macro_auc_ci,
            "macro_auc_paper": _format_metric_interval(
                macro_auc_value,
                macro_auc_ci,
            ),
        },
        "predictions": predictions_payload,
    }


def save_final_validation_evaluation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_final_validation_evaluation(
    trainer,
    *,
    output_path: Path,
    log_path: Path,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
    log_fn=None,
    grad_cam_output_dir: Path | None = None,
    grad_cam_checkpoint_kind: str | None = None,
) -> dict:
    val_metrics = []
    trainer.get_architecture().eval()
    for batch_idx, batch in enumerate(trainer.get_val_dataloader(), start=1):
        val_metrics.append(trainer.validate_step(batch, batch_idx))

    payload = build_final_validation_evaluation(
        val_metrics,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    save_final_validation_evaluation(output_path, payload)
    if log_fn is not None:
        log_fn(f"Saved final validation evaluation to {output_path}", log_path)
    if grad_cam_output_dir is not None:
        metadata_path = export_validation_grad_cam(
            trainer,
            output_dir=grad_cam_output_dir,
            checkpoint_kind=(
                "last" if grad_cam_checkpoint_kind is None else grad_cam_checkpoint_kind
            ),
            log_fn=log_fn,
            log_path=log_path,
        )
        if log_fn is not None:
            log_fn(
                f"Saved Grad-CAM outputs to {grad_cam_output_dir} ({metadata_path.name})",
                log_path,
            )
    return payload

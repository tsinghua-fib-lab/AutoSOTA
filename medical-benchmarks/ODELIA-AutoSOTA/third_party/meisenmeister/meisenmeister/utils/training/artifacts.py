from __future__ import annotations

import json
import logging
import math
import shutil
import sys
from pathlib import Path

import blosc2
import matplotlib
import numpy as np
from sklearn.metrics import auc, roc_curve

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_LOGGER_CACHE: dict[Path, logging.Logger] = {}
_TRAINING_CURVE_COLORS = {
    "train_loss": "#0072B2",
    "val_loss": "#D55E00",
    "val_balanced_accuracy": "#009E73",
    "val_macro_auc": "#CC79A7",
    "ema_val_balanced_accuracy": "#E69F00",
}


def build_experiment_paths(
    *,
    results_dir: Path,
    dataset_name: str,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    fold: int | str,
) -> dict[str, Path]:
    experiment_name = f"{trainer_name}_{architecture_name}"
    if experiment_postfix:
        experiment_name = f"{experiment_name}_{experiment_postfix}"
    experiment_dir = results_dir / dataset_name / experiment_name
    fold_dir = experiment_dir / f"fold_{fold}"
    return {
        "experiment_dir": experiment_dir,
        "fold_dir": fold_dir,
        "log_path": fold_dir / "train.log",
        "last_checkpoint_path": fold_dir / "model_last.pt",
        "best_checkpoint_path": fold_dir / "model_best.pt",
        "eval_last_path": fold_dir / "eval_last.json",
        "eval_best_path": fold_dir / "eval_best.json",
        "grad_cam_last_dir": fold_dir / "grad_cam_last",
        "grad_cam_best_dir": fold_dir / "grad_cam_best",
        "plot_path": fold_dir / "training_curves.png",
        "da_preview_path": fold_dir / "da_preview.png",
    }


def ensure_portable_inference_metadata(
    *,
    dataset_dir: Path,
    preprocessed_dataset_dir: Path,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "dataset.json": dataset_dir / "dataset.json",
        "mmPlans.json": preprocessed_dataset_dir / "mmPlans.json",
    }
    for filename, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required inference metadata file not found: {source_path}"
            )
        destination_path = experiment_dir / filename
        if destination_path.exists():
            continue
        shutil.copy2(source_path, destination_path)


def prepare_output_dir(
    *,
    fold_dir: Path,
    log_path: Path,
    continue_training: bool,
) -> str | None:
    fold_dir_exists = fold_dir.exists()
    fold_dir.mkdir(parents=True, exist_ok=True)
    if not continue_training:
        log_path.write_text("", encoding="utf-8")
    if fold_dir_exists and not continue_training:
        return f"WARNING: YOU ARE OVERWRITING EXISTING TRAINING OUTPUT IN {fold_dir}"
    return None


def log_message(message: str, log_path: Path) -> None:
    logger = _get_training_logger(log_path)
    if message.startswith("WARNING:"):
        logger.warning(message)
        return
    logger.info(message)


def _get_training_logger(log_path: Path) -> logging.Logger:
    resolved_path = log_path.resolve()
    logger = _LOGGER_CACHE.get(resolved_path)
    if logger is None:
        logger_name = "meisenmeister.training." + str(resolved_path).replace(
            "/", "."
        ).replace(":", "_")
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _LOGGER_CACHE[resolved_path] = logger

    needs_reconfigure = True
    if len(logger.handlers) == 2:
        stream_handler, file_handler = logger.handlers
        stream_matches = (
            isinstance(stream_handler, logging.StreamHandler)
            and getattr(stream_handler, "stream", None) is sys.stdout
        )
        file_matches = (
            isinstance(file_handler, logging.FileHandler)
            and Path(file_handler.baseFilename) == resolved_path
        )
        needs_reconfigure = not (stream_matches and file_matches)

    if needs_reconfigure:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def save_training_curves(history: dict[str, list[float]], plot_path: Path) -> None:
    epochs = history["epoch"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    metric_axis = axes[0].twinx()

    loss_lines = []
    loss_lines.extend(
        axes[0].plot(
            epochs,
            history["train_loss"],
            label="train_loss",
            color=_TRAINING_CURVE_COLORS["train_loss"],
        )
    )
    loss_lines.extend(
        axes[0].plot(
            epochs,
            history["val_loss"],
            label="val_loss",
            color=_TRAINING_CURVE_COLORS["val_loss"],
        )
    )
    metric_lines = []
    metric_lines.extend(
        metric_axis.plot(
            epochs,
            history["val_balanced_accuracy"],
            label="val_balanced_accuracy",
            color=_TRAINING_CURVE_COLORS["val_balanced_accuracy"],
        )
    )
    metric_lines.append(
        metric_axis.plot(
            epochs,
            history["val_macro_auc"],
            label="val_macro_auc",
            color=_TRAINING_CURVE_COLORS["val_macro_auc"],
        )[0]
    )
    metric_lines.extend(
        metric_axis.plot(
            epochs,
            history["ema_val_balanced_accuracy"],
            label="ema_val_balanced_accuracy",
            color=_TRAINING_CURVE_COLORS["ema_val_balanced_accuracy"],
        )
    )
    axes[0].set_ylabel("loss")
    metric_axis.set_ylabel("metrics")
    axes[0].legend(
        loss_lines + metric_lines,
        [line.get_label() for line in loss_lines + metric_lines],
        loc="best",
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["epoch_time_sec"], label="epoch_time_sec")
    axes[1].set_ylabel("seconds")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["lr"], label="lr")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)


def save_confusion_matrix_plot(
    confusion: np.ndarray,
    plot_path: Path,
    *,
    class_labels: list[str] | None = None,
) -> None:
    confusion_array = np.asarray(confusion, dtype=np.int64)
    if confusion_array.shape != (3, 3):
        raise ValueError(
            "Confusion matrix plot expects a 3x3 matrix for 3-class evaluation"
        )

    labels = class_labels or ["0", "1", "2"]
    if len(labels) != 3:
        raise ValueError("Confusion matrix plot expects exactly three class labels")

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_array, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(range(3), labels=labels)
    axis.set_yticks(range(3), labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")

    max_value = int(confusion_array.max()) if confusion_array.size else 0
    threshold = max_value / 2.0 if max_value else 0.0
    for row_index in range(3):
        for column_index in range(3):
            value = int(confusion_array[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_macro_auc_curve_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    plot_path: Path,
    *,
    class_labels: list[str] | None = None,
) -> None:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.ndim != 2 or probabilities_array.shape[1] != 3:
        raise ValueError("Macro AUC plot expects probabilities with shape (N, 3)")

    labels = class_labels or ["0", "1", "2"]
    if len(labels) != 3:
        raise ValueError("Macro AUC plot expects exactly three class labels")

    figure, axis = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0.0, 1.0, 256)
    macro_tpr = np.zeros_like(mean_fpr)
    class_aucs: list[float] = []

    for class_index, class_label in enumerate(labels):
        y_true = (labels_array == class_index).astype(np.int64)
        fpr, tpr, _ = roc_curve(y_true, probabilities_array[:, class_index])
        class_auc = float(auc(fpr, tpr))
        class_aucs.append(class_auc)
        macro_tpr += np.interp(mean_fpr, fpr, tpr)
        axis.plot(
            fpr,
            tpr,
            linewidth=1.5,
            label=f"class {class_label} (AUC={class_auc:.4f})",
        )

    macro_tpr /= 3.0
    macro_auc = float(auc(mean_fpr, macro_tpr))
    axis.plot(
        mean_fpr,
        macro_tpr,
        linewidth=2.5,
        linestyle="--",
        color="black",
        label=f"macro-average (AUC={macro_auc:.4f})",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=":",
        color="#666666",
        linewidth=1.0,
        label="chance",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Macro AUC ROC Curve")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_da_preview(
    samples: list[dict],
    plot_path: Path,
    *,
    slice_titles: tuple[str, str, str] = ("axial", "coronal", "sagittal"),
) -> None:
    if not samples:
        raise ValueError("At least one sample is required to save a DA preview")

    first_image = np.asarray(_to_numpy_image(samples[0]["image"]))
    if first_image.ndim != 4:
        raise ValueError(
            "DA preview expects channel-first 4D images with shape (C, D, H, W)"
        )
    num_channels = int(first_image.shape[0])
    num_rows = len(samples) * num_channels
    figure, axes = plt.subplots(
        num_rows,
        3,
        figsize=(9, max(3, num_rows * 2)),
        squeeze=False,
    )

    row_index = 0
    for sample in samples:
        image = np.asarray(_to_numpy_image(sample["image"]), dtype=np.float32)
        if image.ndim != 4:
            raise ValueError(
                "DA preview expects channel-first 4D images with shape (C, D, H, W)"
            )
        if int(image.shape[0]) != num_channels:
            raise ValueError("All DA preview samples must have the same channel count")

        sample_id = str(sample.get("sample_id", f"sample_{row_index}"))
        for channel_index in range(num_channels):
            channel = image[channel_index]
            for column_index, slice_2d in enumerate(_extract_mid_slices(channel)):
                axis = axes[row_index, column_index]
                axis.imshow(slice_2d, cmap="gray")
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(slice_titles[column_index])
            axes[row_index, 0].set_ylabel(
                f"{sample_id}\nch {channel_index}",
                rotation=0,
                labelpad=32,
                va="center",
            )
            row_index += 1

    figure.tight_layout()
    figure.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def _to_numpy_image(image) -> np.ndarray:
    if hasattr(image, "detach"):
        return image.detach().cpu().numpy()
    return np.asarray(image)


def _extract_mid_slices(
    channel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if channel.ndim != 3:
        raise ValueError(
            f"DA preview expects 3D channel data with shape (D, H, W), got {channel.shape!r}"
        )
    depth_mid = channel.shape[0] // 2
    height_mid = channel.shape[1] // 2
    width_mid = channel.shape[2] // 2
    axial = channel[depth_mid, :, :]
    coronal = channel[:, height_mid, :]
    sagittal = channel[:, :, width_mid]
    return axial, coronal, sagittal


def save_grad_cam_outputs(
    grad_cam_results: list[dict],
    output_dir: Path,
    *,
    run_metadata: dict,
) -> Path:
    initialize_grad_cam_output_dir(output_dir)
    metadata_predictions: dict[str, dict] = {}

    for result in grad_cam_results:
        metadata_predictions[str(result["sample_id"])] = save_grad_cam_sample(
            result,
            output_dir,
        )

    return write_grad_cam_metadata(
        output_dir,
        run_metadata=run_metadata,
        predictions=metadata_predictions,
    )


def initialize_grad_cam_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.b2nd"):
        stale_path.unlink()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()


def save_grad_cam_sample(result: dict, output_dir: Path) -> dict:
    sample_id = str(result["sample_id"])
    mask_path = output_dir / f"{sample_id}.b2nd"
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)
    blosc2.asarray(
        heatmap,
        urlpath=str(mask_path),
        mode="w",
        chunks=heatmap.shape,
    )
    return {
        "sample_id": sample_id,
        "case_id": str(result["case_id"]),
        "roi_name": str(result["roi_name"]),
        "label": int(result["label"]),
        "prediction": int(result["prediction"]),
        "target_class": int(result["target_class"]),
        "probabilities": [float(value) for value in result["probabilities"]],
        "input_shape": [int(value) for value in result["input_shape"]],
        "mask_path": str(mask_path),
    }


def write_grad_cam_metadata(
    output_dir: Path,
    *,
    run_metadata: dict,
    predictions: dict[str, dict],
) -> Path:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **run_metadata,
                "num_samples": len(predictions),
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path

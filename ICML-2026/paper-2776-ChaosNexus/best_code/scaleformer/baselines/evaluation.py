from collections import defaultdict
from typing import Callable

import numpy as np
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.itertools import batcher
from scaleformer.scaleformer.dataset import TimeSeriesDataset
from scaleformer.utils import safe_standardize
from tqdm import tqdm


def evaluate_forecasting_model(
    model: Callable,
    systems: dict[str, TimeSeriesDataset],
    batch_size: int,
    prediction_length: int,
    metric_names: list[str] | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    eval_subintervals: list[tuple[int, int]] | None = None,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    """
    Evaluate the model on each test system and save metrics.

    Args:
        model: The baseline model to evaluate.
        systems: A dictionary mapping system names to their respective TimeSeriesDataset.
        batch_size: The batch size to use for evaluation.
        prediction_length: The length of the predictions to make.
        metric_names: Optional list of metric names to compute.
        return_predictions: Whether to return the predictions.
        return_contexts: Whether to return the contexts.
        return_labels: Whether to return the future values.
        redo_normalization: Whether to redo the normalization of the future values.

    Returns:
        A tuple containing:
        - system_predictions: A dictionary mapping system names to their predictions.
            Only returned if `return_predictions` is True.
        - system_contexts: A dictionary mapping system names to their contexts.
            Only returned if `return_contexts` is True.
        - system_labels: A dictionary mapping system names to their future values.
            Only returned if `return_labels` is True.
        - system_metrics: A nested dictionary containing computed metrics for each system.
    """
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)

    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    for system in tqdm(systems, desc="Forecasting..."):
        dataset = systems[system]
        predictions, labels, contexts, future_values = [], [], [], []

        # length of the dataset iterator is equal to len(dataset.datasets)*num_eval_windows
        # for each eval window style:
        # - sampled: num_eval_windows = num_test_instances
        # - rolling: num_eval_windows = (T - context_length - prediction_length) // window_stride + 1
        # - single: num_eval_windows = 1
        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )
            # shape: (batch_size, context_length, num_channels)
            context_batch = np.stack(past_values, axis=0)

            # shape: (batch size, prediction_length, num_channels)
            preds = model(context_batch)

            # shape: (batch size, prediction_length, num_channels)
            label_batch = np.stack(future_values, axis=0)

            # Truncate predictions to match future_batch length if needed
            if preds.shape[2] > label_batch.shape[1]:
                preds = preds[..., : label_batch.shape[1], :]

            # standardize using stats from the past_batch
            if redo_normalization:
                preds = safe_standardize(preds, context=context_batch, axis=1)
                label_batch = safe_standardize(
                    label_batch, context=context_batch, axis=1
                )
                context_batch = safe_standardize(context_batch, axis=1)

            labels.append(label_batch)
            predictions.append(preds)
            contexts.append(context_batch)

        # shape: (num_eval_windows * num_datasets, prediction_length, num_channels)
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        # shape: (num_eval_windows * num_datasets, context_length, num_channels)
        contexts = np.concatenate(contexts, axis=0)

        # evaluate metrics for multiple forecast lengths on user-specified subintervals
        # as well as the full prediction length interval
        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals), (
                "All start indices must be less than the prediction length"
            )
            for start, end in eval_subintervals:
                system_metrics[end - start][system] = compute_metrics(
                    predictions[:, start:end, :],
                    labels[:, start:end, :],
                    include=metric_names,
                    batch_axis=0,
                )

        # NOTE: these all need to be of shape: (num_eval_windows * num_datasets, num_channels, prediction_length)
        if return_predictions:
            system_predictions[system] = predictions.transpose(0, 2, 1)
        if return_contexts:
            system_contexts[system] = contexts.transpose(0, 2, 1)
        if return_labels:
            system_labels[system] = labels.transpose(0, 2, 1)

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )

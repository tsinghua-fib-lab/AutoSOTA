"""Metrics for evaluating performances of different approximation methods."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from shapiq import powerset
from shapiq.utils.sets import count_interactions
from shapiq.approximator.sampling import CoalitionSampler

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

__all__ = ["get_all_metrics"]


def _remove_empty_value(interaction: InteractionValues) -> InteractionValues:
    """Manually sets the empty value (e.g. baseline value) to zero in the values array.

    Args:
        interaction: The interaction values to remove the empty value from.

    Returns:
        The interaction values without the empty value.

    """
    if () in interaction.interactions:
        new_interaction = copy.deepcopy(interaction)
        new_interaction.interactions.pop(())
        return new_interaction
    return interaction


DIFF_METRICS = Literal["MSE", "MAE", "SSE", "SAE"]
RANKING_METRICS = Literal["KendallTau@k", "Precision@k"]
METRICS = DIFF_METRICS | RANKING_METRICS | Literal["SpearmanCorrelation", "Faithfulness"]


class Metric(NamedTuple):
    """A named tuple to represent a metric.

    Attributes:
        metric_id: The name of the metric.
        value: The value of the metric.
        order: The order of the metric, if applicable. `None` if not applicable.
        computed_k: The k value used for the metric, if applicable. `None` if not applicable.
    """

    metric_id: METRICS
    value: float
    order: int | None = None
    computed_k: int | None = None


def compute_diff_metrics(
    ground_truth: InteractionValues, estimated: InteractionValues
) -> list[Metric]:
    """Compute metrics via the difference between the ground truth and estimated interaction values.

    Computes the following metrics:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Sum of Squared Errors (SSE)
        - Sum of Absolute Errors (SAE)

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.

    Returns:
        The metrics between the ground truth and estimated interaction values.

    """
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    n_values = count_interactions(
        ground_truth.n_players,
        ground_truth.max_order,
        ground_truth.min_order,
    )
    return [
        Metric(metric_id="MSE", value=np.sum(diff_values**2) / n_values),
        Metric(metric_id="MAE", value=np.sum(np.abs(diff_values)) / n_values),
        Metric(metric_id="SSE", value=np.sum(diff_values**2)),
        Metric(metric_id="SAE", value=np.sum(np.abs(diff_values))),
    ]


def compute_kendall_tau(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    k: int | None = None,
) -> Metric:
    """Compute the Kendall Tau between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to `None`, which considers all
            interactions.

    Returns:
        The Kendall Tau between the ground truth and estimated interaction values.

    """
    # get the interactions as a sorted array
    gt_values, estimated_values = [], []
    ground_truth = _remove_empty_value(ground_truth)
    estimated = _remove_empty_value(estimated)

    for interaction in set(ground_truth.interactions.keys()).union(
        estimated.interactions.keys()
    ):
        gt_values.append(ground_truth[interaction])
        estimated_values.append(estimated[interaction])
    # array conversion
    gt_values, estimated_values = np.array(gt_values), np.array(estimated_values)
    # Clip value smaller 1e-10 to avoid issues with kendalltau
    gt_values = np.where(np.abs(gt_values) < 1e-10, 0.0, gt_values)
    estimated_values = np.where(np.abs(estimated_values) < 1e-10, 0.0, estimated_values)

    # sort the values
    gt_indices, estimated_indices = np.argsort(gt_values), np.argsort(estimated_values)
    if k is not None:
        gt_indices, estimated_indices = gt_indices[:k], estimated_indices[:k]
    # compute the Kendall Tau
    tau, _ = kendalltau(gt_indices, estimated_indices)
    return Metric(metric_id="KendallTau", value=tau, computed_k=k)


def compute_spearmans_correlation(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = None
) -> float:
    """Compute the Spearman's correlation between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to `None`, which considers all
            interactions.

    Returns:
        The Spearman's correlation between the ground truth and estimated interaction values.
    """
    # get the interactions as a sorted array
    gt_values, estimated_values = [], []
    for interaction in powerset(
        range(ground_truth.n_players),
        min_size=1,
        max_size=ground_truth.max_order,
    ):
        gt_values.append(ground_truth[interaction])
        estimated_values.append(estimated[interaction])

    spearman_corr, pval = spearmanr(gt_values, estimated_values)
    # # array conversion
    # gt_values, estimated_values = np.array(gt_values), np.array(estimated_values)
    # # sort the values
    # gt_indices, estimated_indices = np.argsort(gt_values), np.argsort(estimated_values)
    # if k is not None:
    #     gt_indices, estimated_indices = gt_indices[:k], estimated_indices[:k]
    # # compute the Spearman's correlation
    # correlation = np.corrcoef(gt_indices, estimated_indices)[0, 1]
    # correlation = float(correlation)
    return Metric(
        metric_id="SpearmanCorrelation",
        value=spearman_corr,
        computed_k=k,
    )


def compute_precision_at_k(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    k: int = 10,
) -> Metric:
    """Compute the precision at k between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to 10.

    Returns:
        The precision at k between the ground truth and estimated interaction values.

    """
    ground_truth_values = _remove_empty_value(ground_truth)
    estimated_values = _remove_empty_value(estimated)
    top_k, _ = ground_truth_values.get_top_k(k=k, as_interaction_values=False)
    top_k_estimated, _ = estimated_values.get_top_k(k=k, as_interaction_values=False)
    precision_at_k = (
        len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k
    )
    return Metric(
        metric_id=f"Precision@k",
        value=precision_at_k,
        computed_k=k,
    )


def compute_faithullness(
    ground_truth: InteractionValues, estimated: InteractionValues
) -> Metric:
    """Compute the Faithullness between two interaction values.
    Sample coaltions and sum the banzhaf interactions in both ground truth and estimated, which are contained in the sampleed coalitions.
    Then compute the R^2 between the two sums.
    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
    Returns:
        The Faithullness between the ground truth and estimated interaction values.
    """
    n_players = ground_truth.n_players
    n_samples = min(1000, 2**n_players - 1)
    sampler = CoalitionSampler(n_players=n_players,
                               sampling_weights=np.ones(n_players + 1),
                               random_state=42)
    sampler.sample(sampling_budget=n_samples)
    sampled_coalitions = sampler.coalitions_matrix
    gt_sums, est_sums = [], []
    for coalition in sampled_coalitions:
        coalition_set = set(np.where(coalition)[0])
        gt_sum = 0.0
        est_sum = 0.0
        for interaction in ground_truth.interactions.keys():
            if set(interaction).issubset(coalition_set):
                gt_sum += ground_truth[interaction]
        for interaction in estimated.interactions.keys():
            if set(interaction).issubset(coalition_set):
                est_sum += estimated[interaction]
        gt_sums.append(gt_sum)
        est_sums.append(est_sum)
    # compute R^2
    correlation_matrix = np.corrcoef(gt_sums, est_sums)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2
    return Metric(
        metric_id="Faithfulness",
        value=r_squared,
    )


def get_all_metrics(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
) -> list[Metric]:
    """Get all metrics for the interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.

    Returns:
        The metrics as a dictionary.

    """

    metrics = [
        compute_spearmans_correlation(ground_truth, estimated, k=10),
        compute_spearmans_correlation(ground_truth, estimated),
        compute_precision_at_k(ground_truth, estimated, k=5),
        compute_precision_at_k(ground_truth, estimated, k=10),
        compute_kendall_tau(ground_truth, estimated),
        compute_kendall_tau(ground_truth, estimated, k=10),
    ]
    if ground_truth.index == "BII":
        metrics.append(compute_faithullness(ground_truth, estimated))
    metrics_diff = compute_diff_metrics(ground_truth, estimated)
    metrics.extend(metrics_diff)
    return metrics

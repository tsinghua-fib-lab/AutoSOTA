"""
FILE: cord.py
INPUT: Correlation matrices and threshold-search settings.
OUTPUT: CORD distance matrices, greedy clusters, and membership labels.
POS: Source-grounded ACC/CORD baseline helper for ICML experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.optimize
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
Cluster = list[Hashable]

logger = logging.getLogger(__name__)


def _as_correlation_frame(corr: FloatArray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(corr, pd.DataFrame):
        return corr.copy()
    values = np.asarray(corr, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("corr must be a square matrix.")
    return pd.DataFrame(values)


def compute_cord_values(
    corr: FloatArray | pd.DataFrame,
    distinguish_direction: bool = True,
) -> tuple[pd.DataFrame, FloatArray | None]:
    """Compute the CORD matrix from correlations, matching the complete old source implementation."""
    corr_frame = _as_correlation_frame(corr)
    dim = corr_frame.shape[0]
    corr_values = corr_frame.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0.0)

    if dim <= 1000:
        expanded = np.expand_dims(corr_values, axis=1)
        transposed = np.transpose(expanded, axes=(1, 0, 2))
        if distinguish_direction:
            diff = np.abs(expanded - transposed)
        else:
            diff = np.minimum(np.abs(expanded - transposed), np.abs(expanded + transposed))
        diff.flat[[(i * dim + j) * dim + i for i in range(dim) for j in range(dim)]] = 0.0
        diff.flat[[(i * dim + j) * dim + j for i in range(dim) for j in range(dim)]] = 0.0
        cord = diff.max(axis=2)
    else:
        diff = None
        cord = np.zeros_like(corr_values)
        for first_idx in range(dim - 1):
            for second_idx in range(first_idx + 1, dim):
                first_profile = np.delete(corr_values[first_idx, :], [first_idx, second_idx])
                second_profile = np.delete(corr_values[second_idx, :], [first_idx, second_idx])
                if distinguish_direction:
                    distance = np.abs(first_profile - second_profile).max()
                else:
                    distance = np.minimum(
                        np.abs(first_profile - second_profile),
                        np.abs(first_profile + second_profile),
                    ).max()
                cord[first_idx, second_idx] = distance
                cord[second_idx, first_idx] = distance
    np.fill_diagonal(cord, np.inf)
    cord_frame = pd.DataFrame(cord, index=corr_frame.index, columns=corr_frame.columns)
    return cord_frame, diff


def threshold_greedy_procedure(distance: pd.DataFrame, thresh: float, min_or_max: str = "min") -> list[Cluster]:
    """Group items greedily using the CORD threshold rule from the complete old source implementation."""
    if min_or_max not in {"min", "max"}:
        raise ValueError("min_or_max must be either 'min' or 'max'.")
    remaining = set(distance.columns)
    clustering: list[Cluster] = []

    while remaining:
        remaining_items = list(remaining)
        if len(remaining_items) == 1:
            cluster = remaining_items
        else:
            sub_distance = distance.loc[remaining_items, remaining_items]
            first_pos, second_pos = np.unravel_index(np.argmin(sub_distance.to_numpy()), sub_distance.shape)
            first_item = sub_distance.columns[first_pos]
            second_item = sub_distance.columns[second_pos]
            if float(distance.loc[first_item, second_item]) > thresh:
                cluster = [first_item]
            elif min_or_max == "min":
                cluster = list(
                    {
                        item
                        for item in remaining
                        if distance.loc[first_item, item] <= thresh or distance.loc[second_item, item] <= thresh
                    }.union({first_item, second_item})
                )
            else:
                cluster = list(
                    {
                        item
                        for item in remaining
                        if distance.loc[first_item, item] <= thresh and distance.loc[second_item, item] <= thresh
                    }.union({first_item, second_item})
                )
        clustering.append(cluster)
        remaining = remaining - set(cluster)
    return clustering


def data_splitting_loss(
    thresh: float,
    distance1: pd.DataFrame,
    distance2: pd.DataFrame,
    diff1: FloatArray,
    diff2: FloatArray,
    min_or_max: str = "min",
) -> float:
    """Evaluate the data-splitting CORD threshold loss from the complete old source implementation."""
    clustering_mask = distance2.copy()
    clustering_mask[:] = 1.0
    clustering = threshold_greedy_procedure(distance1, thresh, min_or_max=min_or_max)
    for cluster in clustering:
        clustering_mask.loc[cluster, cluster] = 0.0
    loss = np.abs((np.expand_dims(clustering_mask.to_numpy(), axis=2) * diff2) - diff1).max(axis=2) ** 2
    return float(loss.sum())


def avg_intracluster_corr(
    corr: FloatArray | pd.DataFrame,
    clustering: Sequence[Sequence[Hashable]],
    min_clusters: int | None = None,
    max_clusters: int | None = None,
) -> float:
    """Return average within-cluster correlation with the source CORD validity checks."""
    corr_frame = _as_correlation_frame(corr)
    if min_clusters is not None and len(clustering) < min_clusters:
        return -1.0
    if max_clusters is not None and len(clustering) > max_clusters:
        return -1.0
    corr_copy = corr_frame.copy()
    np.fill_diagonal(corr_copy.values, 0.0)
    n_edges = 0
    sum_corr = 0.0
    for cluster in clustering:
        if len(cluster) > 1:
            sum_corr += float(corr_copy.loc[list(cluster), list(cluster)].to_numpy().sum())
            n_edges += len(cluster) * (len(cluster) - 1)
    if n_edges == 0:
        return -1.0
    return sum_corr / n_edges


def avg_corr_loss_wrapper(
    thresh: float,
    cord: pd.DataFrame,
    corr: pd.DataFrame,
    min_or_max: str,
    min_clusters: int | None,
    max_clusters: int | None,
) -> float:
    clustering = threshold_greedy_procedure(cord, thresh, min_or_max=min_or_max)
    return -avg_intracluster_corr(corr, clustering, min_clusters, max_clusters)


def cord_clustering(
    corr: FloatArray | pd.DataFrame,
    n_clusters: int | None = None,
    min_clusters: int | None = None,
    max_clusters: int | None = None,
    calibration: str = "avg_intra_cluster_corr",
    search_range: tuple[float, float] | None = None,
    search_Ns: int | None = None,
    min_or_max: str = "min",
    distinguish_direction: bool = True,
) -> list[Cluster]:
    """Run the CORD greedy clustering routine ported from the complete old source tree."""
    corr_frame = _as_correlation_frame(corr)
    cord, _ = compute_cord_values(corr_frame, distinguish_direction=distinguish_direction)
    dim = corr_frame.shape[0]
    if n_clusters is not None:
        if n_clusters <= 0 or n_clusters > dim:
            raise ValueError("n_clusters must be positive and no larger than the matrix dimension.")
        clustering: list[Cluster] = [[] for _ in range(dim)]
        low = 0.0
        high = 2.0
        while True:
            thresh = (high + low) / 2.0
            clustering = threshold_greedy_procedure(cord, thresh, min_or_max=min_or_max)
            if high - low < 1e-8 or len(clustering) == n_clusters:
                logger.info("CORD threshold selected: %.8g", thresh)
                return clustering
            if len(clustering) < n_clusters:
                high = thresh
            elif len(clustering) > n_clusters:
                low = thresh

    if calibration != "avg_intra_cluster_corr":
        raise ValueError("Only avg_intra_cluster_corr calibration is implemented in the current source.")
    if search_range is None or search_Ns is None:
        raise ValueError("search_range and search_Ns are required when n_clusters is not fixed.")
    search_range = (max(0.0, search_range[0]), min(2.0, search_range[1]))
    clustering: list[Cluster] = []
    if min_clusters is None or max_clusters is None:
        raise ValueError("min_clusters and max_clusters are required when n_clusters is not fixed.")
    while len(clustering) > max_clusters or len(clustering) < min_clusters:
        threshold = scipy.optimize.brute(
            avg_corr_loss_wrapper,
            ranges=[search_range],
            Ns=search_Ns,
            finish=None,
            args=(cord, corr_frame, min_or_max, min_clusters, max_clusters),
            workers=-1,
        )
        threshold_value = float(np.asarray(threshold).reshape(-1)[0])
        clustering = threshold_greedy_procedure(cord, threshold_value, min_or_max=min_or_max)
        if len(clustering) > max_clusters:
            search_range = (search_range[1], search_range[1] * 2.0)
        elif len(clustering) < min_clusters:
            search_range = (search_range[0] * 0.5, search_range[0])
    return clustering


def cluster_list_to_membership(
    clusters: Sequence[Sequence[Hashable]],
    items: Iterable[Hashable] | None = None,
) -> IntArray:
    """Convert CORD cluster lists to one integer label per item."""
    if items is None:
        flattened = [item for cluster in clusters for item in cluster]
        if not all(isinstance(item, (int, np.integer)) for item in flattened):
            raise ValueError("items must be provided when clusters are not integer-indexed.")
        ordered_items = list(range(int(max(flattened)) + 1)) if flattened else []
    else:
        ordered_items = list(items)

    labels = np.full(len(ordered_items), -1, dtype=int)
    index_by_item = {item: idx for idx, item in enumerate(ordered_items)}
    for cluster_idx, cluster in enumerate(clusters):
        for item in cluster:
            if item not in index_by_item:
                raise ValueError(f"Cluster item {item!r} is not present in items.")
            labels[index_by_item[item]] = cluster_idx
    if np.any(labels < 0):
        raise ValueError("At least one item was not assigned to a CORD cluster.")
    return labels


def zipf_weibull_estimation(ret_df: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    """Estimate CORD tail parameters, preserving the helper from the complete old source tree."""
    if k <= 0:
        raise ValueError("k must be positive.")
    import scipy.linalg

    company_ids = ret_df.columns
    centered = ret_df - ret_df.fillna(0).mean()
    sqrt_precision = scipy.linalg.sqrtm(scipy.linalg.inv(centered.cov()))
    whitened_ret = np.matmul(sqrt_precision, centered.T).T
    n_samples = whitened_ret.shape[0]
    topk = np.array([whitened_ret[column].abs().nlargest(k + 1).values[1:].tolist() for column in whitened_ret.columns])
    x_values = np.log(np.log(2 * n_samples / np.arange(1, k + 1)))
    xbar = x_values.mean()
    denominator = np.dot(x_values - xbar, x_values - xbar)
    y_values = np.log(topk)
    ybar = y_values.mean(axis=1, keepdims=True)
    theta = np.matmul(y_values - ybar, x_values - xbar) / denominator
    l_prime = np.exp(ybar - theta * xbar)
    alpha = 1.0 / theta
    return pd.Series(alpha.flatten(), index=company_ids), pd.Series(l_prime.flatten(), index=company_ids)

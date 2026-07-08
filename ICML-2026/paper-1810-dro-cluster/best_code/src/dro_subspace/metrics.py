"""
FILE: metrics.py
INPUT: Coefficient matrices, predicted labels, and true labels.
OUTPUT: Spectral clustering labels and clustering quality scores.
POS: Evaluation helper for reproducible ICML experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.cluster import adjusted_mutual_info_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class ClusteringScore:
    error_rate: float
    accuracy: float
    adjusted_mutual_information: float


def spectral_clustering(coefs: FloatArray, n_clusters: int, embedding_scale: int) -> IntArray:
    """Cluster variables from a self-regression coefficient matrix."""
    if coefs.ndim != 2 or coefs.shape[0] != coefs.shape[1]:
        raise ValueError("coefs must be a square matrix.")
    if n_clusters <= 0 or n_clusters > coefs.shape[0]:
        raise ValueError("n_clusters must be positive and no larger than the matrix dimension.")

    weights = np.abs(coefs) + np.abs(coefs.T)

    # Sparsify: zero out weak connections to suppress inter-cluster noise
    dim = weights.shape[0]
    for i in range(dim):
        row = weights[i].copy()
        nonzero = row[row > 0]
        if len(nonzero) > 1:
            threshold = np.percentile(nonzero, 50)
            weights[i, weights[i] < threshold] = 0.0
    # Re-symmetrize
    weights = np.maximum(weights, weights.T)

    degrees = weights.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("Spectral clustering requires every variable to have positive weighted degree.")
    inv_sqrt_degree = 1.0 / np.sqrt(degrees)
    laplacian = np.diag(inv_sqrt_degree) @ weights @ np.diag(inv_sqrt_degree)
    svd = TruncatedSVD(n_components=n_clusters, n_iter=10, random_state=0)
    svd.fit(laplacian)
    embedding = svd.components_.T
    centered = embedding - embedding.mean(axis=0)
    orthogonal_embedding, _ = np.linalg.qr(centered)
    # Row-wise L2 normalization (standard Ng-Jordan-Weiss spectral clustering)
    row_norms = np.linalg.norm(orthogonal_embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    scaled_embedding = orthogonal_embedding / row_norms
    # Consensus KMeans: co-association matrix from multiple seeds
    n_consensus = 20
    n_points = scaled_embedding.shape[0]
    co_association = np.zeros((n_points, n_points))
    for seed in range(n_consensus):
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1).fit(scaled_embedding).labels_
        co_association += (labels[:, None] == labels[None, :]).astype(np.float64)
    co_association /= n_consensus
    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(co_association).labels_.astype(int)


def clustering_score(true_labels: IntArray, predicted_labels: IntArray) -> ClusteringScore:
    """Evaluate labels up to a permutation of cluster ids."""
    true = np.asarray(true_labels, dtype=int).reshape(-1)
    predicted = np.asarray(predicted_labels, dtype=int).reshape(-1)
    if true.shape != predicted.shape:
        raise ValueError("true_labels and predicted_labels must have the same shape.")
    if true.size == 0:
        raise ValueError("At least one label is required.")

    n_clusters = int(max(true.max(), predicted.max())) + 1
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for pred_idx in range(n_clusters):
        for true_idx in range(n_clusters):
            confusion[pred_idx, true_idx] = int(np.sum((predicted == pred_idx) & (true == true_idx)))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = int(confusion[row_ind, col_ind].sum())
    accuracy = matched / true.size
    ami = float(adjusted_mutual_info_score(true, predicted, average_method="max"))
    return ClusteringScore(error_rate=1.0 - accuracy, accuracy=accuracy, adjusted_mutual_information=ami)

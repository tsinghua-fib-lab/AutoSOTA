"""
FILE: baselines.py
INPUT: Normalized sample matrices and baseline method settings.
OUTPUT: Source-grounded non-DRO baseline cluster labels.
POS: Optional baseline wrappers for ICML experiment reproduction scripts.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]
DEFAULT_RANDOM_STATE = 0
DEFAULT_KMEANS_INIT = 20
DEFAULT_MFC_ITERATIONS = 200


def kmedoids_clustering(samples: FloatArray, n_clusters: int) -> IntArray:
    """Run the pyclustering k-medoids baseline from the old simulation scripts."""
    try:
        from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
        from pyclustering.cluster.kmedoids import kmedoids
    except ModuleNotFoundError as exc:
        message = "Install optional baseline dependencies with `pip install -e '.[baselines]'`."
        raise ModuleNotFoundError(message) from exc

    corr = np.corrcoef(samples.T)
    distance = 1.0 - corr**2
    initial_medoids = kmeans_plusplus_initializer(
        samples.T,
        n_clusters,
        kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE,
    ).initialize(return_index=True)
    instance = kmedoids(distance, initial_medoids, data_type="distance_matrix")
    instance.process()
    clusters = instance.get_clusters()
    membership = np.full(samples.shape[1], -1, dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for variable_idx in cluster:
            membership[variable_idx] = cluster_idx
    if np.any(membership < 0):
        raise RuntimeError("k-medoids did not assign every variable to a cluster.")
    return membership


class MultifactorClustering:
    """
    Multi-factor clustering baseline ported from historical commit 8a30e26.

    The input matrix follows the original contract: N rows are items to be clustered and
    T columns are observations/features for each item.
    """

    def __init__(self, n_clusters: int = 2, n_factors_global: int = 1, n_factors_local: int = 1):
        if n_factors_global < 0 or n_factors_local <= 0:
            raise ValueError("n_factors_global must be non-negative and n_factors_local must be positive.")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        self.K = n_clusters
        self.r_global = n_factors_global
        self.r_local = n_factors_local
        self.svd_global = TruncatedSVD(n_components=self.r_global, random_state=DEFAULT_RANDOM_STATE)
        self.svd_local = TruncatedSVD(n_components=self.r_local, random_state=DEFAULT_RANDOM_STATE)
        self.labels = np.empty(0, dtype=int)
        self.X = np.empty((0, 0), dtype=float)
        self.Z = np.empty((0, 0), dtype=float)
        self.X_c = np.empty((0, 0), dtype=float)
        self.RSS = np.empty((0, 0), dtype=float)
        self.N = 0
        self.T = 0

    def fit(self, items_by_features: FloatArray, n_iter: int = 500) -> None:
        """Fit the historical alternating MFC routine."""
        self.X = items_by_features.copy().reshape(len(items_by_features), -1)
        self.N, self.T = self.X.shape

        kmeans = KMeans(n_clusters=self.K, n_init=DEFAULT_KMEANS_INIT, random_state=DEFAULT_RANDOM_STATE)
        kmeans.fit(self.X)
        self.labels = kmeans.labels_
        self.Z = self.X.copy()

        for _ in range(n_iter):
            self.estimate_factors_local()
            self.clustering()
            if self.r_global > 0:
                self.estimate_factors_global()

    def estimate_factors_local(self) -> None:
        """Estimate cluster-specific factors and residual sums of squares."""
        self.X_c = np.zeros((self.N, self.T))
        self.RSS = np.zeros((self.N, self.K))
        for cluster_idx in range(self.K):
            idx_cluster = np.where(self.labels == cluster_idx)[0]
            z_cluster = self.Z[idx_cluster]

            self.svd_local.fit(z_cluster)
            projection = self.svd_local.transform(self.Z) @ self.svd_local.components_
            self.X_c[idx_cluster] = self.X[idx_cluster] - projection[idx_cluster]
            self.RSS[:, cluster_idx] = np.linalg.norm(self.Z - projection, axis=1)

    def clustering(self) -> None:
        """Assign each item to the cluster with the smallest residual."""
        self.labels = np.argmin(self.RSS, axis=1)

    def estimate_factors_global(self) -> None:
        """Remove common factors from the local residual matrix."""
        self.svd_global.fit(self.X_c)
        self.Z = self.X - self.svd_global.transform(self.X_c) @ self.svd_global.components_


def multifactor_clustering(
    samples: FloatArray,
    n_clusters: int,
    n_factors_global: int = 1,
    n_factors_local: int = 1,
    n_iter: int = DEFAULT_MFC_ITERATIONS,
) -> IntArray:
    """Cluster sample columns using the historical MFC baseline."""
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional matrix.")
    model = MultifactorClustering(
        n_clusters=n_clusters,
        n_factors_global=n_factors_global,
        n_factors_local=n_factors_local,
    )
    model.fit(samples.T, n_iter=n_iter)
    return model.labels.astype(int)

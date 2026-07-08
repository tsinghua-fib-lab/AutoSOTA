"""
FILE: __init__.py
INPUT: Public imports from the dro_subspace package.
OUTPUT: Stable package-level API for release scripts.
POS: Package API boundary.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from dro_subspace.baselines import MultifactorClustering, kmedoids_clustering, multifactor_clustering
from dro_subspace.cord import cluster_list_to_membership, cord_clustering, compute_cord_values
from dro_subspace.dro import compute_dro_coefficients, dro_sqrt_delta, spectral_regression_admm
from dro_subspace.metrics import ClusteringScore, clustering_score, spectral_clustering
from dro_subspace.nodewise import compute_lasso_coefficients, compute_sqrt_lasso_coefficients
from dro_subspace.synthetic import make_cluster_labels, normalize_design, sample_cluster_sizes, sample_random_subspace

__all__ = [
    "ClusteringScore",
    "MultifactorClustering",
    "cluster_list_to_membership",
    "clustering_score",
    "compute_dro_coefficients",
    "compute_cord_values",
    "cord_clustering",
    "compute_lasso_coefficients",
    "compute_sqrt_lasso_coefficients",
    "dro_sqrt_delta",
    "kmedoids_clustering",
    "make_cluster_labels",
    "multifactor_clustering",
    "normalize_design",
    "sample_cluster_sizes",
    "sample_random_subspace",
    "spectral_clustering",
    "spectral_regression_admm",
]

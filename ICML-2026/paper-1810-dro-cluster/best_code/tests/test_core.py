"""
FILE: test_core.py
INPUT: Small deterministic synthetic examples.
OUTPUT: Release smoke tests for generation and clustering metrics.
POS: Test coverage for public helper APIs.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import numpy as np

from dro_subspace.baselines import multifactor_clustering
from dro_subspace.cord import cluster_list_to_membership, cord_clustering, compute_cord_values
from dro_subspace.dro import compute_dro_coefficients, dro_sqrt_delta
from dro_subspace.metrics import clustering_score
from dro_subspace.synthetic import make_cluster_labels, normalize_design, sample_random_subspace
from scripts.run_synthetic_experiments import parse_methods


def test_sample_random_subspace_is_deterministic() -> None:
    first, *_ = sample_random_subspace(
        n_samples=12,
        n_variables=8,
        cluster_sizes=[4, 4],
        n_group_factors=1,
        rho_range=(0.0, 0.1),
        noise_range=(0.1, 0.2),
        seed=7,
    )
    second, *_ = sample_random_subspace(
        n_samples=12,
        n_variables=8,
        cluster_sizes=[4, 4],
        n_group_factors=1,
        rho_range=(0.0, 0.1),
        noise_range=(0.1, 0.2),
        seed=7,
    )
    assert np.allclose(first, second)
    assert normalize_design(first).shape == (12, 8)


def test_clustering_score_is_permutation_invariant() -> None:
    true_labels = make_cluster_labels([2, 3, 1])
    predicted_labels = np.array([2, 2, 0, 0, 0, 1])
    score = clustering_score(true_labels, predicted_labels)
    assert score.error_rate == 0.0
    assert score.accuracy == 1.0
    assert score.adjusted_mutual_information == 1.0


def test_dro_delta_is_seeded_and_coefficients_have_zero_diagonal() -> None:
    samples, *_ = sample_random_subspace(
        n_samples=20,
        n_variables=6,
        cluster_sizes=[3, 3],
        n_group_factors=1,
        noise_range=(0.1, 0.2),
        seed=11,
    )
    normalized = normalize_design(samples)
    covariance = np.cov(normalized.T)
    first_delta = dro_sqrt_delta(covariance, normalized.shape[0], normalized, n_simulations=8, seed=123)
    second_delta = dro_sqrt_delta(covariance, normalized.shape[0], normalized, n_simulations=8, seed=123)
    coefs = compute_dro_coefficients(
        normalized,
        n_simulations=8,
        seed=123,
        max_outer_iterations=5000,
    )
    assert first_delta == second_delta
    assert first_delta > 0.0
    assert coefs.shape == (6, 6)
    assert np.allclose(np.diag(coefs), 0.0)


def test_cord_clustering_assigns_every_item() -> None:
    corr = np.array(
        [
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.0, 0.1, 0.8, 1.0],
        ]
    )
    cord, diff = compute_cord_values(corr)
    clusters = cord_clustering(corr, n_clusters=2)
    membership = cluster_list_to_membership(clusters)
    assert cord.shape == (4, 4)
    assert diff.shape == (4, 4, 4)
    assert membership.shape == (4,)
    assert set(membership.tolist()) == {0, 1}


def test_cord_signed_distance_matches_modified_acc_rule() -> None:
    corr = np.array(
        [
            [1.0, 0.0, 0.7, -0.6],
            [0.0, 1.0, -0.7, 0.6],
            [0.7, -0.7, 1.0, 0.1],
            [-0.6, 0.6, 0.1, 1.0],
        ]
    )
    directional, _ = compute_cord_values(corr, distinguish_direction=True)
    signed, _ = compute_cord_values(corr, distinguish_direction=False)
    assert signed.loc[0, 1] < directional.loc[0, 1]
    assert signed.loc[0, 1] == 0.0


def test_multifactor_clustering_assigns_every_item() -> None:
    samples = np.array(
        [
            [1.0, 1.1, 0.0, 0.1],
            [0.9, 1.0, 0.1, 0.0],
            [1.1, 0.9, -0.1, 0.0],
            [0.0, 0.1, 1.0, 0.9],
            [0.1, 0.0, 0.9, 1.1],
            [-0.1, 0.0, 1.1, 1.0],
        ]
    )
    predicted = multifactor_clustering(
        samples,
        n_clusters=2,
        n_factors_global=0,
        n_factors_local=1,
        n_iter=3,
    )
    score = clustering_score(np.array([0, 0, 1, 1]), predicted)
    assert predicted.shape == (4,)
    assert score.accuracy == 1.0


def test_method_parser_accepts_mfc_variants() -> None:
    assert parse_methods("dro,mfc,mfc_1x1,mfc_2x3") == ["dro", "mfc", "mfc_1x1", "mfc_2x3"]

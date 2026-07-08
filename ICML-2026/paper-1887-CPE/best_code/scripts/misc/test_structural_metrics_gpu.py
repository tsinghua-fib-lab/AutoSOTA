# test_structural_metrics_gpu.py
import numpy as np
import torch

import metrics.structural_metrics as sm_cpu
import metrics.structural_metrics_gpu as sm_gpu


def _make_truth_dag(D: int, rng: np.random.Generator) -> np.ndarray:
    """Upper-triangular weighted DAG (acyclic by construction)."""
    W = np.zeros((D, D), dtype=float)
    tri = np.triu(rng.normal(0.0, 1.0, size=(D, D)), k=1)
    mask = rng.random((D, D)) < 0.10
    W[mask] = tri[mask]
    np.fill_diagonal(W, 0.0)
    return W


def _make_particles(W_star: np.ndarray, S: int, rng: np.random.Generator):
    """Noisy particles around truth + a few extra edges."""
    D = W_star.shape[0]
    parts = []
    for _ in range(S):
        W = W_star.copy()

        # noise on truth edges
        W += rng.normal(0.0, 0.1, size=(D, D)) * (np.abs(W_star) > 1e-12)

        # a few extra edges (can be anywhere off-diagonal)
        extra = rng.random((D, D)) < 0.02
        np.fill_diagonal(extra, 0)
        W += extra * (rng.normal(0.4, 0.2, size=(D, D)))

        parts.append(W)

    w = rng.random(S)
    w = w / w.sum()
    return parts, w


def test_structural_metrics_gpu_matches_cpu():
    rng = np.random.default_rng(0)

    D = 25
    S = 80
    W_star = _make_truth_dag(D, rng)
    A_true = (np.abs(W_star) > 1e-12).astype(int)
    np.fill_diagonal(A_true, 0)

    particles, weights = _make_particles(W_star, S, rng)

    # CPU baseline
    out_cpu = sm_cpu.metrics_from_weighted_samples(particles, weights, A_true)

    # GPU version, CPU device (still exercises torch vectorization path if we pass tensor)
    particles_t = torch.tensor(np.stack(particles, axis=0), dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)
    out_gpu = sm_gpu.metrics_from_weighted_samples(particles_t, weights_t, A_true)

    assert out_cpu.keys() == out_gpu.keys()

    # These should match extremely closely (same definitions, just vectorized).
    for k in out_cpu:
        np.testing.assert_allclose(out_gpu[k], out_cpu[k], rtol=1e-10, atol=1e-12)


def test_structural_metrics_gpu_accepts_numpy_particles_too():
    """Ensures GPU module also works as a drop-in with list-of-numpy particles."""
    rng = np.random.default_rng(1)

    D = 20
    S = 50
    W_star = _make_truth_dag(D, rng)
    A_true = (np.abs(W_star) > 1e-12).astype(int)
    np.fill_diagonal(A_true, 0)

    particles, weights = _make_particles(W_star, S, rng)

    out_cpu = sm_cpu.metrics_from_weighted_samples(particles, weights, A_true)
    out_gpu = sm_gpu.metrics_from_weighted_samples(particles, weights, A_true)

    for k in out_cpu:
        np.testing.assert_allclose(out_gpu[k], out_cpu[k], rtol=1e-12, atol=1e-12)
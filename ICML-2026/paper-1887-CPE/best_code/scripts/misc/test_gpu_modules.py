import numpy as np
import torch
import pytest

# ---- CPU baselines ----
from feedback import expert as expert_cpu
from metrics import metrics as metrics_cpu
from metrics import structural_metrics as sm_cpu
from inference import static_baselines as sb_cpu

# ---- GPU versions (your uploaded files) ----
from feedback import  expert_gpu
from metrics import metrics_gpu
from metrics import  structural_metrics_gpu as sm_gpu
from inference import  static_baselines_gpu as sb_gpu

from inference.ParticlePosterior import ParticlePosterior as ParticlePosteriorCPU
from inference.ParticlePosterior_gpu import ParticlePosterior as ParticlePosteriorGPU


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def make_truth(D: int, rng: np.random.Generator) -> np.ndarray:
    """Simple acyclic truth DAG (upper triangular)."""
    W = np.zeros((D, D))
    tri = np.triu(rng.normal(0.0, 1.0, size=(D, D)), k=1)
    mask = rng.random((D, D)) < 0.08
    W[mask] = tri[mask]
    np.fill_diagonal(W, 0.0)
    return W


def make_particles(W_star: np.ndarray, S: int, rng: np.random.Generator):
    """Small noisy particle cloud around truth."""
    D = W_star.shape[0]
    parts = []
    for _ in range(S):
        W = W_star.copy()
        W += rng.normal(0.0, 0.1, size=(D, D)) * (np.abs(W_star) > 1e-12)
        extra = rng.random((D, D)) < 0.01
        np.fill_diagonal(extra, 0)
        W += extra * (rng.normal(0.4, 0.2, size=(D, D)))
        parts.append(W)
    w = rng.random(S)
    w /= w.sum()
    return parts, w


def make_posteriors(D=20, S=64, seed=0):
    rng = np.random.default_rng(seed)
    W_star = make_truth(D, rng)
    parts, w = make_particles(W_star, S, rng)

    post_cpu = ParticlePosteriorCPU(parts, w)

    parts_t = torch.tensor(np.stack(parts), dtype=torch.float64)
    w_t = torch.tensor(w, dtype=torch.float64)
    post_gpu = ParticlePosteriorGPU(parts_t, w_t, device=torch.device("cpu"))

    A_true = (np.abs(W_star) > 1e-12).astype(int)
    np.fill_diagonal(A_true, 0)
    return W_star, A_true, post_cpu, post_gpu, rng


# ---------------------------------------------------------------------
# Expert tests
# ---------------------------------------------------------------------

def test_expert_gpu_matches_cpu_distribution():
    rng = np.random.default_rng(123)
    D = 25
    W_star = make_truth(D, rng)

    i, j = 4, 17
    beta_edge, beta_dir, lam = 3.0, 2.0, 0.7

    # CPU probabilities
    _, p_cpu = expert_cpu.simulate_expert_answer(
        W_star, i, j, beta_edge, beta_dir, lam,
        phi_star_ij=1.0, phi_star_ji=1.0,
        rng=np.random.default_rng(999),
    )

    # GPU probabilities
    _, p_gpu = expert_gpu.simulate_expert_answer(
        W_star, i, j, beta_edge, beta_dir, lam,
        phi_star_ij=1.0, phi_star_ji=1.0,
        rng=np.random.default_rng(999),
    )

    np.testing.assert_allclose(p_cpu, p_gpu, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("beta_edge,beta_dir,lam", [
    (3.0, 2.0, 0.5),
    (1.5, 1.2, 2.0),
])
def test_metrics_gpu_matches_cpu(beta_edge, beta_dir, lam):
    W_star, A_true, post_cpu, post_gpu, _ = make_posteriors(D=18, S=64, seed=7)

    m_cpu = metrics_cpu.expected_true_class_prob(
        post_cpu, beta_edge, beta_dir, lam, A_true
    )
    m_gpu = metrics_gpu.expected_true_class_prob(
        post_gpu, beta_edge, beta_dir, lam, A_true
    )
    assert abs(m_cpu - m_gpu) < 1e-9

    b_cpu = metrics_cpu.mean_brier_score(
        post_cpu, beta_edge, beta_dir, lam, A_true
    )
    b_gpu = metrics_gpu.mean_brier_score(
        post_gpu, beta_edge, beta_dir, lam, A_true
    )
    assert abs(b_cpu - b_gpu) < 1e-9


# ---------------------------------------------------------------------
# Structural metrics tests
# ---------------------------------------------------------------------

def test_structural_metrics_gpu_matches_cpu():
    rng = np.random.default_rng(0)
    D, S = 15, 50
    W_star = make_truth(D, rng)
    A_true = (np.abs(W_star) > 1e-12).astype(int)
    np.fill_diagonal(A_true, 0)

    parts, w = make_particles(W_star, S, rng)

    cpu = sm_cpu.metrics_from_weighted_samples(parts, w, A_true)
    gpu = sm_gpu.metrics_from_weighted_samples(parts, w, A_true)

    assert cpu.keys() == gpu.keys()
    for k in cpu:
        assert abs(cpu[k] - gpu[k]) < 1e-12


# ---------------------------------------------------------------------
# Static baseline tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["static_eig"])
def test_static_baselines_gpu_matches_cpu(policy):
    W_star, A_true, post_cpu, post_gpu, rng = make_posteriors(D=22, S=64, seed=42)

    D = A_true.shape[0]
    T = 15
    screen_k = 40
    beta_edge, beta_dir, lam = 3.0, 2.0, 0.7

    rng_cpu = np.random.default_rng(1234)
    rng_gpu = np.random.default_rng(1234)

    sched_cpu = sb_cpu.build_static_schedule(
        posterior=post_cpu,
        D=D,
        T=T,
        policy=policy,
        screen_k=screen_k,
        beta_edge=beta_edge,
        beta_dir=beta_dir,
        lam=lam,
        rng=rng_cpu,
    )

    sched_gpu = sb_gpu.build_static_schedule(
        posterior=post_gpu,
        D=D,
        T=T,
        policy=policy,
        screen_k=screen_k,
        beta_edge=beta_edge,
        beta_dir=beta_dir,
        lam=lam,
        rng=rng_gpu,
    )

    assert len(sched_cpu) == len(sched_gpu) == T

    # Allow reordering in case of ties (esp. EIG)
    assert {tuple(x) for x in sched_cpu} == {tuple(x) for x in sched_gpu}


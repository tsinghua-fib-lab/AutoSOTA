# test_prior_neighborhood_match.py
import numpy as np
import torch
import pytest

import prior.prior as prior_cpu
import prior.prior_gpu as prior_gpu


def shd_directed_flip1(A_pred: np.ndarray, A_true: np.ndarray) -> int:
    """
    Directed SHD where an orientation flip counts as 1.
    A_* are (D,D) in {0,1}, diagonal assumed 0.
    """
    D = A_true.shape[0]
    shd = 0
    for i in range(D):
        for j in range(i + 1, D):
            tp = (A_true[i, j], A_true[j, i])
            pp = (A_pred[i, j], A_pred[j, i])

            has_t = (tp[0] + tp[1]) > 0
            has_p = (pp[0] + pp[1]) > 0

            if not has_t and not has_p:
                continue
            if has_t != has_p:
                shd += 1
            else:
                # both have an edge; orientation differs?
                if tp != pp:
                    shd += 1
    return shd


def adj_from_weights_np(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = (np.abs(W) > eps).astype(int)
    np.fill_diagonal(A, 0)
    return A


def adj_from_weights_torch(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = (W.abs() > eps).to(torch.int64)
    D = A.shape[-1]
    eye = torch.eye(D, device=A.device, dtype=torch.bool)
    A = A.masked_fill(eye, 0)
    return A


@pytest.mark.parametrize("D,S,seed", [(30, 200, 0), (40, 300, 1)])
def test_cpu_gpu_prior_neighborhood_distance_matches(D: int, S: int, seed: int):
    """
    CPU and GPU prior samplers should generate particles that are at similar *distance*
    from the truth graph on average (not identical particles).
    """
    rng = np.random.default_rng(seed)

    # Build a truth DAG similar to your pipeline: random upper-triangular weights
    W_star = np.zeros((D, D), dtype=float)
    tri = np.triu(rng.normal(0.0, 1.0, size=(D, D)), k=1)
    mask = rng.random((D, D)) < 0.08
    W_star[mask] = tri[mask]
    np.fill_diagonal(W_star, 0.0)

    # Hyperparams: choose values you actually use
    flip_prob = 0.10
    add_remove_prob = 0.05
    weight_noise = 0.10

    # --- CPU particles ---
    cpu_rng = np.random.default_rng(seed + 123)  # separate stream for the sampler
    cpu_particles = prior_cpu.make_prior_particles_from_truth(
        W_star=W_star,
        S=S,
        flip_prob=flip_prob,
        add_remove_prob=add_remove_prob,
        weight_noise=weight_noise,
        rng=cpu_rng,
    )
    assert isinstance(cpu_particles, list) and len(cpu_particles) == S

    # --- GPU particles ---
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed + 123)  # match the "sampler seed" conceptually

    gpu_particles = prior_gpu.make_prior_particles_from_truth(
        W_star=torch.tensor(W_star, dtype=torch.float64),
        S=S,
        flip_prob=flip_prob,
        add_remove_prob=add_remove_prob,
        weight_noise=weight_noise,
        device=torch.device("cpu"),  # keep test CPU-only; GPU not required
        dtype=torch.float64,
        generator=torch_gen,
        extra_edge_prob=None,  # keep default behavior unless you intentionally tune
    )
    assert isinstance(gpu_particles, torch.Tensor)
    assert gpu_particles.shape == (S, D, D)

    # --- Distances to truth ---
    A_true = adj_from_weights_np(W_star)

    shd_cpu = np.empty(S, dtype=float)
    for s in range(S):
        A = adj_from_weights_np(cpu_particles[s])
        shd_cpu[s] = shd_directed_flip1(A, A_true)

    # GPU adjacency + compute SHD on CPU for clarity
    A_gpu = adj_from_weights_torch(gpu_particles).cpu().numpy()

    shd_gpu = np.empty(S, dtype=float)
    for s in range(S):
        shd_gpu[s] = shd_directed_flip1(A_gpu[s], A_true)

    # --- Compare distributions ---
    mean_cpu = shd_cpu.mean()
    mean_gpu = shd_gpu.mean()

    q_cpu = np.quantile(shd_cpu, [0.1, 0.5, 0.9])
    q_gpu = np.quantile(shd_gpu, [0.1, 0.5, 0.9])

    # Tolerances: adjust to what “roughly the same neighborhood” means for you.
    # These are intentionally not super tight.
    mean_tol = max(2.0, 0.20 * max(mean_cpu, 1.0))  # within 20% or 2 SHD
    quant_tol = max(3.0, 0.25 * max(q_cpu[1], 1.0))  # within 25% of median or 3

    assert abs(mean_cpu - mean_gpu) <= mean_tol, (
        f"Mean SHD mismatch too large: cpu={mean_cpu:.3f}, gpu={mean_gpu:.3f}, tol={mean_tol:.3f}\n"
        f"CPU quantiles={q_cpu}, GPU quantiles={q_gpu}"
    )

    # Also check median & tails are not wildly different
    assert np.all(np.abs(q_cpu - q_gpu) <= quant_tol), (
        f"Quantiles mismatch too large.\nCPU={q_cpu}\nGPU={q_gpu}\nTol={quant_tol}"
    )
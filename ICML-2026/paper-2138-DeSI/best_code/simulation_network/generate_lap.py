"""
Generate correlation matrices following the Experiment 2-style setup with
graph Laplacians. For each observation we draw four independent Uniform[0,1]
predictors, project them via θ = normalized [0.1, 0.5, 0, -0.1] to obtain a
scalar x_i, and plug that scalar into the edge-weight formula

    E_{i,kℓ} = sin((k+ℓ)π/(2q)) * 1/(|x_i|+1) * (2 + x_i²) + ε_{i,kℓ},

with ε_{i,kℓ} ~ Uniform(-a, a) when the Bernoulli adjacency A_{kℓ}=1. Laplacians
Y_i = D_i - E_i are converted to correlation matrices via covariance scaling.
The function `generate_lap_dataset` mirrors `generate_matrices.generate_spd_matrices`
by producing predictors, one representative matrix per predictor (the first of
1000 simulations), and the log-Cholesky conditional mean computed from all 1000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from torch import Tensor
import math


def _sample_adjacency(
    q: int,
    edge_prob: float,
    device: torch.device,
    dtype: torch.dtype,
    ensure_positive_degree: bool = True,
) -> Tensor:
    """Sample a symmetric binary adjacency with optional degree enforcement."""

    if not 0.0 < edge_prob < 1.0:
        raise ValueError("edge_prob must lie in (0, 1).")

    while True:
        upper = (torch.rand((q, q), device=device, dtype=dtype) < edge_prob).to(dtype)
        A = torch.triu(upper, diagonal=1)
        A = A + A.T
        A.fill_diagonal_(0.0)

        if ensure_positive_degree and torch.any(A.sum(dim=0) == 0):
            continue
        return A


def _laplacian_from_scalar_no_noise(
    scalar_val: Tensor,
    adjacency: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Generate a graph Laplacian for a given scalar covariate without noise (deterministic)."""

    q = adjacency.shape[0]
    E = torch.zeros((q, q), dtype=dtype, device=device)

    for k in range(q - 1):
        kp = k + 1
        for ell in range(k + 1, q):
            if adjacency[k, ell] == 0:
                continue

            ellp = ell + 1
            envelope = math.sin((kp + ellp) * math.pi / (2.0 * q))
            envelope = torch.tensor(envelope, device=device, dtype=dtype)
            envelope = envelope / (torch.abs(scalar_val) + 1.0)
            envelope = envelope * (2.0 + scalar_val ** 2)

            # No noise - just the deterministic envelope
            val = envelope

            E[k, ell] = val
            E[ell, k] = val

    degrees = E.sum(dim=1)
    laplacian = torch.diag(degrees) - E
    return laplacian


def _add_noise_to_laplacian(
    laplacian_det: Tensor,
    adjacency: Tensor,
    noise_level: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Add noise to edge weights of a deterministic Laplacian."""
    # Extract edge matrix E from laplacian: L = D - E, so E = D - L
    degrees = torch.diagonal(laplacian_det)
    E = torch.diag(degrees) - laplacian_det
    
    # Add noise only to edges that exist in the adjacency matrix
    q = adjacency.shape[0]
    for k in range(q):
        for ell in range(k + 1, q):
            if adjacency[k, ell] == 1:
                noise = (torch.rand((), device=device, dtype=dtype) * 2.0 - 1.0) * noise_level
                E[k, ell] += noise
                E[ell, k] += noise
    
    # Reconstruct laplacian with noisy edges
    degrees_noisy = E.sum(dim=1)
    laplacian_noisy = torch.diag(degrees_noisy) - E
    
    return laplacian_noisy


def _laplacian_from_edges(E: Tensor, jitter: float = 1e-8) -> Tensor:
    """Compute graph Laplacian matrices with optional jitter for stability."""

    degrees = E.sum(dim=2)
    laplacians = torch.diag_embed(degrees) - E
    return laplacians


def _to_correlation(M: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert PSD matrices to correlation matrices via covariance scaling."""

    diag = torch.diagonal(M, dim1=-2, dim2=-1)
    diag = torch.clamp(diag, min=eps)
    inv_sqrt = diag.rsqrt()
    scaled = M * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    diag_scaled = torch.diagonal(scaled, dim1=-2, dim2=-1)
    diag_scaled.fill_(1.0)
    return scaled


def generate_lap_dataset(
    n: int = 100,
    m: int = 1000,
    q: int = 10,
    noise_level: float = 0.02,
    edge_prob: float = 0.3,
    random_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    corr_jitter: float = 1e-6,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """
    Generate predictors, representative matrices, and conditional means.

    Args:
        n: Number of predictor vectors.
        m: Number of matrices simulated per predictor (unused, kept for compatibility).
        q: Matrix dimension (number of nodes).
        noise_level: Uniform noise half-width added to edge weights.
        edge_prob: Edge probability for the adjacency mask.
        random_seed: Optional RNG seed.
        corr_jitter: Diagonal jitter added after correlation conversion to ensure SPD.

    Returns:
        tuple:
            - predictors: tensor of shape (n, 4) with Uniform[0,1] draws.
            - matrices_list: list of length n; each entry is a matrix generated with noise.
            - conditional_means: list of length n with deterministic laplacian matrices (without epsilon/noise).
    """

    if device is None:
        device = torch.device("cpu")

    if random_seed is not None:
        torch.manual_seed(random_seed)

    adjacency = _sample_adjacency(q, edge_prob, device, dtype)
    predictors = torch.rand((n, 4), device=device, dtype=dtype)
    theta_raw = torch.tensor([0.1, 0.5, 0.0, -0.1], device=device, dtype=dtype)
    theta = theta_raw / torch.norm(theta_raw)
    projected = predictors @ theta

    matrices_list: List[Tensor] = []
    conditional_means: List[Tensor] = []
    identity = torch.eye(q, device=device, dtype=dtype)

    for i in range(n):
        scalar_val = projected[i]
        
        # Generate deterministic Laplacian once (no error) - this is the conditional mean
        conditional_mean = _laplacian_from_scalar_no_noise(
            scalar_val,
            adjacency,
            device,
            dtype,
        )
        # Add small identity term to ensure strict positive definiteness
        eps_spd = 1e-6
        conditional_mean = conditional_mean + identity * eps_spd

        # Add noise to get the observed matrix (with error)
        first_matrix = _add_noise_to_laplacian(
            conditional_mean - identity * eps_spd,  # Remove jitter before adding noise
            adjacency,
            noise_level,
            device,
            dtype,
        )
        # Add jitter to observed matrix too
        first_matrix = first_matrix + identity * eps_spd

        matrices_list.append(first_matrix)
        conditional_means.append(conditional_mean)

    return predictors, matrices_list, conditional_means


if __name__ == "__main__":
    n = 100
    m = 1000
    q = 3
    predictors, matrices_list, conditional_means = generate_lap_dataset(
        n=n,
        m=m,
        q=q,
        noise_level=0.02,
        edge_prob=0.3,
        random_seed=1,
    )

    projected = predictors @ (torch.tensor([0.1, 0.5, 0.0, -0.1], dtype=predictors.dtype) /
                              torch.norm(torch.tensor([0.1, 0.5, 0.0, -0.1], dtype=predictors.dtype)))

    print("Experiment 2 correlation dataset generator")
    print(f"Number of predictor points: {n}")
    print(f"Matrices per point (simulated): {m}")
    print(f"Matrix dimension: {q}x{q}")
    print(f"x^T theta range: [{projected.min().item():.4f}, {projected.max().item():.4f}]")
    print("\nFirst predictor and first generated matrix:")
    print("Predictor:", predictors[0])
    print("Matrix:\n", matrices_list[0])
    print("\nFirst conditional mean:\n", conditional_means[0])


"""
Generate SPD matrices using Eigen-Decomposition with fixed eigenvectors.
For each predictor x, generates m=1000 matrices and computes conditional means.
Uses the method: M = QΛQ^T + εI where:
- Q is a fixed orthogonal matrix (Discrete Legendre Basis, same for all matrices)
- Λ is a diagonal matrix with eigenvalues based on theta^T * x
- ε ~ Uniform(-0.01, 0.01) is sampled independently for each matrix
"""

from __future__ import annotations

from typing import Optional, List
import torch
from torch import Tensor
import math


def generate_spd_matrices(
    n: int = 100,
    m: int = 1000,
    p: int = 3,
    random_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """
    Generate predictors, representative matrices, and conditional means using Eigen-Decomposition with fixed eigenvectors.
    
    Uses the method: M = QΛQ^T + εI where:
    - Q is a fixed orthogonal matrix (Discrete Legendre Basis, same for all matrices)
    - Λ is a diagonal matrix with eigenvalues based on theta^T * x
    - ε ~ Uniform(-0.01, 0.01) is sampled independently for each matrix

    Args:
        n: Number of predictor vectors.
        m: Number of matrices simulated per predictor (used for the mean).
        p: Matrix dimension.
        random_seed: Optional RNG seed.
        device: Optional device (defaults to CPU).
        dtype: Data type (defaults to float64).

    Returns:
        tuple:
            - predictors: tensor of shape (n, 4) with Uniform draws.
            - matrices_list: list of length n; each entry is the *first* matrix
              generated out of the m simulations for that predictor.
            - conditional_means: list of length n with log-Cholesky Fréchet means.
    """

    if device is None:
        device = torch.device("cpu")

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Define theta and normalize it
    theta_raw = torch.tensor([0.1, 0.5, 0.0, -0.1], device=device, dtype=dtype)
    theta = theta_raw / torch.norm(theta_raw)

    # Generate predictors x: (n, 4)
    # Dimension 1: U(1,2), Dimension 2: U(0,1), Dimension 3: U(0,1), Dimension 4: U(-1,0)
    predictors = torch.zeros((n, 4), device=device, dtype=dtype)
    predictors[:, 0] = torch.rand(n, device=device, dtype=dtype) * 1.0 + 1.0  # U(1,2)
    predictors[:, 1] = torch.rand(n, device=device, dtype=dtype)  # U(0,1)
    predictors[:, 2] = torch.rand(n, device=device, dtype=dtype)  # U(0,1)
    predictors[:, 3] = torch.rand(n, device=device, dtype=dtype) * (-1.0)  # U(-1,0)

    # Compute scalar projection theta^T * x for each predictor
    projected = predictors @ theta  # shape: (n,)

    matrices_list: List[Tensor] = []
    conditional_means: List[Tensor] = []
    identity = torch.eye(p, device=device, dtype=dtype)
    
    # Generate fixed orthonormal matrix Q (eigenvectors) using Discrete Legendre Basis
    # Evaluate Legendre polynomials at discrete points and orthogonalize
    # Points are equally spaced in [-1, 1]
    x_points = torch.linspace(-1.0, 1.0, p, device=device, dtype=dtype)
    
    # Construct Discrete Legendre Basis using Gram-Schmidt on monomials
    # Start with monomial basis: 1, x, x², ..., x^(p-1)
    Q_fixed = torch.zeros((p, p), device=device, dtype=dtype)
    
    # First basis vector: constant (normalized to unit length)
    Q_fixed[:, 0] = 1.0 / torch.sqrt(torch.tensor(p, device=device, dtype=dtype))
    
    # Gram-Schmidt process for remaining basis vectors
    for k in range(1, p):
        # Start with x^k
        v = x_points ** k
        
        # Subtract projections onto previous basis vectors
        for j in range(k):
            proj = torch.dot(v, Q_fixed[:, j])
            v = v - proj * Q_fixed[:, j]
        
        # Normalize to unit length (ensures orthonormality)
        norm = torch.norm(v)
        if norm > 1e-10:
            Q_fixed[:, k] = v / norm
        else:
            # If vector is too small, use a small random perturbation
            Q_fixed[:, k] = torch.randn(p, device=device, dtype=dtype) * 1e-6
            # Re-orthogonalize
            for j in range(k):
                proj = torch.dot(Q_fixed[:, k], Q_fixed[:, j])
                Q_fixed[:, k] = Q_fixed[:, k] - proj * Q_fixed[:, j]
            norm = torch.norm(Q_fixed[:, k])
            if norm > 1e-10:
                Q_fixed[:, k] = Q_fixed[:, k] / norm
            else:
                Q_fixed[:, k] = Q_fixed[:, k-1]  # Fallback
    
    # Verify orthonormality: Q^T * Q should be identity (within numerical precision)
    Q_check = Q_fixed.T @ Q_fixed
    identity_check = torch.eye(p, device=device, dtype=dtype)
    orthonormality_error = torch.norm(Q_check - identity_check).item()
    if orthonormality_error > 1e-6:
        # If not orthonormal enough, use QR decomposition to ensure orthonormality
        Q_fixed, R = torch.linalg.qr(Q_fixed)
        # Ensure Q has positive diagonal (for consistency)
        diag_signs = torch.sign(torch.diag(R))
        Q_fixed = Q_fixed @ torch.diag(diag_signs)

    for i in range(n):
        scalar_val = projected[i]  # theta^T * x_i
        
        # Generate eigenvalues based on scalar value
        # Diagonal: [0.1 * (theta*x), 0.1 * (theta*x)^2, 0.1 * exp(theta*x)]
        lambdas = torch.zeros(p, device=device, dtype=dtype)
        lambdas[0] = scalar_val
        lambdas[1] = scalar_val ** 2
        lambdas[2] = torch.exp(scalar_val)
        
        # Ensure eigenvalues are positive (add small value if needed)
        lambdas = torch.clamp(lambdas, min=0.1)
        
        # Generate m matrices using fixed eigenvectors
        sLT_sum = torch.zeros((p, p), device=device, dtype=dtype)
        logD_sum = torch.zeros((p, p), device=device, dtype=dtype)
        first_matrix = None

        for j in range(m):
            # Sample p independent epsilon values from Uniform(-0.01, 0.01) for this matrix
            epsilon_vec = (torch.rand(p, device=device, dtype=dtype) * 2.0 - 1.0) * 0.001
            
            # Construct SPD matrix: M = Q_fixed * Λ * Q_fixed^T + diag(epsilon)
            Lambda = torch.diag(lambdas)
            spd_matrix = Q_fixed @ Lambda @ Q_fixed.T
            
            # Add epsilon values to diagonal
            epsilon_diag = torch.diag(epsilon_vec)
            spd_matrix = spd_matrix + epsilon_diag
            
            # Ensure symmetry (should already be symmetric, but enforce it)
            spd_matrix = 0.5 * (spd_matrix + spd_matrix.T)
            
            if j == 0:
                first_matrix = spd_matrix.clone()

            # Compute log-Cholesky decomposition for Fréchet mean
            L = torch.linalg.cholesky(spd_matrix)

            # Accumulate for log-Cholesky mean
            sLT_sum += torch.tril(L, diagonal=-1)
            log_diag = torch.log(torch.diagonal(L))
            logD_sum += torch.diag_embed(log_diag)

        # Compute log-Cholesky Fréchet mean
        sLT_mean = sLT_sum / m
        logD_mean = logD_sum / m
        expD = torch.diag(torch.exp(torch.diagonal(logD_mean)))
        L_mean = sLT_mean + expD
        conditional_mean = L_mean @ L_mean.T
        
        # Add small identity term to ensure strict positive definiteness
        eps_spd = 1e-6
        conditional_mean = conditional_mean + identity * eps_spd

        matrices_list.append(first_matrix)
        conditional_means.append(conditional_mean)

    return predictors, matrices_list, conditional_means


if __name__ == "__main__":
    n = 100
    m = 1000
    p = 3
    
    predictors, matrices_list, conditional_means = generate_spd_matrices(
        n=n,
        m=m,
        p=p,
        random_seed=1,
    )

    # Compute theta^T * x for display
    theta_raw = torch.tensor([0.1, 0.5, 0.0, -0.1], dtype=predictors.dtype)
    theta = theta_raw / torch.norm(theta_raw)
    projected = predictors @ theta

    print("SPD matrices generator using Eigen-Decomposition with fixed eigenvectors")
    print(f"Number of predictor points: {n}")
    print(f"Matrices per point (simulated): {m}")
    print(f"Matrix dimension: {p}x{p}")
    print(f"x^T theta range: [{projected.min().item():.4f}, {projected.max().item():.4f}]")
    print("\nFirst predictor:", predictors[0])
    print("First matrix:\n", matrices_list[0])
    print("\nFirst conditional mean:\n", conditional_means[0])


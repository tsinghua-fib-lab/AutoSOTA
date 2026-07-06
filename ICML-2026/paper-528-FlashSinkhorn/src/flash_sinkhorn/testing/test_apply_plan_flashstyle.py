"""Tests for FlashStyle apply_plan_mat kernel with shifted potentials.

These tests verify that the apply_plan_mat_flashstyle and apply_plan_vec_flashstyle
kernels produce correct results by comparing against dense reference computations.

Test strategy:
1. Dense reference comparison: P @ V where P is explicitly materialized
2. Marginal constraint: P @ ones approx a for converged potentials
3. Parity between different kernel configurations (exp/exp2, autotune, etc.)
"""

import pytest
import torch

from flash_sinkhorn.kernels.sinkhorn_triton_apply_sqeuclid import (
    apply_plan_mat_flashstyle,
    apply_plan_vec_flashstyle,
)
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_symmetric,
    shifted_to_standard_potentials,
    standard_to_shifted_potentials,
)
from flash_sinkhorn.kernels._common import log_weights


def _rand_inputs(n, m, d, device, seed=0):
    """Generate random test inputs."""
    torch.manual_seed(seed)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    # Uniform marginals (ensure positive, normalized)
    a = torch.ones(n, device=device, dtype=torch.float32) / n
    b = torch.ones(m, device=device, dtype=torch.float32) / m
    return x, y, a, b


def _compute_transport_plan_dense(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,  # Standard (OTT) potential
    g: torch.Tensor,  # Standard (OTT) potential
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cost_scale: float = 1.0,
) -> torch.Tensor:
    """Compute dense transport plan P = a * b * exp((f + g - C) / eps).

    This is the reference implementation for verifying streaming kernels.
    """
    # Cost matrix C_ij = cost_scale * ||x_i - y_j||^2
    x2 = (x ** 2).sum(dim=1)  # [n]
    y2 = (y ** 2).sum(dim=1)  # [m]
    xy = x @ y.T  # [n, m]
    C = cost_scale * (x2[:, None] + y2[None, :] - 2 * xy)

    # Transport plan P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
    log_P = (f[:, None] + g[None, :] - C) / eps
    P = a[:, None] * b[None, :] * torch.exp(log_P)
    return P


def _get_shifted_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cost_scale: float = 1.0,
    n_iters: int = 50,
) -> tuple:
    """Get converged shifted potentials using FlashSinkhorn symmetric solver.

    Returns:
        f_hat: Shifted f potential [n]
        g_hat: Shifted g potential [m]
        log_a: Log source weights [n]
        log_b: Log target weights [m]
        alpha: Source squared norms [n]
        beta: Target squared norms [m]
    """
    # Compute shifted potential components
    alpha = cost_scale * (x.float() ** 2).sum(dim=1)
    beta = cost_scale * (y.float() ** 2).sum(dim=1)
    log_a = log_weights(a)
    log_b = log_weights(b)

    # Use symmetric solver (GeomLoss-style) - returns STANDARD potentials
    f, g = sinkhorn_flashstyle_symmetric(
        x, y, a, b,
        cost_scale=cost_scale,
        eps=eps,
        n_iters=n_iters,
        use_epsilon_scaling=False,  # Fixed eps for testing
        use_exp2=True,
        allow_tf32=False,
    )

    # Convert back to shifted potentials
    f_hat, g_hat = standard_to_shifted_potentials(f, g, alpha, beta)

    return f_hat, g_hat, log_a, log_b, alpha, beta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("n,m,d", [
    (32, 32, 16),
    (64, 48, 32),
    (100, 80, 64),
    (128, 96, 64),
])
def test_flashstyle_matches_dense(axis, n, m, d):
    """Verify FlashStyle kernel matches dense P @ V computation.

    FlashStyle computes P_ij = a_i * b_j * exp((f+g-C)/eps) (GeomLoss convention).
    Compare against explicit P @ V where P is materialized.
    """
    device = torch.device("cuda")
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    # Convert to standard potentials for dense reference
    f, g = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    # Compute dense transport plan
    P = _compute_transport_plan_dense(x, y, f, g, a, b, eps, cost_scale)

    # Matrix V to multiply
    if axis == 1:
        V = torch.randn(m, d, device=device, dtype=torch.float32)
        out_dense = P @ V  # [n, d]
    else:
        V = torch.randn(n, d, device=device, dtype=torch.float32)
        out_dense = P.T @ V  # [m, d]

    # FlashStyle implementation (shifted potentials directly)
    out_flash = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    # Check parity (allow slightly looser tolerance for numerical differences)
    torch.testing.assert_close(out_flash, out_dense, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("n,m,d", [
    (32, 32, 16),
    (64, 48, 32),
    (100, 80, 64),
])
def test_flashstyle_marginal_constraint(n, m, d):
    """Verify P @ ones ≈ a (row marginal constraint) for converged potentials."""
    device = torch.device("cuda")
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials (more iterations for convergence)
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=100
    )

    # P @ ones should give row marginal a
    ones_m = torch.ones(m, d, device=device, dtype=torch.float32)
    row_marginal = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, ones_m,
        eps=eps,
        axis=1,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )
    # Sum over d dimension to get marginal
    row_marginal_sum = row_marginal.sum(dim=1) / d  # Normalize by d since we used ones matrix

    # Check row marginal matches a (use looser tolerance for marginal check)
    torch.testing.assert_close(row_marginal_sum, a, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("cost_scale", [0.5, 1.0])
def test_flashstyle_half_cost_parity(cost_scale):
    """Verify FlashStyle works with both full and half cost scaling."""
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    # Convert to standard potentials for reference
    f, g = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    # Dense reference
    P = _compute_transport_plan_dense(x, y, f, g, a, b, eps, cost_scale)
    V = torch.randn(m, d, device=device, dtype=torch.float32)
    out_dense = P @ V

    # FlashStyle implementation
    out_flash = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=1,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    torch.testing.assert_close(out_flash, out_dense, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_flashstyle_with_scale():
    """Verify FlashStyle works with optional scale parameter for axis=1."""
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    # Matrix V and scale
    V = torch.randn(m, d, device=device, dtype=torch.float32)
    scale = torch.rand(m, device=device, dtype=torch.float32) + 0.5  # Random positive scale

    # With scale: should be equivalent to V * scale[:, None]
    out_with_scale = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=1,
        cost_scale=cost_scale,
        scale=scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    # Reference: V * scale[:, None] then apply
    V_scaled = V * scale[:, None]
    out_reference = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V_scaled,
        eps=eps,
        axis=1,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    torch.testing.assert_close(out_with_scale, out_reference, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_exp_vs_exp2_parity(axis):
    """Verify exp and exp2 implementations give identical results."""
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    if axis == 1:
        V = torch.randn(m, d, device=device, dtype=torch.float32)
    else:
        V = torch.randn(n, d, device=device, dtype=torch.float32)

    # exp2 version
    out_exp2 = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    # exp version
    out_exp = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=False,
        allow_tf32=False,
        autotune=False,
    )

    torch.testing.assert_close(out_exp2, out_exp, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_autotune_parity(axis):
    """Verify autotuned and manual kernels give identical results."""
    device = torch.device("cuda")
    n, m, d = 128, 96, 64
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    if axis == 1:
        V = torch.randn(m, d, device=device, dtype=torch.float32)
    else:
        V = torch.randn(n, d, device=device, dtype=torch.float32)

    # Manual blocks
    out_manual = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        block_m=32,
        block_n=32,
        block_k=16,
        block_d=16,
        use_exp2=True,
        allow_tf32=False,
        autotune=False,
    )

    # Autotuned
    out_autotune = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, V,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
        autotune=True,
    )

    torch.testing.assert_close(out_manual, out_autotune, rtol=1e-4, atol=1e-4)


# =============================================================================
# FlashStyle Vec Kernel Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_vec_matches_dense(axis):
    """Verify FlashStyle vec kernel matches dense P @ vec computation."""
    device = torch.device("cuda")
    n, m, d = 32, 24, 16
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 0.5

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    # Convert to standard potentials for dense computation
    f, g = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    # Compute dense transport plan
    P = _compute_transport_plan_dense(x, y, f, g, a, b, eps, cost_scale)

    if axis == 1:
        vec = torch.randn(m, device=device, dtype=torch.float32)
        out_dense = P @ vec
    else:
        vec = torch.randn(n, device=device, dtype=torch.float32)
        out_dense = P.T @ vec

    # FlashStyle vec kernel
    out_flash = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=True,
        allow_tf32=False,
    )

    torch.testing.assert_close(out_flash, out_dense, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_vec_matches_mat_first_column(axis):
    """Verify vec kernel matches first column of mat kernel output.

    The mat kernel requires mat.shape[-1] == d (feature dimension).
    So we create a full [m, d] or [n, d] matrix and compare the first column.
    """
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    if axis == 1:
        vec = torch.randn(m, device=device, dtype=torch.float32)
        # Mat kernel requires D=d, so create [m, d] matrix with vec in first column
        mat = torch.zeros(m, d, device=device, dtype=torch.float32)
        mat[:, 0] = vec
    else:
        vec = torch.randn(n, device=device, dtype=torch.float32)
        mat = torch.zeros(n, d, device=device, dtype=torch.float32)
        mat[:, 0] = vec

    # Mat kernel
    out_mat = apply_plan_mat_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, mat,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        autotune=False,
    )[:, 0]  # First column

    # Vec kernel
    out_vec = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
    )

    torch.testing.assert_close(out_vec, out_mat, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_vec_matches_dense_extended(axis):
    """Verify FlashStyle vec matches dense P @ vec computation.

    FlashStyle uses f_hat, g_hat where P = a * b * exp((f + g - C) / eps)
    Compare against explicit dense computation.
    """
    device = torch.device("cuda")
    n, m, d = 48, 40, 24
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 0.5

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    # Convert to standard potentials for dense computation
    f_std, g_std = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

    # Compute dense transport plan
    P = _compute_transport_plan_dense(x, y, f_std, g_std, a, b, eps, cost_scale)

    if axis == 1:
        vec = torch.randn(m, device=device, dtype=torch.float32)
        out_dense = P @ vec
    else:
        vec = torch.randn(n, device=device, dtype=torch.float32)
        out_dense = P.T @ vec

    # FlashStyle vec kernel
    out_flash = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
    )

    torch.testing.assert_close(out_flash, out_dense, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_flashstyle_vec_marginal_constraint():
    """P @ ones should approximately equal a (source marginal)."""
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.5  # Larger eps for easier convergence
    cost_scale = 1.0

    # Get converged shifted potentials
    f_hat, g_hat, log_a, log_b, alpha, beta = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=200  # More iterations for better convergence
    )

    # P @ ones should equal a
    ones_m = torch.ones(m, device=device, dtype=torch.float32)
    row_sums = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, ones_m,
        eps=eps,
        axis=1,
        cost_scale=cost_scale,
    )

    # Marginal constraint depends on Sinkhorn convergence - use relaxed tolerance
    torch.testing.assert_close(row_sums, a, rtol=5e-2, atol=5e-3)

    # P^T @ ones should equal b
    ones_n = torch.ones(n, device=device, dtype=torch.float32)
    col_sums = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, ones_n,
        eps=eps,
        axis=0,
        cost_scale=cost_scale,
    )

    torch.testing.assert_close(col_sums, b, rtol=5e-2, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_flashstyle_vec_exp2_vs_exp(axis):
    """Verify exp2 and exp implementations match."""
    device = torch.device("cuda")
    n, m, d = 48, 40, 24
    x, y, a, b = _rand_inputs(n, m, d, device)
    eps = 0.25
    cost_scale = 0.5

    f_hat, g_hat, log_a, log_b, _, _ = _get_shifted_potentials(
        x, y, a, b, eps, cost_scale, n_iters=50
    )

    if axis == 1:
        vec = torch.randn(m, device=device, dtype=torch.float32)
    else:
        vec = torch.randn(n, device=device, dtype=torch.float32)

    out_exp2 = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=True,
    )

    out_exp = apply_plan_vec_flashstyle(
        x, y, f_hat, g_hat, log_a, log_b, vec,
        eps=eps,
        axis=axis,
        cost_scale=cost_scale,
        use_exp2=False,
    )

    torch.testing.assert_close(out_exp2, out_exp, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

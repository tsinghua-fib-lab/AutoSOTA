"""Parity tests for FlashSinkhorn vs current implementation.

These tests verify that the FlashSinkhorn reformulation produces identical
results to the existing Sinkhorn implementations while using the more efficient
shifted potential formulation.
"""

import math
import pytest
import torch
from torch.testing import assert_close

from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    precompute_flashsinkhorn_inputs,
    compute_bias_f,
    compute_bias_g,
    flashsinkhorn_lse,
    shifted_to_standard_potentials,
    standard_to_shifted_potentials,
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)
from flash_sinkhorn.kernels.sinkhorn_triton_geomloss_sqeuclid import (
    sinkhorn_geomloss_online_potentials_sqeuclid,
)
from flash_sinkhorn.kernels.sinkhorn_triton_ott_sqeuclid import (
    sinkhorn_potentials_sqeuclid,
)
from flash_sinkhorn.kernels._common import log_weights


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def setup_small(device):
    """Small test case for debugging."""
    torch.manual_seed(42)
    n, m, d = 100, 120, 32
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m
    return x, y, a, b


@pytest.fixture
def setup_medium(device):
    """Medium test case for typical usage."""
    torch.manual_seed(42)
    n, m, d = 1000, 1000, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m
    return x, y, a, b


@pytest.fixture
def setup_asymmetric(device):
    """Asymmetric test case (n != m)."""
    torch.manual_seed(42)
    n, m, d = 500, 800, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m
    return x, y, a, b


# =============================================================================
# UNIT TESTS: Mathematical Correctness
# =============================================================================

class TestPrecomputation:
    """Test precomputation utilities."""

    def test_alpha_beta_values(self, setup_small):
        """Verify alpha = cost_scale * ||x||² and beta = cost_scale * ||y||²."""
        x, y, a, b = setup_small
        eps = 0.1

        for cost_scale in [1.0, 0.5]:
            alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
                x, y, a, b, eps, cost_scale=cost_scale
            )

            expected_alpha = cost_scale * (x ** 2).sum(dim=1)
            expected_beta = cost_scale * (y ** 2).sum(dim=1)

            assert_close(alpha, expected_alpha, rtol=1e-5, atol=1e-5)
            assert_close(beta, expected_beta, rtol=1e-5, atol=1e-5)

    def test_gamma_delta_values(self, setup_small):
        """Verify gamma = eps*log(a) and delta = eps*log(b)."""
        x, y, a, b = setup_small
        eps = 0.1

        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale=1.0
        )

        expected_gamma = eps * log_weights(a)
        expected_delta = eps * log_weights(b)

        assert_close(gamma, expected_gamma, rtol=1e-5, atol=1e-5)
        assert_close(delta, expected_delta, rtol=1e-5, atol=1e-5)


class TestShiftedPotentials:
    """Test shifted potential conversions."""

    def test_roundtrip_conversion(self, setup_small):
        """Test that standard -> shifted -> standard is identity."""
        x, y, a, b = setup_small
        eps = 0.1

        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale=1.0
        )

        # Create some random potentials
        f = torch.randn(x.shape[0], device=x.device)
        g = torch.randn(y.shape[0], device=y.device)

        # Convert to shifted and back
        f_hat, g_hat = standard_to_shifted_potentials(f, g, alpha, beta)
        f_back, g_back = shifted_to_standard_potentials(f_hat, g_hat, alpha, beta)

        assert_close(f, f_back, rtol=1e-5, atol=1e-5)
        assert_close(g, g_back, rtol=1e-5, atol=1e-5)

    def test_shifted_values(self, setup_small):
        """Test shifted potential computation."""
        x, y, a, b = setup_small
        eps = 0.1

        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale=1.0
        )

        f = torch.randn(x.shape[0], device=x.device)
        g = torch.randn(y.shape[0], device=y.device)

        f_hat, g_hat = standard_to_shifted_potentials(f, g, alpha, beta)

        assert_close(f_hat, f - alpha, rtol=1e-5, atol=1e-5)
        assert_close(g_hat, g - beta, rtol=1e-5, atol=1e-5)


class TestBiasComputation:
    """Test bias computation functions."""

    def test_bias_f_formula(self, setup_small):
        """Test that bias_f = (g - beta + delta) / eps."""
        x, y, a, b = setup_small
        eps = 0.1

        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale=1.0
        )

        g = torch.randn(y.shape[0], device=y.device)
        u = compute_bias_f(g, beta, delta, eps)

        expected = (g - beta + delta) / eps
        assert_close(u, expected, rtol=1e-5, atol=1e-5)

    def test_bias_g_formula(self, setup_small):
        """Test that bias_g = (f - alpha + gamma) / eps."""
        x, y, a, b = setup_small
        eps = 0.1

        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale=1.0
        )

        f = torch.randn(x.shape[0], device=x.device)
        v = compute_bias_g(f, alpha, gamma, eps)

        expected = (f - alpha + gamma) / eps
        assert_close(v, expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# INTEGRATION TESTS: Single Update Parity
# =============================================================================

class TestSingleUpdateParity:
    """Test that a single FlashSinkhorn update matches the standard/GeomLoss convention.

    NOTE: FlashSinkhorn uses the GeomLoss convention where log marginals are INSIDE
    the logsumexp. This differs from OTT-JAX which puts log marginals OUTSIDE.
    Both conventions produce equivalent transport plans (potentials differ by constant).

    GeomLoss: f = -ε · LSE_j[(g - C)/ε + log(b)]
    OTT-JAX: f = ε·log(a) - ε · LSE_j[(g - C)/ε]
    """

    def test_f_update_parity(self, setup_small):
        """Test single f-update matches GeomLoss/standard convention."""
        x, y, a, b = setup_small
        eps = 0.1
        cost_scale = 1.0

        # Initialize potentials
        n, m = x.shape[0], y.shape[0]
        f = torch.zeros(n, device=x.device, dtype=torch.float32)
        g = torch.zeros(m, device=x.device, dtype=torch.float32)

        # Standard/GeomLoss convention: log(b) INSIDE logsumexp
        x2 = (x ** 2).sum(dim=1) * cost_scale
        y2 = (y ** 2).sum(dim=1) * cost_scale
        logb = log_weights(b)

        # Compute cost matrix (for reference)
        C = x2[:, None] + y2[None, :] - 2.0 * cost_scale * (x @ y.T)

        # Standard f-update: f_i = -eps * LSE_j[(g_j - C_ij)/eps + log(b_j)]
        logits = (g[None, :] - C) / eps + logb[None, :]
        f_expected = -eps * torch.logsumexp(logits, dim=1)

        # FlashSinkhorn f-update (using raw x, y with cost_scale)
        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale
        )
        u = compute_bias_f(g, beta, delta, eps)
        f_hat_flash = flashsinkhorn_lse(
            x, y, u, eps, cost_scale=cost_scale, autotune=False, allow_tf32=False
        )
        f_flash = f_hat_flash + alpha  # Convert from shifted to standard

        assert_close(f_flash, f_expected, rtol=1e-4, atol=1e-4)

    def test_g_update_parity(self, setup_small):
        """Test single g-update matches GeomLoss/standard convention."""
        x, y, a, b = setup_small
        eps = 0.1
        cost_scale = 1.0

        # Initialize potentials
        n, m = x.shape[0], y.shape[0]
        f = torch.randn(n, device=x.device, dtype=torch.float32)  # Non-zero f
        g = torch.zeros(m, device=x.device, dtype=torch.float32)

        # Standard/GeomLoss convention: log(a) INSIDE logsumexp
        x2 = (x ** 2).sum(dim=1) * cost_scale
        y2 = (y ** 2).sum(dim=1) * cost_scale
        loga = log_weights(a)

        # Compute cost matrix (for reference)
        C = x2[:, None] + y2[None, :] - 2.0 * cost_scale * (x @ y.T)

        # Standard g-update: g_j = -eps * LSE_i[(f_i - C_ij)/eps + log(a_i)]
        logits = (f[:, None] - C) / eps + loga[:, None]
        g_expected = -eps * torch.logsumexp(logits, dim=0)

        # FlashSinkhorn g-update (swap x and y for axis=0 reduction)
        alpha, beta, gamma, delta = precompute_flashsinkhorn_inputs(
            x, y, a, b, eps, cost_scale
        )
        v = compute_bias_g(f, alpha, gamma, eps)
        g_hat_flash = flashsinkhorn_lse(
            y, x, v, eps, cost_scale=cost_scale, autotune=False, allow_tf32=False
        )
        g_flash = g_hat_flash + beta  # Convert from shifted to standard

        assert_close(g_flash, g_expected, rtol=1e-4, atol=1e-4)


# =============================================================================
# SOLVER PARITY TESTS
# =============================================================================

class TestAlternatingSolverParity:
    """Test FlashSinkhorn alternating solver convergence and transport plan validity.

    NOTE: FlashSinkhorn uses the GeomLoss convention (log marginals inside logsumexp),
    while OTT-JAX uses a different convention (log marginals outside logsumexp).
    The transport plans are equivalent, but potentials differ by a constant.
    """

    def test_transport_plan_marginals(self, setup_small):
        """Test that FlashSinkhorn produces a valid transport plan."""
        x, y, a, b = setup_small
        eps = 0.1
        n_iters = 100

        # FlashSinkhorn
        f_flash, g_flash = sinkhorn_flashstyle_alternating(
            x, y, a, b,
            eps=eps, n_iters=n_iters, cost_scale=1.0,
            allow_tf32=False, use_exp2=True, autotune=False
        )

        # Compute transport plan P = exp((f + g - C) / eps)
        x2 = (x ** 2).sum(dim=1)
        y2 = (y ** 2).sum(dim=1)
        C = x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)

        # For GeomLoss convention: P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
        logb = log_weights(b)
        loga = log_weights(a)
        log_P = loga[:, None] + logb[None, :] + (f_flash[:, None] + g_flash[None, :] - C) / eps
        P = torch.exp(log_P)

        # Check marginals
        row_sum = P.sum(dim=1)
        col_sum = P.sum(dim=0)

        assert_close(row_sum, a, rtol=0.05, atol=0.05)
        assert_close(col_sum, b, rtol=0.05, atol=0.05)

    def test_half_cost_transport_plan(self, setup_small):
        """Test FlashSinkhorn with half cost produces valid transport plan."""
        x, y, a, b = setup_small
        eps = 0.1
        n_iters = 100
        cost_scale = 0.5

        # FlashSinkhorn with half cost
        f_flash, g_flash = sinkhorn_flashstyle_alternating(
            x, y, a, b,
            eps=eps, n_iters=n_iters, cost_scale=cost_scale,
            allow_tf32=False, use_exp2=True, autotune=False
        )

        # Compute transport plan with half cost
        x2 = cost_scale * (x ** 2).sum(dim=1)
        y2 = cost_scale * (y ** 2).sum(dim=1)
        C = x2[:, None] + y2[None, :] - 2 * cost_scale * (x @ y.T)

        # For GeomLoss convention
        logb = log_weights(b)
        loga = log_weights(a)
        log_P = loga[:, None] + logb[None, :] + (f_flash[:, None] + g_flash[None, :] - C) / eps
        P = torch.exp(log_P)
        row_sum = P.sum(dim=1)

        # Should be close to marginal a
        assert_close(row_sum, a, rtol=0.05, atol=0.05)

    def test_vs_ott_style_transport_plan(self, setup_small):
        """Test FlashSinkhorn alternating vs OTT-style produces same transport plan.

        The potential conventions differ:
        - OTT-style: f = ε·log(a) - ε·LSE_j[(g-C)/ε]  (log marginals OUTSIDE)
        - FlashSinkhorn: f = -ε·LSE_j[(g-C)/ε + log(b)]  (log marginals INSIDE)

        But both produce the SAME transport plan P, which is what matters.
        """
        x, y, a, b = setup_small
        eps = 0.1
        n_iters = 100

        loga = log_weights(a)
        logb = log_weights(b)

        # OTT-style implementation
        f_ott, g_ott = sinkhorn_potentials_sqeuclid(
            x, y, loga, logb, eps, n_iters,
            fused=True, allow_tf32=False
        )

        # FlashSinkhorn alternating (Gauss-Seidel, matches OTT-style)
        f_flash, g_flash = sinkhorn_flashstyle_alternating(
            x, y, a, b,
            eps=eps, n_iters=n_iters, cost_scale=1.0,
            allow_tf32=False, use_exp2=True, autotune=False
        )

        # Compute cost matrix
        x2 = (x ** 2).sum(dim=1)
        y2 = (y ** 2).sum(dim=1)
        C = x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)

        # OTT-style transport plan: P_ij = exp((f_i + g_j - C_ij) / eps)
        # (log marginals are absorbed into potentials)
        log_P_ott = (f_ott[:, None] + g_ott[None, :] - C) / eps
        P_ott = torch.exp(log_P_ott)

        # FlashSinkhorn transport plan (GeomLoss convention):
        # P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
        log_P_flash = loga[:, None] + logb[None, :] + (f_flash[:, None] + g_flash[None, :] - C) / eps
        P_flash = torch.exp(log_P_flash)

        # Transport plans should be very close
        # Note: Small differences expected due to different numerical paths
        # (shifted vs standard potentials, exp2 vs exp, etc.)
        assert_close(P_flash, P_ott, rtol=0.02, atol=0.01)

        # Both should have correct marginals (this is the key correctness check)
        assert_close(P_flash.sum(dim=1), a, rtol=0.02, atol=0.02)
        assert_close(P_ott.sum(dim=1), a, rtol=0.02, atol=0.02)

        # OT costs should match closely (computed via transport plan)
        C = x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)
        cost_flash = (P_flash * C).sum()
        cost_ott = (P_ott * C).sum()
        assert_close(cost_flash, cost_ott, rtol=0.01, atol=0.01)

    def test_vs_ott_style_cost(self, setup_small):
        """Test that FlashSinkhorn and OTT-style produce same OT cost."""
        x, y, a, b = setup_small
        eps = 0.1
        n_iters = 100

        loga = log_weights(a)
        logb = log_weights(b)

        # OTT-style
        f_ott, g_ott = sinkhorn_potentials_sqeuclid(
            x, y, loga, logb, eps, n_iters,
            fused=True, allow_tf32=False
        )

        # FlashSinkhorn (Gauss-Seidel, matches OTT-style)
        f_flash, g_flash = sinkhorn_flashstyle_alternating(
            x, y, a, b,
            eps=eps, n_iters=n_iters, cost_scale=1.0,
            allow_tf32=False, use_exp2=True, autotune=False
        )

        # OT cost = <a, f> + <b, g> (for both conventions, adjusted)
        # OTT: cost = a @ f_ott + b @ g_ott
        cost_ott = (a * f_ott).sum() + (b * g_ott).sum()

        # FlashSinkhorn (GeomLoss): cost = a @ f_flash + b @ g_flash + eps * <a, log(a)> + eps * <b, log(b)>
        # The extra entropy terms come from the convention difference
        entropy_a = eps * (a * loga).sum()
        entropy_b = eps * (b * logb).sum()
        cost_flash = (a * f_flash).sum() + (b * g_flash).sum() + entropy_a + entropy_b

        # Costs should match
        assert_close(cost_flash, cost_ott, rtol=1e-2, atol=1e-2)


class TestSymmetricSolverParity:
    """Test FlashSinkhorn symmetric solver vs current GeomLoss implementation."""

    def test_vs_current_geomloss_basic(self, setup_small):
        """Test FlashSinkhorn symmetric vs current GeomLoss implementation."""
        x, y, a, b = setup_small
        blur = 0.1  # eps = blur^2 = 0.01

        # Current implementation
        f_current, g_current = sinkhorn_geomloss_online_potentials_sqeuclid(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=1.0
        )

        # FlashSinkhorn
        f_flash, g_flash = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=1.0
        )

        # Check potentials are close
        # With raw x,y coordinates (not pre-scaled Q,K), parity is excellent
        assert_close(f_flash, f_current, rtol=1e-3, atol=1e-3)
        assert_close(g_flash, g_current, rtol=1e-3, atol=1e-3)

    def test_vs_current_geomloss_half_cost(self, setup_small):
        """Test FlashSinkhorn symmetric with half cost (GeomLoss default)."""
        x, y, a, b = setup_small
        blur = 0.1
        cost_scale = 0.5  # Half cost to match GeomLoss p=2 default

        # Current implementation with half cost
        f_current, g_current = sinkhorn_geomloss_online_potentials_sqeuclid(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=cost_scale
        )

        # FlashSinkhorn with half cost
        f_flash, g_flash = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=cost_scale
        )

        assert_close(f_flash, f_current, rtol=1e-3, atol=1e-3)
        assert_close(g_flash, g_current, rtol=1e-3, atol=1e-3)

    def test_fused_vs_separate_symmetric(self, setup_small):
        """Test symmetric fused vs separate kernels produce identical results.

        Both fused and separate now use raw x,y coordinates (not pre-scaled Q,K),
        ensuring TF32 rounding patterns match. Parity should be ~1e-4 or better.
        """
        x, y, a, b = setup_small
        eps = 0.1
        n_iters = 50

        # Fused (1 kernel launch)
        f_fused, g_fused = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            n_iters=n_iters, use_epsilon_scaling=False, eps=eps,
            fused=True, allow_tf32=False, autotune=False,
        )

        # Separate (2 kernel launches)
        f_sep, g_sep = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            n_iters=n_iters, use_epsilon_scaling=False, eps=eps,
            fused=False, allow_tf32=False, autotune=False,
        )

        # Should be very close - both use raw x,y with same TF32 rounding
        assert_close(f_fused, f_sep, rtol=1e-4, atol=1e-4)
        assert_close(g_fused, g_sep, rtol=1e-4, atol=1e-4)

    def test_fused_vs_separate_tf32_parity(self, setup_medium):
        """Test fused vs separate match even with TF32 enabled.

        This is a critical regression test. Before the raw x,y coordinate fix,
        TF32 caused ~3% difference (2.9e-02) between fused and separate kernels
        because they used different matmul input scales.

        After fix: Both use raw x,y coordinates with coord_scale applied inside,
        ensuring identical TF32 rounding patterns. Parity should be ~1e-4.
        """
        x, y, a, b = setup_medium
        eps = 0.1
        n_iters = 50

        # Fused with TF32 enabled
        f_fused, g_fused = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            n_iters=n_iters, use_epsilon_scaling=False, eps=eps,
            fused=True, allow_tf32=True, autotune=False,
        )

        # Separate with TF32 enabled
        f_sep, g_sep = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            n_iters=n_iters, use_epsilon_scaling=False, eps=eps,
            fused=False, allow_tf32=True, autotune=False,
        )

        # With raw x,y coordinates, TF32 parity should be ~1e-4 (not 3%!)
        assert_close(f_fused, f_sep, rtol=1e-3, atol=1e-3)
        assert_close(g_fused, g_sep, rtol=1e-3, atol=1e-3)


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_uniform_weights(self, setup_small):
        """Test with uniform marginal weights."""
        x, y, a, b = setup_small
        eps = 0.1

        # a and b are already uniform in setup_small
        f, g = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=eps, n_iters=50,
            allow_tf32=False, autotune=False
        )

        # Check no NaN/Inf
        assert torch.isfinite(f).all(), "f contains NaN/Inf"
        assert torch.isfinite(g).all(), "g contains NaN/Inf"

    def test_small_epsilon(self, setup_small):
        """Test with small regularization (harder case)."""
        x, y, a, b = setup_small
        eps = 0.01

        f, g = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=eps, n_iters=100,
            allow_tf32=False, use_exp2=True, autotune=False
        )

        # Check no NaN/Inf
        assert torch.isfinite(f).all(), "f contains NaN/Inf with small eps"
        assert torch.isfinite(g).all(), "g contains NaN/Inf with small eps"

    def test_large_dimension(self, device):
        """Test with larger feature dimension."""
        torch.manual_seed(42)
        n, m, d = 100, 100, 512
        x = torch.randn(n, d, device=device, dtype=torch.float32)
        y = torch.randn(m, d, device=device, dtype=torch.float32)
        a = torch.ones(n, device=device) / n
        b = torch.ones(m, device=device) / m
        eps = 0.1

        f, g = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=eps, n_iters=50,
            allow_tf32=False, autotune=False
        )

        # Check no NaN/Inf
        assert torch.isfinite(f).all(), "f contains NaN/Inf with large d"
        assert torch.isfinite(g).all(), "g contains NaN/Inf with large d"


# =============================================================================
# COST VALUE PARITY TESTS
# =============================================================================

class TestCostParity:
    """Test that converged potentials give consistent OT cost."""

    def test_cost_convergence(self, setup_medium):
        """Test OT cost converges to a stable value."""
        x, y, a, b = setup_medium
        eps = 0.1

        # FlashSinkhorn with different iteration counts
        f_50, g_50 = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=eps, n_iters=50,
            allow_tf32=False, autotune=False
        )
        f_100, g_100 = sinkhorn_flashstyle_alternating(
            x, y, a, b, eps=eps, n_iters=100,
            allow_tf32=False, autotune=False
        )

        # Compute OT costs using dual formula: <a, f> + <b, g>
        cost_50 = (a * f_50).sum() + (b * g_50).sum()
        cost_100 = (a * f_100).sum() + (b * g_100).sum()

        # Cost should be similar (converged)
        assert_close(cost_50, cost_100, rtol=0.05, atol=0.5)

    def test_cost_vs_geomloss(self, setup_medium):
        """Test OT cost matches GeomLoss symmetric solver."""
        x, y, a, b = setup_medium
        blur = 0.316  # eps = blur^2 ≈ 0.1

        # GeomLoss-style symmetric solver (current implementation)
        f_geomloss, g_geomloss = sinkhorn_geomloss_online_potentials_sqeuclid(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=1.0
        )

        # FlashSinkhorn symmetric solver
        f_flash, g_flash = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur, scaling=0.5,
            allow_tf32=False, use_exp2=True, autotune=False,
            cost_scale=1.0
        )

        # Compute OT costs
        cost_geomloss = (a * f_geomloss).sum() + (b * g_geomloss).sum()
        cost_flash = (a * f_flash).sum() + (b * g_flash).sum()

        assert_close(cost_flash, cost_geomloss, rtol=0.02, atol=0.5)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

"""Tests for c-transform (hard argmin) kernel and autograd function.

Tests verify:
1. Dense parity (values + indices) against naive torch implementation
2. Gradient parity via analytical Danskin formula (not gradcheck — hard argmin)
3. Weighted marginals
4. Cost scale parity
5. Tie-breaking (smallest-j wins across tiles)
6. Asymmetric sizes (n != m)
7. TF32 parity
8. Manual vs autotune parity
"""

import pytest
import torch
import torch.testing

from flash_sinkhorn.c_transform import c_transform_fwd, c_transform_cost


# =============================================================================
# Helpers
# =============================================================================

def _dense_c_transform(x, y, psi, cost_scale):
    """Reference c-transform via dense cost matrix."""
    x_f = x.float()
    y_f = y.float()
    psi_f = psi.float()
    # cost_ij = cost_scale * ||x_i - y_j||²
    diff = x_f.unsqueeze(1) - y_f.unsqueeze(0)  # [n, m, d]
    cost = cost_scale * (diff * diff).sum(dim=-1)  # [n, m]
    # c_i = min_j [cost_ij - psi_j]
    vals = cost - psi_f.unsqueeze(0)  # [n, m]
    c_values, argmin_idx = vals.min(dim=1)
    return c_values, argmin_idx


def _dense_c_transform_cost(x, y, psi, cost_scale, a, b):
    """Reference semi-dual objective via dense cost matrix."""
    c_values, _ = _dense_c_transform(x, y, psi, cost_scale)
    return (a * c_values).sum() + (b * psi.float()).sum()


def _danskin_grad_x(x, y, argmin_idx, a, cost_scale):
    """Analytical grad_x via Danskin's theorem."""
    x_f = x.float()
    y_f = y.float()
    y_matched = y_f[argmin_idx]  # [n, d]
    return (a * 2.0 * cost_scale).unsqueeze(1) * (x_f - y_matched)


def _danskin_grad_psi(argmin_idx, a, b, m):
    """Analytical grad_psi via Danskin's theorem."""
    assigned_mass = torch.zeros(m, device=a.device, dtype=torch.float32)
    assigned_mass.scatter_add_(0, argmin_idx, a)
    return b - assigned_mass


# =============================================================================
# Test 1: Dense parity (values + indices)
# =============================================================================

@pytest.mark.parametrize("n,m,d", [
    (100, 100, 4),
    (100, 300, 4),
    (500, 100, 64),
    (500, 300, 64),
])
def test_dense_parity(n, m, d):
    """Triton c-transform matches dense reference for values and indices."""
    torch.manual_seed(0)
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda")
    cost_scale = 1.0

    # Use allow_tf32=False for exact parity (TF32 tested separately)
    c_triton, idx_triton = c_transform_fwd(
        x, y, psi, cost_scale=cost_scale, allow_tf32=False,
    )
    c_dense, idx_dense = _dense_c_transform(x, y, psi, cost_scale)

    torch.testing.assert_close(c_triton, c_dense, atol=1e-4, rtol=1e-5)
    assert torch.equal(idx_triton, idx_dense), (
        f"Index mismatch: {(idx_triton != idx_dense).sum().item()} / {n} differ"
    )


# =============================================================================
# Test 2: Gradient parity (analytical Danskin formula)
# =============================================================================

@pytest.mark.parametrize("n,m,d", [
    (100, 200, 4),
    (200, 100, 64),
])
def test_gradient_parity(n, m, d):
    """Autograd gradients match analytical Danskin formula."""
    torch.manual_seed(0)
    x = torch.randn(n, d, device="cuda", requires_grad=True)
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda", requires_grad=True)
    cost_scale = 1.0

    a = torch.full((n,), 1.0 / n, device="cuda")
    b = torch.full((m,), 1.0 / m, device="cuda")

    # Autograd path
    loss = c_transform_cost(x, y, psi, cost_scale=cost_scale, a=a, b=b)
    grad_x_auto, grad_psi_auto = torch.autograd.grad(loss, [x, psi])

    # Analytical Danskin path
    _, argmin_idx = _dense_c_transform(x, y, psi, cost_scale)
    grad_x_danskin = _danskin_grad_x(x, y, argmin_idx, a, cost_scale)
    grad_psi_danskin = _danskin_grad_psi(argmin_idx, a, b, m)

    torch.testing.assert_close(grad_x_auto, grad_x_danskin, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_psi_auto, grad_psi_danskin, atol=1e-6, rtol=1e-6)


# =============================================================================
# Test 3: Weighted marginals
# =============================================================================

def test_weighted_marginals():
    """Non-uniform weights change loss and gradients vs uniform."""
    torch.manual_seed(0)
    n, m, d = 100, 200, 4
    x = torch.randn(n, d, device="cuda", requires_grad=True)
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda", requires_grad=True)

    a_uniform = torch.full((n,), 1.0 / n, device="cuda")
    b_uniform = torch.full((m,), 1.0 / m, device="cuda")

    # Non-uniform weights (still normalized to 1)
    a_nonunif = torch.softmax(torch.randn(n, device="cuda"), dim=0)
    b_nonunif = torch.softmax(torch.randn(m, device="cuda"), dim=0)

    loss_uniform = c_transform_cost(x, y, psi, a=a_uniform, b=b_uniform)
    loss_nonunif = c_transform_cost(x, y, psi, a=a_nonunif, b=b_nonunif)

    # Losses should differ
    assert not torch.allclose(loss_uniform, loss_nonunif), "Non-uniform weights should change the loss"

    # Verify non-uniform gradients match Danskin
    grad_x, grad_psi = torch.autograd.grad(loss_nonunif, [x, psi])
    _, argmin_idx = _dense_c_transform(x, y, psi, cost_scale=1.0)
    grad_x_expected = _danskin_grad_x(x, y, argmin_idx, a_nonunif, 1.0)
    grad_psi_expected = _danskin_grad_psi(argmin_idx, a_nonunif, b_nonunif, m)

    torch.testing.assert_close(grad_x, grad_x_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_psi, grad_psi_expected, atol=1e-6, rtol=1e-6)


# =============================================================================
# Test 4: Cost scale parity
# =============================================================================

def test_cost_scale_parity():
    """c_transform(cost_scale=0.5) == 0.5 * c_transform(cost_scale=1.0) when psi=0."""
    torch.manual_seed(0)
    n, m, d = 200, 300, 4
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    psi = torch.zeros(m, device="cuda")

    c_full, _ = c_transform_fwd(x, y, psi, cost_scale=1.0)
    c_half, _ = c_transform_fwd(x, y, psi, cost_scale=0.5)

    torch.testing.assert_close(c_half, 0.5 * c_full, atol=1e-5, rtol=1e-5)


# =============================================================================
# Test 5: Tie-breaking (smallest-j wins across tiles)
# =============================================================================

def test_tie_breaking():
    """When two targets are equidistant, the one with smaller index wins.

    Constructs targets that are equidistant by placing them symmetrically.
    Uses autotune=False with small block_n to force targets into different tiles.
    """
    torch.manual_seed(0)
    n, d = 1, 4
    # Source at origin
    x = torch.zeros(n, d, device="cuda")

    # Two targets at equal distance, in different tile positions
    # Target j=0: at [1, 0, 0, 0]
    # Target j=64: at [-1, 0, 0, 0] (same distance, different tile with block_n=32)
    m = 96
    y = torch.zeros(m, d, device="cuda")
    y[0, 0] = 1.0
    y[64, 0] = -1.0
    # All other targets are far away
    y[1:64, 0] = 100.0
    y[65:, 0] = 100.0

    psi = torch.zeros(m, device="cuda")

    _, idx = c_transform_fwd(
        x, y, psi,
        autotune=False,
        block_n=32,  # Forces j=0 and j=64 into different tiles
    )

    # Smallest-j should win: index 0
    assert idx.item() == 0, f"Expected argmin index 0 (smallest-j tie-break), got {idx.item()}"


# =============================================================================
# Test 6: Asymmetric sizes
# =============================================================================

@pytest.mark.parametrize("n,m", [(200, 500), (500, 200)])
def test_asymmetric_sizes(n, m):
    """c-transform works for n != m in both directions."""
    torch.manual_seed(0)
    d = 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda")

    # Use allow_tf32=False for exact parity (TF32 tested separately)
    c_triton, idx_triton = c_transform_fwd(x, y, psi, allow_tf32=False)
    c_dense, idx_dense = _dense_c_transform(x, y, psi, cost_scale=1.0)

    torch.testing.assert_close(c_triton, c_dense, atol=1e-4, rtol=1e-5)
    assert torch.equal(idx_triton, idx_dense)


# =============================================================================
# Test 7: TF32 parity
# =============================================================================

def test_tf32_parity():
    """allow_tf32=True and False produce close results."""
    torch.manual_seed(0)
    n, m, d = 200, 300, 64
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda")

    c_tf32, idx_tf32 = c_transform_fwd(x, y, psi, allow_tf32=True)
    c_notf32, idx_notf32 = c_transform_fwd(x, y, psi, allow_tf32=False)

    # Values should be close
    torch.testing.assert_close(c_tf32, c_notf32, atol=1e-3, rtol=1e-3)

    # For indices: only compare where the gap between top-2 is large enough
    # that TF32 rounding can't shift the argmin
    c_dense, _ = _dense_c_transform(x, y, psi, cost_scale=1.0)
    x_f = x.float()
    y_f = y.float()
    psi_f = psi.float()
    diff = x_f.unsqueeze(1) - y_f.unsqueeze(0)
    cost = (diff * diff).sum(dim=-1)
    vals = cost - psi_f.unsqueeze(0)
    sorted_vals, _ = vals.sort(dim=1)
    gap = sorted_vals[:, 1] - sorted_vals[:, 0]  # gap between best and second-best

    safe_mask = gap > 1e-3
    if safe_mask.any():
        assert torch.equal(idx_tf32[safe_mask], idx_notf32[safe_mask]), (
            "TF32 indices differ where gap is large"
        )


# =============================================================================
# Test 8: Manual vs autotune parity
# =============================================================================

def test_manual_vs_autotune():
    """Autotune and manual paths produce close results."""
    torch.manual_seed(0)
    n, m, d = 200, 300, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    psi = torch.randn(m, device="cuda")

    c_auto, idx_auto = c_transform_fwd(x, y, psi, autotune=True)
    c_manual, idx_manual = c_transform_fwd(x, y, psi, autotune=False)

    torch.testing.assert_close(c_auto, c_manual, atol=1e-5, rtol=1e-5)

    # For indices, only assert where min-gap is large enough
    x_f = x.float()
    y_f = y.float()
    psi_f = psi.float()
    diff = x_f.unsqueeze(1) - y_f.unsqueeze(0)
    cost = (diff * diff).sum(dim=-1)
    vals = cost - psi_f.unsqueeze(0)
    sorted_vals, _ = vals.sort(dim=1)
    gap = sorted_vals[:, 1] - sorted_vals[:, 0]

    safe_mask = gap > 1e-3
    if safe_mask.any():
        assert torch.equal(idx_auto[safe_mask], idx_manual[safe_mask]), (
            "Autotune vs manual indices differ where gap is large"
        )


# =============================================================================
# Test: grad_y guard raises NotImplementedError
# =============================================================================

def test_grad_y_guard_fwd():
    """c_transform_fwd raises NotImplementedError when y.requires_grad."""
    x = torch.randn(10, 4, device="cuda")
    y = torch.randn(20, 4, device="cuda", requires_grad=True)
    psi = torch.randn(20, device="cuda")

    with pytest.raises(NotImplementedError, match="grad_y not supported"):
        c_transform_fwd(x, y, psi)


def test_grad_y_guard_cost():
    """c_transform_cost raises NotImplementedError when y.requires_grad."""
    x = torch.randn(10, 4, device="cuda")
    y = torch.randn(20, 4, device="cuda", requires_grad=True)
    psi = torch.randn(20, device="cuda")

    with pytest.raises(NotImplementedError, match="grad_y not supported"):
        c_transform_cost(x, y, psi)


# =============================================================================
# Test: weight requires_grad guard
# =============================================================================

def test_weight_requires_grad_guard():
    """c_transform_cost raises ValueError when a or b requires grad."""
    x = torch.randn(10, 4, device="cuda")
    y = torch.randn(20, 4, device="cuda")
    psi = torch.randn(20, device="cuda")

    a_bad = torch.ones(10, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="a must not require grad"):
        c_transform_cost(x, y, psi, a=a_bad)

    b_bad = torch.ones(20, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="b must not require grad"):
        c_transform_cost(x, y, psi, b=b_bad)

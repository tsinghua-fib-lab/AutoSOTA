"""Tests for half_cost parameter (half squared Euclidean cost convention)."""

from __future__ import annotations

import pytest
import torch

from flash_sinkhorn import SamplesLoss


@pytest.fixture
def setup_data():
    """Create test data."""
    torch.manual_seed(42)
    n, m, d = 100, 100, 32
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    return x, y


def test_half_cost_forward_ratio(setup_data):
    """Test that half_cost=True produces half the loss of half_cost=False."""
    x, y = setup_data

    loss_full = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=False)
    loss_half = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True)

    val_full = loss_full(x, y)
    val_half = loss_half(x, y)

    ratio = val_full.item() / val_half.item()
    assert 1.95 < ratio < 2.05, f"Expected ratio ~2, got {ratio}"


def test_half_cost_backward_ratio(setup_data):
    """Test that gradients scale by 2x between full and half cost."""
    x, y = setup_data

    x_full = x.clone().requires_grad_(True)
    loss_full = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=False)
    val_full = loss_full(x_full, y)
    val_full.backward()
    grad_full = x_full.grad.clone()

    x_half = x.clone().requires_grad_(True)
    loss_half = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True)
    val_half = loss_half(x_half, y)
    val_half.backward()
    grad_half = x_half.grad.clone()

    # Gradient norm ratio should be ~2
    ratio = grad_full.norm().item() / grad_half.norm().item()
    assert 1.95 < ratio < 2.05, f"Expected grad ratio ~2, got {ratio}"

    # Gradient directions should be the same (small variation due to convergence)
    cos_sim = torch.nn.functional.cosine_similarity(
        grad_full.flatten(), grad_half.flatten(), dim=0
    )
    assert cos_sim.item() > 0.98, f"Expected cos_sim ~1, got {cos_sim.item()}"


def test_half_cost_matches_geomloss_default(setup_data):
    """Test that half_cost=True matches GeomLoss default p=2 (half squared Euclidean)."""
    geomloss = pytest.importorskip("geomloss")

    x, y = setup_data
    blur = 0.1

    # FlashSinkhorn with half_cost=True
    x_flash = x.clone().requires_grad_(True)
    loss_flash = SamplesLoss(
        loss="sinkhorn", blur=blur, half_cost=True, debias=True, scaling=0.5
    )
    val_flash = loss_flash(x_flash, y)
    val_flash.backward()
    grad_flash = x_flash.grad.clone()

    # GeomLoss with default p=2 (half squared Euclidean)
    x_geomloss = x.clone().requires_grad_(True)
    loss_geomloss = geomloss.SamplesLoss(
        loss="sinkhorn", p=2, blur=blur, scaling=0.5, debias=True, backend="tensorized"
    )
    val_geomloss = loss_geomloss(x_geomloss, y)
    val_geomloss.backward()
    grad_geomloss = x_geomloss.grad.clone()

    # Loss values should be very close
    loss_ratio = val_flash.item() / val_geomloss.item()
    assert 0.98 < loss_ratio < 1.02, f"Expected loss ratio ~1, got {loss_ratio}"

    # Gradients should be nearly identical
    cos_sim = torch.nn.functional.cosine_similarity(
        grad_flash.flatten(), grad_geomloss.flatten(), dim=0
    )
    assert cos_sim.item() > 0.99, f"Expected cos_sim ~1, got {cos_sim.item()}"


def test_half_cost_default_is_full():
    """Test that default half_cost=False gives full squared Euclidean cost."""
    loss = SamplesLoss(loss="sinkhorn", blur=0.1)
    assert loss.half_cost is False
    assert loss.cost_scale == 1.0


def test_half_cost_true_sets_scale():
    """Test that half_cost=True sets cost_scale=0.5."""
    loss = SamplesLoss(loss="sinkhorn", blur=0.1, half_cost=True)
    assert loss.half_cost is True
    assert loss.cost_scale == 0.5


def test_hvp_through_autograd_full_cost(setup_data):
    """Test that HVP through autograd works with half_cost=False (full cost)."""
    x, y = setup_data
    x = x.clone().requires_grad_(True)
    v = torch.randn_like(x)

    loss_fn = SamplesLoss(loss="sinkhorn", blur=0.5, half_cost=False)
    val = loss_fn(x, y)
    grad = torch.autograd.grad(val, x, create_graph=True)[0]
    hvp = torch.autograd.grad(grad, x, grad_outputs=v)[0]

    assert torch.isfinite(hvp).all(), "HVP should be finite"
    assert hvp.norm().item() < 1e10, f"HVP norm too large: {hvp.norm().item()}"


def test_hvp_through_autograd_half_cost(setup_data):
    """Test that HVP through autograd works with half_cost=True."""
    x, y = setup_data
    x = x.clone().requires_grad_(True)
    v = torch.randn_like(x)

    loss_fn = SamplesLoss(loss="sinkhorn", blur=0.5, half_cost=True)
    val = loss_fn(x, y)
    grad = torch.autograd.grad(val, x, create_graph=True)[0]
    hvp = torch.autograd.grad(grad, x, grad_outputs=v)[0]

    assert torch.isfinite(hvp).all(), "HVP should be finite"
    assert hvp.norm().item() < 1e10, f"HVP norm too large: {hvp.norm().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

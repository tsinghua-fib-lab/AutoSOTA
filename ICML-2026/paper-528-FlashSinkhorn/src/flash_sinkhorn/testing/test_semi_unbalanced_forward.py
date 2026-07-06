"""Test semi-unbalanced forward pass for Step 1 of SEMI_UNBALANCED_HVP_PLAN."""

from __future__ import annotations

import torch
import pytest


def test_semi_unbalanced_forward_relaxed_source():
    """Test forward pass with relaxed source, strict target (reach_x=1.0, reach_y=None)."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")

    # Semi-unbalanced: relaxed source, strict target
    loss = SamplesLoss("sinkhorn", blur=0.1, reach_x=1.0, reach_y=None, debias=False)
    val = loss(x, y)

    # Semi-unbalanced OT cost should be positive (fixed in samples_loss.py)
    assert val.isfinite(), f"Semi-unbalanced (relaxed source) gave non-finite: {val}"
    assert val > 0, f"Semi-unbalanced (relaxed source) should be positive, got: {val}"
    print(f"✓ Semi-unbalanced (relaxed source, strict target): OT cost = {val.item():.6f}")


def test_semi_unbalanced_forward_relaxed_target():
    """Test forward pass with strict source, relaxed target (reach_x=None, reach_y=1.0)."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")

    # Semi-unbalanced: strict source, relaxed target
    loss = SamplesLoss("sinkhorn", blur=0.1, reach_x=None, reach_y=1.0, debias=False)
    val = loss(x, y)

    # Semi-unbalanced OT cost should be positive (fixed in samples_loss.py)
    assert val.isfinite(), f"Semi-unbalanced (relaxed target) gave non-finite: {val}"
    assert val > 0, f"Semi-unbalanced (relaxed target) should be positive, got: {val}"
    print(f"✓ Semi-unbalanced (strict source, relaxed target): OT cost = {val.item():.6f}")


def test_fully_unbalanced_asymmetric():
    """Test forward pass with asymmetric relaxation (reach_x=1.0, reach_y=2.0)."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")

    # Fully unbalanced, asymmetric
    loss = SamplesLoss("sinkhorn", blur=0.1, reach_x=1.0, reach_y=2.0, debias=False)
    val = loss(x, y)

    # Fully unbalanced OT cost should be positive
    assert val.isfinite(), f"Fully unbalanced (asymmetric) gave non-finite: {val}"
    assert val > 0, f"Fully unbalanced (asymmetric) should be positive, got: {val}"
    print(f"✓ Fully unbalanced (asymmetric reach_x=1.0, reach_y=2.0): OT cost = {val.item():.6f}")


def test_semi_unbalanced_vs_balanced_different():
    """Verify semi-unbalanced gives different result than balanced."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")

    # Balanced
    loss_balanced = SamplesLoss("sinkhorn", blur=0.1, debias=False)
    val_balanced = loss_balanced(x, y)

    # Semi-unbalanced
    loss_semi = SamplesLoss("sinkhorn", blur=0.1, reach_x=1.0, reach_y=None, debias=False)
    val_semi = loss_semi(x, y)

    assert not torch.isclose(val_balanced, val_semi), (
        f"Expected different values for balanced ({val_balanced:.6f}) "
        f"vs semi-unbalanced ({val_semi:.6f})"
    )
    print(f"✓ Balanced: {val_balanced.item():.6f}, Semi-unbalanced: {val_semi.item():.6f} (different as expected)")


def test_semi_unbalanced_potentials():
    """Test that potentials can be extracted for semi-unbalanced OT."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    a = torch.ones(n, device="cuda") / n
    b = torch.ones(m, device="cuda") / m

    # Semi-unbalanced with potentials=True
    loss = SamplesLoss("sinkhorn", blur=0.1, reach_x=1.0, reach_y=None, potentials=True, debias=False)
    f, g = loss(a, x, b, y)

    assert f.shape == (n,), f"Expected f shape (n,), got {f.shape}"
    assert g.shape == (m,), f"Expected g shape (m,), got {g.shape}"
    assert f.isfinite().all(), "f contains non-finite values"
    assert g.isfinite().all(), "g contains non-finite values"
    print(f"✓ Semi-unbalanced potentials: f shape={f.shape}, g shape={g.shape}")


def test_semi_unbalanced_gradient():
    """Test that gradients can be computed for semi-unbalanced OT."""
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    n, m, d = 256, 256, 16
    x = torch.randn(n, d, device="cuda", requires_grad=True)
    y = torch.randn(m, d, device="cuda")

    # Semi-unbalanced with gradient
    loss = SamplesLoss("sinkhorn", blur=0.1, reach_x=1.0, reach_y=None, debias=False)
    val = loss(x, y)
    grad_x = torch.autograd.grad(val, x)[0]

    assert grad_x.shape == x.shape, f"Expected grad shape {x.shape}, got {grad_x.shape}"
    assert grad_x.isfinite().all(), "grad_x contains non-finite values"
    print(f"✓ Semi-unbalanced gradient: shape={grad_x.shape}, norm={grad_x.norm().item():.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Testing semi-unbalanced forward pass")
    print("=" * 60)

    test_semi_unbalanced_forward_relaxed_source()
    test_semi_unbalanced_forward_relaxed_target()
    test_fully_unbalanced_asymmetric()
    test_semi_unbalanced_vs_balanced_different()
    test_semi_unbalanced_potentials()
    test_semi_unbalanced_gradient()

    print("=" * 60)
    print("All Step 1 tests passed!")
    print("=" * 60)

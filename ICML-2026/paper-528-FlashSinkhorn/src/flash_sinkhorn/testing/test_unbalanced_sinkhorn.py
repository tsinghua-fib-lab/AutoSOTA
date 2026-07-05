"""Test unbalanced Sinkhorn OT against GeomLoss reference."""

from __future__ import annotations

import pytest
import torch

# Skip all tests if geomloss is not available
geomloss = pytest.importorskip("geomloss")

from geomloss.sinkhorn_samples import sinkhorn_tensorized

from flash_sinkhorn import SamplesLoss
from flash_sinkhorn.kernels._common import max_diameter
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import sinkhorn_flashstyle_symmetric


def _sqdist_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost for batched inputs."""
    return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)


@pytest.fixture
def sample_data():
    """Generate sample point clouds for testing."""
    torch.manual_seed(42)
    n, m, d = 512, 512, 32
    x = torch.randn(n, d, device="cuda", dtype=torch.float32)
    y = torch.randn(m, d, device="cuda", dtype=torch.float32)
    a = torch.ones(n, device="cuda", dtype=torch.float32) / n
    b = torch.ones(m, device="cuda", dtype=torch.float32) / m
    return x, y, a, b


class TestUnbalancedPotentials:
    """Test unbalanced OT potential computation."""

    @pytest.mark.parametrize("reach", [0.5, 1.0, 2.0, 5.0])
    def test_potentials_vs_geomloss(self, sample_data, reach: float):
        """Compare unbalanced potentials against GeomLoss sinkhorn_tensorized."""
        x, y, a, b = sample_data
        blur = 0.5
        scaling = 0.5
        diameter = max_diameter(x, y)

        # GeomLoss reference using sinkhorn_tensorized
        f_geo, g_geo = sinkhorn_tensorized(
            a.unsqueeze(0),
            x.unsqueeze(0),
            b.unsqueeze(0),
            y.unsqueeze(0),
            p=2,
            blur=blur,
            reach=reach,
            scaling=scaling,
            diameter=diameter,
            debias=False,
            potentials=True,
            cost=_sqdist_cost,
        )
        f_geo = f_geo.squeeze()
        g_geo = g_geo.squeeze()

        # OT Triton
        f_tri, g_tri = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            scaling=scaling,
            reach=reach,
            diameter=diameter,
            use_epsilon_scaling=True,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            autotune=False,
        )

        print(f"\nreach={reach}:")
        print(f"  GeomLoss f: mean={f_geo.mean():.4f}, std={f_geo.std():.4f}")
        print(f"  Triton f:   mean={f_tri.mean():.4f}, std={f_tri.std():.4f}")

        # Direct comparison (should match exactly)
        torch.testing.assert_close(f_tri, f_geo, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g_tri, g_geo, rtol=1e-4, atol=1e-4)

    def test_balanced_unchanged(self, sample_data):
        """Verify balanced OT (reach=None) still works correctly."""
        x, y, a, b = sample_data
        blur = 0.5
        scaling = 0.5
        diameter = max_diameter(x, y)

        # GeomLoss reference (balanced)
        f_geo, g_geo = sinkhorn_tensorized(
            a.unsqueeze(0),
            x.unsqueeze(0),
            b.unsqueeze(0),
            y.unsqueeze(0),
            p=2,
            blur=blur,
            reach=None,
            scaling=scaling,
            diameter=diameter,
            debias=False,
            potentials=True,
            cost=_sqdist_cost,
        )
        f_geo = f_geo.squeeze()
        g_geo = g_geo.squeeze()

        # OT Triton (balanced)
        f_tri, g_tri = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            scaling=scaling,
            reach=None,
            diameter=diameter,
            use_epsilon_scaling=True,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            autotune=False,
        )

        print(f"\nBalanced (reach=None):")
        print(f"  GeomLoss f: mean={f_geo.mean():.4f}, std={f_geo.std():.4f}")
        print(f"  Triton f:   mean={f_tri.mean():.4f}, std={f_tri.std():.4f}")

        torch.testing.assert_close(f_tri, f_geo, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(g_tri, g_geo, rtol=1e-5, atol=1e-5)


class TestUnbalancedCost:
    """Test unbalanced OT cost computation.

    Note: For unbalanced OT, the cost formula differs from balanced OT.
    The standard formula uses UnbalancedWeight scaling. For now, we test
    that potentials match, which is the foundation for correct costs.
    """

    @pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
    def test_cost_from_potentials(self, sample_data, reach: float):
        """Test that balanced cost formula still works (sanity check)."""
        x, y, a, b = sample_data
        blur = 0.5
        scaling = 0.5
        diameter = max_diameter(x, y)

        # Get potentials from both
        f_geo, g_geo = sinkhorn_tensorized(
            a.unsqueeze(0),
            x.unsqueeze(0),
            b.unsqueeze(0),
            y.unsqueeze(0),
            p=2,
            blur=blur,
            reach=reach,
            scaling=scaling,
            diameter=diameter,
            debias=False,
            potentials=True,
            cost=_sqdist_cost,
        )
        f_geo = f_geo.squeeze()
        g_geo = g_geo.squeeze()

        f_tri, g_tri = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            scaling=scaling,
            reach=reach,
            diameter=diameter,
            use_epsilon_scaling=True,
            last_extrapolation=True,
            allow_tf32=False,
            use_exp2=False,
            autotune=False,
        )

        # Compute balanced-style cost from potentials
        cost_geo = (a * f_geo).sum() + (b * g_geo).sum()
        cost_tri = (a * f_tri).sum() + (b * g_tri).sum()

        print(f"\nreach={reach}:")
        print(f"  Cost from GeomLoss potentials: {cost_geo.item():.6f}")
        print(f"  Cost from Triton potentials:   {cost_tri.item():.6f}")

        # Costs should match since potentials match
        torch.testing.assert_close(cost_tri, cost_geo, rtol=1e-4, atol=1e-4)


class TestDampeningBehavior:
    """Test that damping has expected effect on potentials."""

    def test_smaller_reach_gives_smaller_potential_variance(self, sample_data):
        """Smaller reach (stronger relaxation) should give smaller potential variance."""
        x, y, a, b = sample_data
        blur = 0.5

        # Large reach (closer to balanced)
        f_large, g_large = sinkhorn_flashstyle_symmetric(
            x, y, a, b, blur=blur, reach=10.0, autotune=False, use_exp2=False
        )

        # Small reach (stronger relaxation)
        f_small, g_small = sinkhorn_flashstyle_symmetric(
            x, y, a, b, blur=blur, reach=0.5, autotune=False, use_exp2=False
        )

        # Smaller reach should give smaller potential variance (due to damping)
        f_large_std = f_large.std().item()
        f_small_std = f_small.std().item()
        g_large_std = g_large.std().item()
        g_small_std = g_small.std().item()

        print(f"\nPotential standard deviations:")
        print(f"  reach=10.0: std(f)={f_large_std:.4f}, std(g)={g_large_std:.4f}")
        print(f"  reach=0.5:  std(f)={f_small_std:.4f}, std(g)={g_small_std:.4f}")

        assert f_small_std < f_large_std, "Smaller reach should give smaller f variance"
        assert g_small_std < g_large_std, "Smaller reach should give smaller g variance"

    def test_damping_matches_geomloss(self):
        """Verify our dampening function matches GeomLoss."""
        from geomloss.sinkhorn_divergence import dampening as geomloss_dampening
        from flash_sinkhorn.kernels._common import dampening

        test_cases = [
            (0.1, 1.0),
            (0.25, 0.5),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.01, 0.1),
        ]

        for eps, rho in test_cases:
            geo_val = geomloss_dampening(eps, rho)
            tri_val = dampening(eps, rho)
            assert abs(geo_val - tri_val) < 1e-10, f"Dampening mismatch at eps={eps}, rho={rho}"

        # Balanced case
        assert dampening(0.1, None) == 1.0


class TestSemiUnbalancedOT:
    """Test semi-unbalanced OT where only one marginal is relaxed."""

    def test_reach_x_y_equals_reach(self, sample_data):
        """Test that reach_x=reach_y=r gives same result as reach=r."""
        x, y, a, b = sample_data
        blur = 0.5
        reach = 1.0

        # Using legacy reach parameter
        f_legacy, g_legacy = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach=reach,
            autotune=False,
            use_exp2=False,
        )

        # Using new reach_x/reach_y parameters
        f_new, g_new = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=reach,
            reach_y=reach,
            autotune=False,
            use_exp2=False,
        )

        torch.testing.assert_close(f_new, f_legacy, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(g_new, g_legacy, rtol=1e-7, atol=1e-7)

    def test_semi_unbalanced_x_only(self, sample_data):
        """Test relaxing only source marginal (reach_x, reach_y=None)."""
        x, y, a, b = sample_data
        blur = 0.5
        reach_x = 1.0

        # Semi-unbalanced: relax source, strict target
        f_semi, g_semi = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=reach_x,
            reach_y=None,  # Target marginal is strict
            autotune=False,
            use_exp2=False,
        )

        # Balanced case for comparison
        f_bal, g_bal = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=None,
            reach_y=None,
            autotune=False,
            use_exp2=False,
        )

        # Semi-unbalanced should differ from balanced (f is dampened, g is not)
        # The f potentials should have smaller variance than balanced
        assert not torch.allclose(f_semi, f_bal, rtol=1e-2, atol=1e-2), \
            "Semi-unbalanced should differ from balanced"

        # f should be dampened (smaller variance)
        print(f"\nSemi-unbalanced (reach_x={reach_x}, reach_y=None):")
        print(f"  f semi: std={f_semi.std():.4f}, balanced std={f_bal.std():.4f}")
        print(f"  g semi: std={g_semi.std():.4f}, balanced std={g_bal.std():.4f}")

    def test_semi_unbalanced_y_only(self, sample_data):
        """Test relaxing only target marginal (reach_x=None, reach_y)."""
        x, y, a, b = sample_data
        blur = 0.5
        reach_y = 1.0

        # Semi-unbalanced: strict source, relax target
        f_semi, g_semi = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=None,  # Source marginal is strict
            reach_y=reach_y,
            autotune=False,
            use_exp2=False,
        )

        # Balanced case for comparison
        f_bal, g_bal = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=None,
            reach_y=None,
            autotune=False,
            use_exp2=False,
        )

        # Semi-unbalanced should differ from balanced
        assert not torch.allclose(g_semi, g_bal, rtol=1e-2, atol=1e-2), \
            "Semi-unbalanced should differ from balanced"

        print(f"\nSemi-unbalanced (reach_x=None, reach_y={reach_y}):")
        print(f"  f semi: std={f_semi.std():.4f}, balanced std={f_bal.std():.4f}")
        print(f"  g semi: std={g_semi.std():.4f}, balanced std={g_bal.std():.4f}")

    def test_asymmetric_reach(self, sample_data):
        """Test fully asymmetric unbalanced OT (different reach_x and reach_y)."""
        x, y, a, b = sample_data
        blur = 0.5
        reach_x = 0.5  # Strong relaxation on source
        reach_y = 5.0  # Weak relaxation on target

        f_asym, g_asym = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach_x=reach_x,
            reach_y=reach_y,
            autotune=False,
            use_exp2=False,
        )

        # Compare with symmetric case
        f_sym, g_sym = sinkhorn_flashstyle_symmetric(
            x, y, a, b,
            blur=blur,
            reach=1.0,  # Equal reach for both
            autotune=False,
            use_exp2=False,
        )

        # Should differ from symmetric case
        assert not torch.allclose(f_asym, f_sym, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(g_asym, g_sym, rtol=1e-2, atol=1e-2)

        print(f"\nAsymmetric (reach_x={reach_x}, reach_y={reach_y}):")
        print(f"  f asym: std={f_asym.std():.4f}, sym std={f_sym.std():.4f}")
        print(f"  g asym: std={g_asym.std():.4f}, sym std={g_sym.std():.4f}")

    def test_samples_loss_api_semi_unbalanced(self, sample_data):
        """Test SamplesLoss API with semi-unbalanced parameters."""
        x, y, a, b = sample_data
        blur = 0.5

        # Test reach_x only (semi-unbalanced)
        loss_fn = SamplesLoss(
            blur=blur,
            reach_x=1.0,
            reach_y=None,
            debias=False,
            potentials=True,
            autotune=False,
            use_exp2=False,
        )
        f, g = loss_fn(a, x, b, y)
        assert f.shape == (x.shape[0],)
        assert g.shape == (y.shape[0],)

        # Test reach_y only (semi-unbalanced)
        loss_fn_y = SamplesLoss(
            blur=blur,
            reach_x=None,
            reach_y=1.0,
            debias=False,
            potentials=True,
            autotune=False,
            use_exp2=False,
        )
        f_y, g_y = loss_fn_y(a, x, b, y)
        assert f_y.shape == (x.shape[0],)
        assert g_y.shape == (y.shape[0],)

        # Test asymmetric reach
        loss_fn_asym = SamplesLoss(
            blur=blur,
            reach_x=0.5,
            reach_y=2.0,
            debias=False,
            potentials=True,
            autotune=False,
            use_exp2=False,
        )
        f_asym, g_asym = loss_fn_asym(a, x, b, y)
        assert f_asym.shape == (x.shape[0],)
        assert g_asym.shape == (y.shape[0],)

    def test_samples_loss_cost_semi_unbalanced(self, sample_data):
        """Test that cost computation works for semi-unbalanced OT."""
        x, y, a, b = sample_data
        blur = 0.5

        # Semi-unbalanced cost (reach_x only)
        loss_x = SamplesLoss(
            blur=blur,
            reach_x=1.0,
            reach_y=None,
            debias=False,
            potentials=False,
            autotune=False,
            use_exp2=False,
        )
        cost_x = loss_x(a, x, b, y)
        assert cost_x.dim() == 0, "Cost should be a scalar"
        assert torch.isfinite(cost_x), "Cost should be finite"

        # Semi-unbalanced cost (reach_y only)
        loss_y = SamplesLoss(
            blur=blur,
            reach_x=None,
            reach_y=1.0,
            debias=False,
            potentials=False,
            autotune=False,
            use_exp2=False,
        )
        cost_y = loss_y(a, x, b, y)
        assert cost_y.dim() == 0, "Cost should be a scalar"
        assert torch.isfinite(cost_y), "Cost should be finite"

        # Balanced cost for reference
        loss_bal = SamplesLoss(
            blur=blur,
            debias=False,
            potentials=False,
            autotune=False,
            use_exp2=False,
        )
        cost_bal = loss_bal(a, x, b, y)

        # Fully unbalanced cost for comparison
        loss_unbal = SamplesLoss(
            blur=blur,
            reach=1.0,
            debias=False,
            potentials=False,
            autotune=False,
            use_exp2=False,
        )
        cost_unbal = loss_unbal(a, x, b, y)

        print(f"\nSemi-unbalanced costs:")
        print(f"  reach_x=1.0, reach_y=None: {cost_x.item():.6f}")
        print(f"  reach_x=None, reach_y=1.0: {cost_y.item():.6f}")
        print(f"  reach=1.0 (both):          {cost_unbal.item():.6f}")
        print(f"  balanced:                  {cost_bal.item():.6f}")


class TestAlternatingUnbalancedParity:
    """Regression tests for alternating backend with unbalanced OT.

    Locks fix for the rho/reach double-squaring bug: _autograd.py previously
    passed config.rho_x (already reach^2) to a reach_x parameter, which
    squared it again, giving rho = reach^4 instead of reach^2.
    """

    @pytest.mark.parametrize("reach", [1.0, 2.0, 5.0])
    def test_autograd_vs_direct_cost(self, sample_data, reach):
        """SamplesLoss (autograd path) must match direct solver cost."""
        x, y, a, b = sample_data
        eps = 0.1
        n_iters = 50

        # Autograd path (goes through _SinkhornCostFn)
        loss_fn = SamplesLoss(
            loss="sinkhorn",
            backend="alternating",
            eps=eps,
            n_iters=n_iters,
            use_epsilon_scaling=False,
            reach=reach,
            half_cost=True,
            debias=False,
        )
        cost_autograd = loss_fn(a, x, b, y)

        # Direct solver path
        from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
            sinkhorn_flashstyle_alternating,
        )
        rho = reach ** 2
        f, g = sinkhorn_flashstyle_alternating(
            x, y, a, b,
            eps=eps,
            n_iters=n_iters,
            cost_scale=0.5,
            rho_x=rho,
            rho_y=rho,
        )
        cost_direct = (a * f).sum() + (b * g).sum()

        rel_err = abs(cost_autograd.item() - cost_direct.item()) / (
            abs(cost_direct.item()) + 1e-12
        )
        assert rel_err < 1e-4, (
            f"Autograd vs direct cost mismatch at reach={reach}: "
            f"autograd={cost_autograd.item():.6f}, direct={cost_direct.item():.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize(
        "reach_x,reach_y",
        [(1.0, None), (None, 2.0), (1.0, 5.0)],
    )
    def test_semi_unbalanced_alternating(self, sample_data, reach_x, reach_y):
        """Semi-unbalanced alternating OT through SamplesLoss must be finite."""
        x, y, a, b = sample_data
        loss_fn = SamplesLoss(
            loss="sinkhorn",
            backend="alternating",
            eps=0.1,
            n_iters=50,
            use_epsilon_scaling=False,
            reach_x=reach_x,
            reach_y=reach_y,
            half_cost=True,
            debias=False,
        )
        cost = loss_fn(a, x, b, y)
        assert torch.isfinite(cost), (
            f"Non-finite cost with reach_x={reach_x}, reach_y={reach_y}: {cost.item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

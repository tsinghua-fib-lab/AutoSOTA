"""Tests for LPP (log_prob) and ACAUC metrics."""

import math

import pytest
import torch

from posteriors import (
    DirectPosteriorEstimator,
    DualFlowPosteriorEstimator,
    Estimator,
)
from flow_matching.torch_flow import FlowMatching, _GaussianBase, _DataEpsBase
from utils.metrics import acauc


# ============================================================================
# ACAUC Tests
# ============================================================================


class TestACAUC:
    """Test the paper-exact ACAUC implementation (Appendix J)."""

    def test_perfect_calibration(self):
        """When theta_true are drawn from the same distribution as posterior
        samples, ACAUC should be ≈ 0. This simulates perfect calibration
        where ranks are uniformly distributed."""
        torch.manual_seed(42)
        n_obs, n_samples, d = 500, 2000, 3

        # Both true and samples from the same N(0,1) -- perfectly calibrated
        theta_true = torch.randn(n_obs, d)
        posterior_samples = torch.randn(n_samples, n_obs, d)

        result = acauc(theta_true, posterior_samples)
        assert abs(result) < 0.03, f"ACAUC for perfect calibration should be ≈ 0, got {result}"

    def test_overconfident(self):
        """Narrow posteriors: true values often fall outside credible intervals."""
        torch.manual_seed(42)
        n_obs, n_samples, d = 500, 1000, 2

        # True values from a wider distribution than the posterior
        theta_true = torch.randn(n_obs, d) * 3.0

        # Narrow posterior centered at 0 (overconfident)
        posterior_samples = torch.randn(n_samples, n_obs, d) * 0.5

        result = acauc(theta_true, posterior_samples)
        assert result > 0.1, f"ACAUC for overconfident posterior should be > 0.1, got {result}"

    def test_underconfident(self):
        """Wide posteriors: true values always well inside credible intervals."""
        torch.manual_seed(42)
        n_obs, n_samples, d = 500, 1000, 2

        # True values from a narrow distribution
        theta_true = torch.randn(n_obs, d) * 0.1

        # Wide posterior (underconfident)
        posterior_samples = torch.randn(n_samples, n_obs, d) * 10.0

        result = acauc(theta_true, posterior_samples)
        assert result < -0.1, f"ACAUC for underconfident posterior should be < -0.1, got {result}"

    def test_returns_scalar(self):
        """ACAUC should return a Python float."""
        theta_true = torch.randn(10, 2)
        posterior_samples = torch.randn(50, 10, 2)
        result = acauc(theta_true, posterior_samples)
        assert isinstance(result, float)


# ============================================================================
# Base Distribution log_prob Tests
# ============================================================================


class TestBaseDistLogProb:
    """Test log_prob for _GaussianBase and _DataEpsBase."""

    def test_gaussian_base_log_prob_shape(self):
        """log_prob should return (batch,) shaped tensor."""
        base = _GaussianBase(torch.Size([3]))
        x = torch.randn(10, 3)
        lp = base.log_prob(x)
        assert lp.shape == (10,)

    def test_gaussian_base_log_prob_values(self):
        """log_prob should match manual N(0, I) computation."""
        base = _GaussianBase(torch.Size([2]))
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        lp = base.log_prob(x)

        # log N(0,I) at origin: -0.5 * (0 + 2*log(2pi))
        expected_origin = -0.5 * 2 * math.log(2 * math.pi)
        assert abs(lp[0].item() - expected_origin) < 1e-5

        # log N(0,I) at (1,0): -0.5 * (1 + 2*log(2pi))
        expected_10 = -0.5 * (1.0 + 2 * math.log(2 * math.pi))
        assert abs(lp[1].item() - expected_10) < 1e-5

    def test_data_eps_base_log_prob_shape(self):
        """log_prob should return (batch,) shaped tensor."""
        base = _DataEpsBase(torch.Size([3]), eps=0.1)
        x = torch.randn(10, 3)
        cond = torch.randn(10, 3)
        lp = base.log_prob(x, cond)
        assert lp.shape == (10,)

    def test_data_eps_base_log_prob_values(self):
        """log_prob at cond should be the mode (highest density)."""
        base = _DataEpsBase(torch.Size([2]), eps=0.5)
        cond = torch.tensor([[1.0, 2.0]])

        # At the mean, log_prob should be highest
        lp_at_mean = base.log_prob(cond, cond)
        lp_away = base.log_prob(cond + 1.0, cond)
        assert lp_at_mean.item() > lp_away.item()

    def test_data_eps_requires_cond(self):
        """DataEpsBase.log_prob should raise if cond is None."""
        base = _DataEpsBase(torch.Size([2]), eps=0.1)
        x = torch.randn(5, 2)
        with pytest.raises(ValueError, match="requires conditioning"):
            base.log_prob(x)


# ============================================================================
# FlowMatching log_prob Tests
# ============================================================================


class TestFlowMatchingLogProb:
    """Test FlowMatching.log_prob via instantaneous change of variables."""

    @pytest.fixture
    def simple_flow(self):
        """Create a small FlowMatching model for testing."""
        dim = torch.Size([2])
        cond_dim = torch.Size([2])
        flow = FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="gaussian",
            dim=dim,
            cond_dim=cond_dim,
            num_steps=10,
            drift={"architecture": "resmlp", "hidden_dim": [32, 32]},
        )
        return flow

    def test_log_prob_returns_finite(self, simple_flow):
        """log_prob should return finite values."""
        target = torch.randn(5, 2)
        cond = torch.randn(5, 2)
        device = torch.device("cpu")

        lp = simple_flow.log_prob(target, cond, device, num_steps=5)
        assert lp.shape == (5,)
        assert torch.isfinite(lp).all(), f"Non-finite log_prob values: {lp}"

    def test_log_prob_batch_consistency(self, simple_flow):
        """log_prob of same input computed twice should give same result."""
        torch.manual_seed(123)
        target = torch.randn(3, 2)
        cond = torch.randn(3, 2)
        device = torch.device("cpu")

        lp1 = simple_flow.log_prob(target, cond, device, num_steps=5)
        lp2 = simple_flow.log_prob(target, cond, device, num_steps=5)
        assert torch.allclose(lp1, lp2, atol=1e-5), f"Inconsistent log_prob: {lp1} vs {lp2}"

    def test_exact_vs_hutchinson(self, simple_flow):
        """Exact and Hutchinson divergence should give similar results."""
        torch.manual_seed(42)
        target = torch.randn(10, 2)
        cond = torch.randn(10, 2)
        device = torch.device("cpu")

        lp_exact = simple_flow.log_prob(
            target, cond, device, num_steps=10, exact_trace=True
        )
        lp_hutch = simple_flow.log_prob(
            target, cond, device, num_steps=10, exact_trace=False, n_hutchinson_probes=50
        )

        # With enough probes, Hutchinson should be close to exact
        diff = (lp_exact - lp_hutch).abs().mean()
        assert diff < 2.0, f"Exact vs Hutchinson divergence too large: mean abs diff = {diff}"

    def test_log_prob_higher_for_likely_samples(self, simple_flow):
        """After training, samples from the model should have higher log_prob than random noise."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        # Generate simple training data: theta ~ N(0,1), cond ~ N(0,1)
        theta_train = torch.randn(200, 2)
        cond_train = torch.randn(200, 2)

        # Brief training
        optimizer = torch.optim.Adam(simple_flow.parameters(), lr=1e-3)
        for _ in range(100):
            optimizer.zero_grad()
            loss = simple_flow.compute_loss(theta_train, cond_train)
            loss.backward()
            optimizer.step()

        # Evaluate: in-distribution vs far out-of-distribution
        cond_test = torch.randn(20, 2)
        theta_in = torch.randn(20, 2)  # In-distribution
        theta_out = torch.randn(20, 2) * 10  # Far from training distribution

        lp_in = simple_flow.log_prob(theta_in, cond_test, device, num_steps=10)
        lp_out = simple_flow.log_prob(theta_out, cond_test, device, num_steps=10)

        # In-distribution should generally have higher log_prob
        assert lp_in.mean() > lp_out.mean(), (
            f"In-distribution log_prob ({lp_in.mean():.2f}) should be > "
            f"out-of-distribution ({lp_out.mean():.2f})"
        )

    def test_log_prob_with_data_eps_base(self):
        """log_prob should work with data_eps base distribution."""
        dim = torch.Size([2])
        cond_dim = torch.Size([2])
        flow = FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="data_eps",
            dim=dim,
            cond_dim=cond_dim,
            num_steps=5,
            drift={"architecture": "resmlp", "hidden_dim": [16, 16]},
            base_dist_params={"eps": 0.1},
        )

        target = torch.randn(5, 2)
        cond = torch.randn(5, 2)
        device = torch.device("cpu")

        lp = flow.log_prob(target, cond, device, num_steps=5)
        assert lp.shape == (5,)
        assert torch.isfinite(lp).all(), f"Non-finite log_prob with data_eps: {lp}"


# ============================================================================
# FlowMatching reverse_ode Tests
# ============================================================================


class TestFlowMatchingReverseOde:
    """Test FlowMatching.reverse_ode method."""

    @pytest.fixture
    def simple_flow(self):
        dim = torch.Size([2])
        cond_dim = torch.Size([2])
        return FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="gaussian",
            dim=dim,
            cond_dim=cond_dim,
            num_steps=10,
            drift={"architecture": "resmlp", "hidden_dim": [32, 32]},
        )

    def test_reverse_ode_returns_correct_shapes(self, simple_flow):
        """reverse_ode should return (x0, cond_scaled, log_det) with correct shapes."""
        target = torch.randn(5, 2)
        cond = torch.randn(5, 2)
        device = torch.device("cpu")

        x0, cond_scaled, log_det = simple_flow.reverse_ode(target, cond, device, num_steps=5)
        assert x0.shape == (5, 2)
        assert cond_scaled.shape == (5, 2)
        assert log_det.shape == (5,)

    def test_reverse_ode_consistent_with_log_prob(self, simple_flow):
        """log_prob computed via reverse_ode + base should match log_prob directly."""
        torch.manual_seed(42)
        target = torch.randn(5, 2)
        cond = torch.randn(5, 2)
        device = torch.device("cpu")

        lp_direct = simple_flow.log_prob(target, cond, device, num_steps=5)

        x0, cond_scaled, log_det = simple_flow.reverse_ode(target, cond, device, num_steps=5)
        base_lp = simple_flow.base_dist.log_prob(x0)
        lp_manual = base_lp + log_det

        assert torch.allclose(lp_direct, lp_manual, atol=1e-5)


# ============================================================================
# DualFlowPosteriorEstimator log_prob Tests
# ============================================================================


class TestDualFlowPosteriorLogProb:
    """Test DualFlowPosteriorEstimator.log_prob."""

    @pytest.fixture
    def dual_flow_no_denoiser(self):
        """Create a DualFlowPosteriorEstimator without denoiser for testing."""
        torch.manual_seed(42)
        theta_dim = torch.Size([2])
        obs_dim = torch.Size([3])

        # Create proposal (NPE-based)
        estimator = Estimator(
            task_name="pure_gaussian",
            dim=theta_dim,
            cond_dim=obs_dim,
            density_estimator="nsf",
        )
        proposal = DirectPosteriorEstimator("proposal", estimator)

        # Create posterior transform flow
        flow_theta = FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="gaussian",
            dim=theta_dim,
            cond_dim=obs_dim,
            num_steps=10,
            drift={"architecture": "resmlp", "hidden_dim": [32, 32]},
        )

        return DualFlowPosteriorEstimator(
            name="test_dual",
            base_dist=proposal,
            flow_theta=flow_theta,
            flow_x=None,
        )

    @pytest.fixture
    def dual_flow_with_denoiser(self):
        """Create a DualFlowPosteriorEstimator with denoiser for testing."""
        torch.manual_seed(42)
        theta_dim = torch.Size([2])
        obs_dim = torch.Size([3])

        # Create proposal (NPE-based)
        estimator = Estimator(
            task_name="pure_gaussian",
            dim=theta_dim,
            cond_dim=obs_dim,
            density_estimator="nsf",
        )
        proposal = DirectPosteriorEstimator("proposal", estimator)

        # Create posterior transform flow
        flow_theta = FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="gaussian",
            dim=theta_dim,
            cond_dim=obs_dim,
            num_steps=10,
            drift={"architecture": "resmlp", "hidden_dim": [32, 32]},
        )

        # Create denoiser flow (obs_dim -> obs_dim, conditioned on obs_dim)
        flow_x = FlowMatching(
            probability_path="ot2",
            prior="uniform",
            base_dist="gaussian",
            dim=obs_dim,
            cond_dim=obs_dim,
            num_steps=10,
            drift={"architecture": "resmlp", "hidden_dim": [32, 32]},
        )

        return DualFlowPosteriorEstimator(
            name="test_dual_denoiser",
            base_dist=proposal,
            flow_theta=flow_theta,
            flow_x=flow_x,
        )

    def test_log_prob_no_denoiser_returns_finite(self, dual_flow_no_denoiser):
        """log_prob without denoiser should return finite values."""
        theta = torch.randn(5, 2)
        y = torch.randn(5, 3)

        lp = dual_flow_no_denoiser.log_prob(theta, y, num_steps=5)
        assert lp.shape == (5,), f"Expected shape (5,), got {lp.shape}"
        assert torch.isfinite(lp).all(), f"Non-finite log_prob values: {lp}"

    def test_log_prob_with_denoiser_returns_finite(self, dual_flow_with_denoiser):
        """log_prob with denoiser should return finite values."""
        theta = torch.randn(5, 2)
        y = torch.randn(5, 3)

        lp = dual_flow_with_denoiser.log_prob(theta, y, num_steps=5, ntraj=3)
        assert lp.shape == (5,), f"Expected shape (5,), got {lp.shape}"
        assert torch.isfinite(lp).all(), f"Non-finite log_prob values: {lp}"

    def test_log_prob_deterministic_without_denoiser(self, dual_flow_no_denoiser):
        """Without denoiser, log_prob should be deterministic."""
        torch.manual_seed(42)
        theta = torch.randn(3, 2)
        y = torch.randn(3, 3)

        lp1 = dual_flow_no_denoiser.log_prob(theta, y, num_steps=5)
        lp2 = dual_flow_no_denoiser.log_prob(theta, y, num_steps=5)
        assert torch.allclose(lp1, lp2, atol=1e-5), f"Non-deterministic: {lp1} vs {lp2}"

    def test_log_prob_ntraj_1_vs_no_denoiser(self, dual_flow_no_denoiser):
        """With ntraj=1 and no denoiser, result should be same regardless of ntraj kwarg."""
        theta = torch.randn(3, 2)
        y = torch.randn(3, 3)

        lp1 = dual_flow_no_denoiser.log_prob(theta, y, num_steps=5, ntraj=1)
        lp2 = dual_flow_no_denoiser.log_prob(theta, y, num_steps=5, ntraj=5)
        # Without denoiser, ntraj is irrelevant
        assert torch.allclose(lp1, lp2, atol=1e-5)

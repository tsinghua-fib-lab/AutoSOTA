"""Tests for the RobustConjugateGP class."""

import torch
import numpy as np
import pytest

from rcgp.models.robust_gp import RobustConjugateGP
from rcgp.models.standard_gp import StandardGP
from rcgp.weighting import PlateauIMQ
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean


class TestRCGPClass:
    """Test class for RobustConjugateGP functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data and models."""
        self.noise_variance = 0.01
        self.sigma = np.sqrt(self.noise_variance)
        self.X_test = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.double)
        
    def _create_components(self):
        """Create reusable model components."""
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-8))
        likelihood.noise = torch.tensor(self.noise_variance, dtype=torch.double)
        likelihood.raw_noise.requires_grad = False
        
        mean = ConstantMean()
        mean.constant = torch.tensor(0.0, dtype=torch.double)
        mean.raw_constant.requires_grad = False
        
        base_kernel = RBFKernel(ard_num_dims=1)
        base_kernel.lengthscale = torch.tensor(0.3, dtype=torch.double)
        base_kernel.raw_lengthscale.requires_grad = False
        
        covar = ScaleKernel(base_kernel)
        covar.outputscale = torch.tensor(1.0, dtype=torch.double)
        covar.raw_outputscale.requires_grad = False
        
        return likelihood, mean, covar
        
    def create_models(self, X_train, Y_train, plateau_width=2.0, outcome_transform=None):
        """Create RCGP and BoTorch models with identical hyperparameters."""
        weighting_fn = PlateauIMQ(plateau_width=plateau_width, sigma=self.sigma, c=1.0)
        
        rcgp_likelihood, rcgp_mean, rcgp_covar = self._create_components()
        botorch_likelihood, botorch_mean, botorch_covar = self._create_components()
        
        rcgp_model = RobustConjugateGP(
            X_train, Y_train, weighting_function=weighting_fn,
            likelihood=rcgp_likelihood, mean_module=rcgp_mean,
            covar_module=rcgp_covar, outcome_transform=outcome_transform
        )
        
        botorch_model = SingleTaskGP(
            X_train, Y_train, likelihood=botorch_likelihood,
            mean_module=botorch_mean, covar_module=botorch_covar,
            outcome_transform=outcome_transform
        )
        
        rcgp_model.eval()
        botorch_model.eval()
        
        return rcgp_model, botorch_model, weighting_fn
    
    def _assert_ratio_in_range(self, value1, value2, min_ratio=0.99, max_ratio=1.01, name=""):
        """Assert ratio is within expected range."""
        if abs(value2) > 1e-6:
            ratio = value1 / value2
            assert min_ratio <= ratio <= max_ratio, \
                f"{name} ratio {ratio:.4f} outside [{min_ratio}, {max_ratio}]"
        else:
            assert abs(value1) < 1e-4, f"{name} values should both be near zero"
    
    def _assert_weights_in_plateau(self, weights, expected_beta):
        """Assert weights match expected plateau value."""
        assert torch.allclose(weights, torch.full_like(weights, expected_beta), rtol=1e-3), \
            f"Expected weights {expected_beta:.6f}, got {weights}"
    
    def _assert_weights_outside_plateau(self, weights, expected_beta):
        """Assert weights are smaller than plateau value."""
        assert torch.all(weights < expected_beta), \
            f"Expected weights < {expected_beta:.6f}, got {weights}"
    
    def _compare_posteriors(self, rcgp_model, botorch_model, X_test_points, check_variance_larger=False):
        """Compare posteriors between RCGP and BoTorch models."""
        with torch.no_grad():
            for X_test_single in X_test_points:
                X_test_point = X_test_single.unsqueeze(0)
                rcgp_posterior = rcgp_model.posterior(X_test_point)
                botorch_posterior = botorch_model.posterior(X_test_point)
                
                rcgp_mean = rcgp_posterior.mean.item()
                rcgp_var = rcgp_posterior.variance.item()
                botorch_mean = botorch_posterior.mean.item()
                botorch_var = botorch_posterior.variance.item()
                
                if not check_variance_larger:
                    self._assert_ratio_in_range(rcgp_mean, botorch_mean, name="Mean")
                    self._assert_ratio_in_range(rcgp_var, botorch_var, name="Variance")
                else:
                    assert rcgp_var > botorch_var, \
                        f"RCGP variance {rcgp_var:.6f} not > BoTorch {botorch_var:.6f}"
                    assert rcgp_var / botorch_var > 1.01, \
                        f"Variance ratio {rcgp_var/botorch_var:.4f} not meaningfully larger"
    
    def test_within_plateau_similarity(self):
        """Test RCGP behaves like Standard GP when observations are within plateau."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[0.5], [0.3], [-0.8], [0.2]], dtype=torch.double)
        
        rcgp_model, botorch_model, _ = self.create_models(X_train, Y_train, plateau_width=2.0)
        
        weights = rcgp_model.get_weights()
        expected_beta = self.sigma / np.sqrt(2)
        self._assert_weights_in_plateau(weights, expected_beta)
        
        self._compare_posteriors(rcgp_model, botorch_model, self.X_test)
        
        # J matrix should be approximately identity
        _, J_matrix, _ = rcgp_model._get_robust_components(
            rcgp_model.train_inputs[0], rcgp_model.train_targets)
        I_matrix = torch.eye(len(weights), dtype=torch.double)
        assert torch.allclose(J_matrix, I_matrix, rtol=1e-2)
    
    def test_outside_plateau_robustness(self):
        """Test RCGP variance is larger when observations are outside plateau."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[2.5], [3.0], [-2.8], [2.2]], dtype=torch.double)
        
        rcgp_model, botorch_model, _ = self.create_models(X_train, Y_train, plateau_width=0.5)
        
        weights = rcgp_model.get_weights()
        expected_beta = self.sigma / np.sqrt(2)
        self._assert_weights_outside_plateau(weights, expected_beta)
        
        _, J_matrix, _ = rcgp_model._get_robust_components(
            rcgp_model.train_inputs[0], rcgp_model.train_targets)
        assert torch.all(torch.diag(J_matrix) > 1.0)
        
        self._compare_posteriors(rcgp_model, botorch_model, self.X_test, check_variance_larger=True)
    
    def test_mixed_plateau_behavior(self):
        """Test RCGP with mixed in/out plateau observations."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[0.5], [2.5], [-0.3], [-2.0]], dtype=torch.double)
        
        rcgp_model, _, _ = self.create_models(X_train, Y_train, plateau_width=1.0)
        
        weights = rcgp_model.get_weights()
        expected_beta = self.sigma / np.sqrt(2)
        
        Y_abs = torch.abs(Y_train.squeeze())
        in_plateau = Y_abs <= 1.0
        out_plateau = Y_abs > 1.0
        
        if torch.any(in_plateau):
            assert torch.allclose(weights[in_plateau], 
                                torch.full_like(weights[in_plateau], expected_beta), rtol=1e-2)
        if torch.any(out_plateau):
            assert torch.all(weights[out_plateau] < expected_beta * 0.99)
        
        # Check J matrix values
        _, J_matrix, _ = rcgp_model._get_robust_components(
            rcgp_model.train_inputs[0], rcgp_model.train_targets)
        J_diagonal = torch.diag(J_matrix)
        
        if torch.any(in_plateau):
            assert torch.allclose(J_diagonal[in_plateau], torch.ones_like(J_diagonal[in_plateau]), rtol=1e-1)
        if torch.any(out_plateau):
            assert torch.all(J_diagonal[out_plateau] > 1.01)
    
    def test_hyperparameter_consistency(self):
        """Test models have identical hyperparameters."""
        X_train = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
        Y_train = torch.tensor([[0.1], [0.3], [-0.2]], dtype=torch.double)
        
        rcgp_model, botorch_model, _ = self.create_models(X_train, Y_train)
        
        assert abs(rcgp_model.likelihood.noise.item() - botorch_model.likelihood.noise.item()) < 1e-8
        assert abs(rcgp_model.covar_module.base_kernel.lengthscale.item() - 
                  botorch_model.covar_module.base_kernel.lengthscale.item()) < 1e-6
        assert abs(rcgp_model.covar_module.outputscale.item() - 
                  botorch_model.covar_module.outputscale.item()) < 1e-6
    
    def test_mathematical_correctness(self):
        """Test J matrix computation and weight calculations."""
        X_train = torch.tensor([[0.1], [0.4], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[0.2], [1.5], [-2.0]], dtype=torch.double)
        
        rcgp_model, _, weighting_fn = self.create_models(X_train, Y_train, plateau_width=1.0)
        
        weights = rcgp_model.get_weights()
        _, J_matrix, _ = rcgp_model._get_robust_components(
            rcgp_model.train_inputs[0], rcgp_model.train_targets)
        
        # Verify J matrix formula: J_ii = σ²/(2*w_i²)
        expected_J_diagonal = (self.sigma**2) / (2 * weights**2)
        assert torch.allclose(torch.diag(J_matrix), expected_J_diagonal, rtol=1e-4)
        
        # Verify weights match weighting function
        direct_weights = weighting_fn.weight(rcgp_model.train_inputs[0], rcgp_model.train_targets)
        assert torch.allclose(weights, direct_weights, rtol=1e-6)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single observation
        X_single = torch.tensor([[0.5]], dtype=torch.double)
        Y_single = torch.tensor([[1.0]], dtype=torch.double)
        rcgp_single, _, _ = self.create_models(X_single, Y_single)
        
        with torch.no_grad():
            pred = rcgp_single.posterior(torch.tensor([[0.3]], dtype=torch.double))
            assert torch.isfinite(pred.mean) and torch.isfinite(pred.variance)
            assert pred.variance > 0
        
        # Identical observations
        X_identical = torch.tensor([[0.2], [0.2], [0.2]], dtype=torch.double)
        Y_identical = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.double)
        rcgp_identical, _, _ = self.create_models(X_identical, Y_identical)
        
        weights = rcgp_identical.get_weights()
        assert torch.all(torch.isfinite(weights)) and len(weights) == 3
    
    def test_outcome_standardization(self):
        """Test outcome standardization works correctly."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.double)
        Y_train = torch.tensor([[10.0], [12.0], [8.0], [15.0], [11.0]], dtype=torch.double)
        
        # Test with standardization
        rcgp_std, _, _ = self.create_models(X_train, Y_train, plateau_width=2.0,
                                          outcome_transform=Standardize(m=1))
        
        # Check standardized targets
        std_targets = rcgp_std.train_targets
        assert abs(std_targets.mean().item()) < 1e-6
        assert abs(std_targets.std().item() - 1.0) < 1e-6
        
        # Test without standardization
        rcgp_no_std, _, _ = self.create_models(X_train, Y_train, plateau_width=10.0,
                                             outcome_transform=None)
        
        # Both should predict in original scale
        with torch.no_grad():
            pred_std = rcgp_std.posterior(torch.tensor([[0.4]], dtype=torch.double))
            pred_no_std = rcgp_no_std.posterior(torch.tensor([[0.4]], dtype=torch.double))
            assert 5 < pred_std.mean.item() < 20
            assert 5 < pred_no_std.mean.item() < 20
    
    def test_posterior_untransforming(self):
        """Test posterior predictions are correctly untransformed."""
        X_train = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.double)
        Y_train = torch.tensor([[100.0], [102.0], [98.0], [101.0]], dtype=torch.double)
        
        rcgp_model, _, _ = self.create_models(X_train, Y_train, plateau_width=10.0,
                                            outcome_transform=Standardize(m=1))
        
        with torch.no_grad():
            for i, x in enumerate(X_train):
                pred = rcgp_model.posterior(x.unsqueeze(0))
                pred_mean = pred.mean.item()
                true_val = Y_train[i].item()
                assert abs(pred_mean - true_val) < 5.0
                assert 90 < pred_mean < 110
    
    def test_condition_on_observations_with_standardization(self):
        """Test conditioning on new observations with standardization."""
        X_init = torch.tensor([[0.3], [0.7]], dtype=torch.double)
        Y_init = torch.tensor([[50.0], [55.0]], dtype=torch.double)
        
        rcgp_model, _, _ = self.create_models(X_init, Y_init, plateau_width=5.0,
                                            outcome_transform=Standardize(m=1))
        
        X_new = torch.tensor([[0.5]], dtype=torch.double)
        Y_new = torch.tensor([[52.0]], dtype=torch.double)
        
        rcgp_updated = rcgp_model.condition_on_observations(X_new, Y_new)
        rcgp_updated.eval()
        
        assert len(rcgp_updated.train_inputs[0]) == 3
        
        with torch.no_grad():
            pred = rcgp_updated.posterior(torch.tensor([[0.4]], dtype=torch.double))
            assert 45 < pred.mean.item() < 60
        
        if rcgp_updated.outcome_transform is not None:
            assert hasattr(rcgp_updated.outcome_transform, 'means')
            assert hasattr(rcgp_updated.outcome_transform, 'stdvs')
    
    def test_standardization_with_standard_gp(self):
        """Test StandardGP with outcome standardization."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[20.0], [25.0], [22.0], [27.0]], dtype=torch.double)
        
        std_gp_with = StandardGP(X_train, Y_train, outcome_transform=Standardize(m=1))
        std_gp_without = StandardGP(X_train, Y_train, outcome_transform=None)
        
        std_gp_with.eval()
        std_gp_without.eval()
        
        assert not torch.allclose(std_gp_with.train_targets, std_gp_without.train_targets)
        
        with torch.no_grad():
            X_test = torch.tensor([[0.4]], dtype=torch.double)
            pred_with = std_gp_with.posterior(X_test).mean.item()
            pred_without = std_gp_without.posterior(X_test).mean.item()
            
            assert 18 < pred_with < 30
            assert 18 < pred_without < 30
    
    def test_standardization_numerical_stability(self):
        """Test standardization with poorly scaled data."""
        X_train = torch.tensor([[0.1], [0.3], [0.5], [0.7]], dtype=torch.double)
        Y_train = torch.tensor([[1e6], [1e6 + 100], [1e6 - 100], [1e6 + 50]], dtype=torch.double)
        
        rcgp_model, _, _ = self.create_models(X_train, Y_train, plateau_width=200.0,
                                            outcome_transform=Standardize(m=1))
        
        with torch.no_grad():
            pred = rcgp_model.posterior(torch.tensor([[0.4]], dtype=torch.double))
            assert torch.isfinite(pred.mean) and torch.isfinite(pred.variance)
            assert pred.variance > 0
            assert 0.9e6 < pred.mean.item() < 1.1e6
    
    def test_gradient_computations(self):
        """Test gradient computations are correct."""
        X_train = torch.tensor([[0.2], [0.6], [0.9]], dtype=torch.double)
        Y_train = torch.tensor([[0.3], [1.8], [-2.5]], dtype=torch.double)
        
        rcgp_model, _, _ = self.create_models(X_train, Y_train, plateau_width=1.0)
        
        _, _, gradient_correction = rcgp_model._get_robust_components(
            rcgp_model.train_inputs[0], rcgp_model.train_targets)
        
        assert torch.all(torch.isfinite(gradient_correction))
        
        Y_abs = torch.abs(Y_train.squeeze())
        in_plateau = Y_abs <= 1.0
        out_plateau = Y_abs > 1.0
        
        if torch.any(in_plateau):
            assert torch.allclose(gradient_correction[in_plateau], 
                                torch.zeros_like(gradient_correction[in_plateau]), atol=1e-6)
        if torch.any(out_plateau):
            assert not torch.allclose(gradient_correction[out_plateau], 
                                     torch.zeros_like(gradient_correction[out_plateau]), atol=1e-3)
    
    def test_rcgp_posterior_variance_higher_than_standard(self):
        """Test that RCGP posterior variance is systematically higher than standard GP 
        when robustness is triggered, and examine the theoretical relationship."""
        
        # Test scenario 1: Gradually increasing outlier severity
        X_train = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.double)
        test_scenarios = [
            # (Y_train, plateau_width, description)
            (torch.tensor([[0.1], [0.2], [0.1], [0.2]], dtype=torch.double), 1.0, "all in plateau"),
            (torch.tensor([[0.1], [0.2], [1.5], [0.2]], dtype=torch.double), 1.0, "one mild outlier"),
            (torch.tensor([[0.1], [0.2], [3.0], [0.2]], dtype=torch.double), 1.0, "one strong outlier"),
            (torch.tensor([[0.1], [3.0], [3.0], [0.2]], dtype=torch.double), 1.0, "two outliers"),
            (torch.tensor([[3.0], [3.0], [3.0], [3.0]], dtype=torch.double), 0.5, "all outliers"),
        ]
        
        X_test_comprehensive = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.double)
        
        for Y_train, plateau_width, description in test_scenarios:
            rcgp_model, botorch_model, _ = self.create_models(
                X_train, Y_train, plateau_width=plateau_width)
            
            weights = rcgp_model.get_weights()
            
            # Check if any observations are outside plateau
            Y_abs = torch.abs(Y_train.squeeze())
            has_outliers = torch.any(Y_abs > plateau_width)
            
            with torch.no_grad():
                variance_ratios = []
                for X_test_point in X_test_comprehensive:
                    rcgp_posterior = rcgp_model.posterior(X_test_point.unsqueeze(0))
                    botorch_posterior = botorch_model.posterior(X_test_point.unsqueeze(0))
                    
                    rcgp_var = rcgp_posterior.variance.item()
                    botorch_var = botorch_posterior.variance.item()
                    
                    assert rcgp_var > 0 and botorch_var > 0, "Variances must be positive"
                    
                    ratio = rcgp_var / botorch_var
                    variance_ratios.append(ratio)
                    
                    if has_outliers:
                        # When outliers are present, RCGP should have strictly higher variance
                        assert ratio > 1.0, f"RCGP variance should be > standard GP for {description}, got {ratio:.6f}"
                        # For strong outliers, expect meaningful increase
                        if "strong outlier" in description or "all outliers" in description:
                            assert ratio > 1.03, f"Expected >3% variance increase for {description}, got {ratio:.3f}"
                    else:
                        # When all in plateau, should be essentially equal (tight tolerance for numerical precision)
                        assert abs(ratio - 1.0) < 0.01, f"Should be equal for {description}, got ratio {ratio:.6f}"
                
                avg_ratio = sum(variance_ratios) / len(variance_ratios)
                print(f"  {description}: avg variance ratio = {avg_ratio:.3f}, weights = {weights.numpy()}")
        
        # Test scenario 2: Verify theoretical relationship with J matrix
        X_theory = torch.tensor([[0.3], [0.7]], dtype=torch.double)
        Y_theory = torch.tensor([[0.1], [5.0]], dtype=torch.double)  # One clean, one outlier
        
        rcgp_theory, botorch_theory, _ = self.create_models(X_theory, Y_theory, plateau_width=1.0)
        
        weights_theory = rcgp_theory.get_weights()
        _, J_matrix, _ = rcgp_theory._get_robust_components(
            rcgp_theory.train_inputs[0], rcgp_theory.train_targets)
        
        # Verify J matrix theoretical relationship
        expected_J = (self.sigma**2) / (2 * weights_theory**2)
        assert torch.allclose(torch.diag(J_matrix), expected_J, rtol=1e-4), \
            "J matrix should follow theoretical formula J_ii = σ²/(2*w_i²)"
        
        # The outlier should have smaller weight and larger J value
        outlier_idx = torch.argmax(torch.abs(Y_theory.squeeze()))
        clean_idx = 1 - outlier_idx
        
        assert weights_theory[outlier_idx] < weights_theory[clean_idx], \
            "Outlier should have smaller weight"
        assert J_matrix[outlier_idx, outlier_idx] > J_matrix[clean_idx, clean_idx], \
            "Outlier should have larger J matrix diagonal value"
        
        # Test specific prediction point to verify variance increase
        X_test_theory = torch.tensor([[0.5]], dtype=torch.double)  # Between the two points
        
        with torch.no_grad():
            rcgp_pred = rcgp_theory.posterior(X_test_theory)
            botorch_pred = botorch_theory.posterior(X_test_theory)
            
            variance_increase = rcgp_pred.variance.item() / botorch_pred.variance.item()
            assert variance_increase > 1.1, f"Expected >10% variance increase, got {variance_increase:.3f}"
            
            print(f"  Theoretical test: RCGP var = {rcgp_pred.variance.item():.6f}, "
                  f"BoTorch var = {botorch_pred.variance.item():.6f}, "
                  f"ratio = {variance_increase:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
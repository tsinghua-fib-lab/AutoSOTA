"""
Test that RCGP with large plateau behaves identically to StandardGP.

Key theoretical result: When all points are within the plateau,
weights w = β = σ/√2, and the J matrix becomes the identity matrix.
"""

import torch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.types import DEFAULT
from rcgp.models.standard_gp import StandardGP
from rcgp.models.robust_gp import RobustConjugateGP
from rcgp.weighting import PlateauIMQ


def create_matched_models(train_X, train_Y, outcome_transform=None):
    """
    Create StandardGP and RobustGP with matched hyperparameters.
    
    Args:
        train_X: Training inputs
        train_Y: Training targets
        outcome_transform: Optional outcome transform
        
    Returns:
        Tuple of (standard_gp, robust_gp, weighting_function)
    """
    # Create StandardGP
    standard_gp = StandardGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=outcome_transform
    )
    
    # Extract sigma from StandardGP
    sigma = torch.sqrt(standard_gp.likelihood.noise).item()
    
    # Create RCGP with huge plateau that contains all points
    huge_plateau = PlateauIMQ(
        plateau_width=1000.0,  # Very large plateau to contain all points
        sigma=sigma,
        c=1.0,
        center_fn=None
    )
    
    robust_gp = RobustConjugateGP(
        train_X=train_X,
        train_Y=train_Y,
        weighting_function=huge_plateau,
        outcome_transform=outcome_transform
    )

    return standard_gp, robust_gp, huge_plateau


def test_j_matrix_identity():
    """Test that J matrix is identity when all points are within plateau."""
    torch.manual_seed(42)
    
    # Generate test data
    train_X = torch.randn(10, 2)
    train_Y = torch.randn(10, 1)
    
    # Create matched models
    _, robust_gp, weighting_fn = create_matched_models(train_X, train_Y)
    
    # Get the robust components
    weights, J_matrix, gradient_correction = robust_gp._get_robust_components(
        robust_gp.train_inputs[0], robust_gp.train_targets
    )
    
    # Verify beta = σ/√2
    sigma = torch.sqrt(robust_gp.likelihood.noise).item()
    expected_beta = sigma / np.sqrt(2)
    assert np.isclose(weighting_fn.beta, expected_beta), f"Beta mismatch: {weighting_fn.beta} vs {expected_beta}"
    
    # Verify all weights equal beta
    assert torch.allclose(weights, torch.full_like(weights, expected_beta), atol=1e-6), "Weights are not all equal to beta"
    
    # Verify J matrix is identity
    expected_J = torch.eye(len(weights), dtype=J_matrix.dtype, device=J_matrix.device)
    J_diff = torch.abs(J_matrix - expected_J).max()
    assert J_diff < 1e-6, f"J matrix is not identity, max diff: {J_diff}"
    
    # Verify gradient correction is zero (all points in plateau)
    grad_norm = torch.abs(gradient_correction).max()
    assert grad_norm < 1e-10, f"Gradient correction should be zero in plateau, got max: {grad_norm}"
    
    print("✓ J matrix is identity and beta = σ/√2 verified!")


def test_prior_equivalence():
    """Test that prior distributions match between RCGP and StandardGP."""
    torch.manual_seed(42)
    
    # Generate test data
    train_X = torch.randn(10, 2)
    train_Y = torch.randn(10, 1)
    test_X = torch.randn(5, 2)
    
    # Create matched models
    standard_gp, robust_gp, _ = create_matched_models(train_X, train_Y)
    
    # Put both models in training mode (to get priors)
    standard_gp.train()
    robust_gp.train()
    
    # Compute priors at test points
    standard_prior = standard_gp(test_X)
    robust_prior = robust_gp(test_X)
    
    # Check that prior means and covariances match
    mean_diff = torch.abs(standard_prior.mean - robust_prior.mean).max()
    cov_diff = torch.abs(standard_prior.covariance_matrix - robust_prior.covariance_matrix).max()
    
    assert mean_diff < 1e-6, f"Prior means differ by {mean_diff}"
    assert cov_diff < 1e-5, f"Prior covariances differ by {cov_diff}"
    
    print("✓ Prior distributions match!")


def test_posterior_equivalence():
    """Test that posterior distributions match between RCGP and StandardGP."""
    torch.manual_seed(42)
    
    # Generate test data
    train_X = torch.randn(15, 3)
    train_Y = torch.randn(15, 1)
    test_X = torch.randn(8, 3)
    
    # Create matched models
    standard_gp, robust_gp, weighting_fn = create_matched_models(train_X, train_Y)
    
    # Put both models in eval mode (to get posteriors)
    standard_gp.eval()
    robust_gp.eval()
    
    # Verify all points are within the plateau
    weights = robust_gp.get_weights()
    expected_beta = weighting_fn.beta
    assert torch.allclose(weights, torch.full_like(weights, expected_beta), atol=1e-6), f"Not all weights equal β={expected_beta}"
    
    # Compute posteriors at test points
    standard_posterior = standard_gp.posterior(test_X)
    robust_posterior = robust_gp.posterior(test_X)
    
    # Check that posterior means and variances match
    mean_diff = torch.abs(standard_posterior.mean - robust_posterior.mean).max()
    var_diff = torch.abs(standard_posterior.variance - robust_posterior.variance).max()
    
    assert mean_diff < 1e-5, f"Posterior means differ by {mean_diff}"
    assert var_diff < 1e-5, f"Posterior variances differ by {var_diff}"
    
    print("✓ Posterior distributions match!")


def test_mll_equivalence():
    """Test that MLL computations match between RCGP and StandardGP."""
    torch.manual_seed(42)
    
    # Generate test data
    train_X = torch.randn(20, 2)
    train_Y = torch.randn(20, 1)
    
    # Create matched models
    standard_gp, robust_gp, _ = create_matched_models(train_X, train_Y)
    
    # Set up MLL for both models
    standard_mll = ExactMarginalLogLikelihood(standard_gp.likelihood, standard_gp)
    robust_mll = ExactMarginalLogLikelihood(robust_gp.likelihood, robust_gp)
    
    # Put both in training mode
    standard_gp.train()
    robust_gp.train()
    
    # Compute MLL values
    standard_output = standard_gp(train_X)
    standard_mll_val = standard_mll(standard_output, standard_gp.train_targets)
    
    robust_output = robust_gp(train_X)
    robust_mll_val = robust_mll(robust_output, robust_gp.train_targets)
    
    # Compare MLL values
    mll_diff = torch.abs(standard_mll_val - robust_mll_val)
    assert mll_diff < 1e-4, f"MLL values differ by {mll_diff.item()}"
    
    print("✓ MLL computations match!")


def test_gradient_computation_equivalence():
    """Test that gradient computations match for hyperparameter optimization."""
    torch.manual_seed(42)
    
    # Generate test data
    train_X = torch.randn(10, 2)
    train_Y = torch.randn(10, 1)
    
    # Create matched models
    standard_gp, robust_gp, _ = create_matched_models(train_X, train_Y)
    
    # Set up MLL
    standard_mll = ExactMarginalLogLikelihood(standard_gp.likelihood, standard_gp)
    robust_mll = ExactMarginalLogLikelihood(robust_gp.likelihood, robust_gp)
    
    # Training mode
    standard_gp.train()
    robust_gp.train()
    
    # Compute gradients for StandardGP
    standard_gp.zero_grad()
    standard_output = standard_gp(train_X)
    standard_loss = -standard_mll(standard_output, standard_gp.train_targets)
    standard_loss.backward()
    
    # Compute gradients for RobustGP
    robust_gp.zero_grad()
    robust_output = robust_gp(train_X)
    robust_loss = -robust_mll(robust_output, robust_gp.train_targets)
    robust_loss.backward()
    
    # Compare gradients
    for (s_name, s_param), (r_name, r_param) in zip(standard_gp.named_parameters(), robust_gp.named_parameters()):
        if s_param.grad is not None and r_param.grad is not None:
            grad_diff = torch.abs(s_param.grad - r_param.grad).max()
            assert grad_diff < 1e-4, f"Gradients for {s_name} differ by {grad_diff}"
    
    print("✓ Gradient computations match!")


def test_with_outcome_transform():
    """Test equivalence when using outcome transforms (standardization)."""
    torch.manual_seed(42)
    
    # Generate test data with different scale
    train_X = torch.randn(15, 2)
    train_Y = 10 + 5 * torch.randn(15, 1)  # Different mean and scale
    test_X = torch.randn(5, 2)
    
    # Create matched models with outcome transform
    standard_gp, robust_gp, _ = create_matched_models(train_X, train_Y, outcome_transform=DEFAULT)
    
    # Eval mode for posteriors
    standard_gp.eval()
    robust_gp.eval()
    
    # Compute posteriors (should be untransformed automatically)
    standard_posterior = standard_gp.posterior(test_X)
    robust_posterior = robust_gp.posterior(test_X)
    
    # Check posterior means and variances match
    mean_diff = torch.abs(standard_posterior.mean - robust_posterior.mean).max()
    var_diff = torch.abs(standard_posterior.variance - robust_posterior.variance).max()
    
    assert mean_diff < 1e-4, f"Posterior means with transform differ by {mean_diff}"
    assert var_diff < 1e-4, f"Posterior variances with transform differ by {var_diff}"
    
    print("✓ Models match with outcome transforms!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RCGP-StandardGP Equivalence with Large Plateau")
    print("=" * 60)
    
    tests = [
        ("J matrix identity and beta = σ/√2", test_j_matrix_identity),
        ("Prior equivalence", test_prior_equivalence),
        ("Posterior equivalence", test_posterior_equivalence),
        ("MLL equivalence", test_mll_equivalence),
        ("Gradient computation equivalence", test_gradient_computation_equivalence),
        ("With outcome transforms", test_with_outcome_transform),
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{i}. Testing {name}...")
        test_func()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("RCGP with large plateau behaves identically to StandardGP")
    print("J matrix is identity when all points are within plateau")
    print("=" * 60)
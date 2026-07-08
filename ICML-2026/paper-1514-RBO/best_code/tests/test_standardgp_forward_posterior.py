"""
Test StandardGP forward() and posterior() methods against BoTorch SingleTaskGP
with identical hyperparameters to ensure they produce the same predictions.
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from rcgp.models.standard_gp import StandardGP


def set_identical_parameters(standard_gp, botorch_gp, lengthscale, noise, mean_constant):
    """Set identical hyperparameters for both models."""
    
    # Set StandardGP parameters
    standard_gp.covar_module.lengthscale = lengthscale
    standard_gp.likelihood.noise = noise
    standard_gp.mean_module.constant = mean_constant
    
    # Set BoTorch GP parameters
    if hasattr(botorch_gp.covar_module, 'base_kernel'):
        botorch_gp.covar_module.base_kernel.lengthscale = lengthscale
    else:
        botorch_gp.covar_module.lengthscale = lengthscale
    botorch_gp.likelihood.noise = noise
    botorch_gp.mean_module.constant = mean_constant


def test_forward_comparison():
    """Test forward() method comparison."""
    print("Testing forward() method comparison")
    print("=" * 50)
    
    # Create synthetic dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.linspace(0, 1, 10).unsqueeze(-1).double()
    true_y = torch.sin(2 * torch.pi * X.squeeze())
    noise = torch.randn(10) * 0.1
    y = (true_y + noise).unsqueeze(-1)
    
    # Create models
    standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Set identical parameters
    lengthscale = torch.tensor(0.3, dtype=torch.float64)
    noise = torch.tensor(0.05, dtype=torch.float64)
    mean_constant = torch.tensor(0.5, dtype=torch.float64)
    
    set_identical_parameters(standard_gp, botorch_gp, lengthscale, noise, mean_constant)
    
    print(f"Set parameters:")
    print(f"  lengthscale: {lengthscale.item()}")
    print(f"  noise: {noise.item()}")
    print(f"  mean_constant: {mean_constant.item()}")
    
    # Verify parameters are identical
    print(f"\nVerifying parameters:")
    print(f"StandardGP lengthscale: {standard_gp.covar_module.lengthscale.item()}")
    print(f"BoTorch GP lengthscale: {botorch_gp.covar_module.lengthscale.item()}")
    print(f"StandardGP noise: {standard_gp.likelihood.noise.item()}")
    print(f"BoTorch GP noise: {botorch_gp.likelihood.noise.item()}")
    print(f"StandardGP mean: {standard_gp.mean_module.constant.item()}")
    print(f"BoTorch GP mean: {botorch_gp.mean_module.constant.item()}")
    
    # Test points
    test_X = torch.linspace(-0.5, 1.5, 15).unsqueeze(-1).double()
    
    # Put both models in eval mode
    standard_gp.eval()
    botorch_gp.eval()
    
    # Test forward() method
    with torch.no_grad():
        standard_output = standard_gp(test_X)
        botorch_output = botorch_gp(test_X)
    
    # Compare means and covariances
    standard_mean = standard_output.mean
    standard_cov = standard_output.covariance_matrix
    
    botorch_mean = botorch_output.mean
    botorch_cov = botorch_output.covariance_matrix
    
    print(f"\nForward() comparison:")
    print(f"Mean shapes - StandardGP: {standard_mean.shape}, BoTorch: {botorch_mean.shape}")
    print(f"Covariance shapes - StandardGP: {standard_cov.shape}, BoTorch: {botorch_cov.shape}")
    
    # Check mean differences
    mean_diff = torch.abs(standard_mean - botorch_mean)
    max_mean_diff = torch.max(mean_diff)
    rel_mean_diff = max_mean_diff / torch.max(torch.abs(botorch_mean))
    
    print(f"Max absolute mean difference: {max_mean_diff.item():.2e}")
    print(f"Max relative mean difference: {rel_mean_diff.item():.2%}")
    
    # Check covariance differences
    cov_diff = torch.abs(standard_cov - botorch_cov)
    max_cov_diff = torch.max(cov_diff)
    rel_cov_diff = max_cov_diff / torch.max(torch.abs(botorch_cov))
    
    print(f"Max absolute covariance difference: {max_cov_diff.item():.2e}")
    print(f"Max relative covariance difference: {rel_cov_diff.item():.2%}")
    
    # Test passes if differences are < 1%
    mean_pass = rel_mean_diff < 0.01
    cov_pass = rel_cov_diff < 0.01
    
    print(f"\nForward test results:")
    print(f"  Mean comparison: {'PASS ✓' if mean_pass else 'FAIL ✗'}")
    print(f"  Covariance comparison: {'PASS ✓' if cov_pass else 'FAIL ✗'}")
    
    return mean_pass and cov_pass


def test_posterior_comparison():
    """Test posterior() method comparison."""
    print("\n\nTesting posterior() method comparison")
    print("=" * 50)
    
    # Create synthetic dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.linspace(0, 1, 10).unsqueeze(-1).double()
    true_y = torch.sin(2 * torch.pi * X.squeeze())
    noise = torch.randn(10) * 0.1
    y = (true_y + noise).unsqueeze(-1)
    
    # Create models
    standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Set identical parameters
    lengthscale = torch.tensor(0.2, dtype=torch.float64)
    noise = torch.tensor(0.01, dtype=torch.float64)
    mean_constant = torch.tensor(0.1, dtype=torch.float64)
    
    set_identical_parameters(standard_gp, botorch_gp, lengthscale, noise, mean_constant)
    
    print(f"Set parameters:")
    print(f"  lengthscale: {lengthscale.item()}")
    print(f"  noise: {noise.item()}")
    print(f"  mean_constant: {mean_constant.item()}")
    
    # Test points
    test_X = torch.linspace(-0.3, 1.3, 20).unsqueeze(-1).double()
    
    # Put both models in eval mode
    standard_gp.eval()
    botorch_gp.eval()
    
    # Test posterior() method
    with torch.no_grad():
        standard_posterior = standard_gp.posterior(test_X)
        botorch_posterior = botorch_gp.posterior(test_X)
    
    # Extract means and variances
    standard_mean = standard_posterior.mean.squeeze()
    standard_var = standard_posterior.variance.squeeze()
    
    botorch_mean = botorch_posterior.mean.squeeze()
    botorch_var = botorch_posterior.variance.squeeze()
    
    print(f"\nPosterior() comparison:")
    print(f"Mean shapes - StandardGP: {standard_mean.shape}, BoTorch: {botorch_mean.shape}")
    print(f"Variance shapes - StandardGP: {standard_var.shape}, BoTorch: {botorch_var.shape}")
    
    # Check mean differences
    mean_diff = torch.abs(standard_mean - botorch_mean)
    max_mean_diff = torch.max(mean_diff)
    rel_mean_diff = max_mean_diff / torch.max(torch.abs(botorch_mean))
    
    print(f"Max absolute mean difference: {max_mean_diff.item():.2e}")
    print(f"Max relative mean difference: {rel_mean_diff.item():.2%}")
    
    # Check variance differences
    var_diff = torch.abs(standard_var - botorch_var)
    max_var_diff = torch.max(var_diff)
    rel_var_diff = max_var_diff / torch.max(torch.abs(botorch_var))
    
    print(f"Max absolute variance difference: {max_var_diff.item():.2e}")
    print(f"Max relative variance difference: {rel_var_diff.item():.2%}")
    
    # Test passes if differences are < 1%
    mean_pass = rel_mean_diff < 0.01
    var_pass = rel_var_diff < 0.01
    
    print(f"\nPosterior test results:")
    print(f"  Mean comparison: {'PASS ✓' if mean_pass else 'FAIL ✗'}")
    print(f"  Variance comparison: {'PASS ✓' if var_pass else 'FAIL ✗'}")
    
    return mean_pass and var_pass


def test_multiple_parameter_sets():
    """Test with multiple different parameter sets."""
    print("\n\nTesting multiple parameter sets")
    print("=" * 50)
    
    # Create dataset
    torch.manual_seed(42)
    X = torch.linspace(0, 1, 8).unsqueeze(-1).double()
    y = (torch.sin(2 * torch.pi * X.squeeze()) + torch.randn(8) * 0.05).unsqueeze(-1)
    
    parameter_sets = [
        {"lengthscale": 0.1, "noise": 0.001, "mean": 0.0},
        {"lengthscale": 0.5, "noise": 0.1, "mean": 1.0},
        {"lengthscale": 1.0, "noise": 0.05, "mean": -0.5},
        {"lengthscale": 0.05, "noise": 0.2, "mean": 2.0},
    ]
    
    all_pass = True
    test_X = torch.linspace(-0.2, 1.2, 12).unsqueeze(-1).double()
    
    for i, params in enumerate(parameter_sets):
        print(f"\nParameter set {i+1}: {params}")
        
        # Create models
        standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
        botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
        
        # Set parameters
        lengthscale = torch.tensor(params["lengthscale"], dtype=torch.float64)
        noise = torch.tensor(params["noise"], dtype=torch.float64)
        mean_constant = torch.tensor(params["mean"], dtype=torch.float64)
        
        set_identical_parameters(standard_gp, botorch_gp, lengthscale, noise, mean_constant)
        
        # Test posterior
        standard_gp.eval()
        botorch_gp.eval()
        
        with torch.no_grad():
            standard_posterior = standard_gp.posterior(test_X)
            botorch_posterior = botorch_gp.posterior(test_X)
        
        # Compare
        standard_mean = standard_posterior.mean.squeeze()
        standard_var = standard_posterior.variance.squeeze()
        botorch_mean = botorch_posterior.mean.squeeze()
        botorch_var = botorch_posterior.variance.squeeze()
        
        mean_rel_diff = torch.max(torch.abs(standard_mean - botorch_mean)) / torch.max(torch.abs(botorch_mean))
        var_rel_diff = torch.max(torch.abs(standard_var - botorch_var)) / torch.max(torch.abs(botorch_var))
        
        mean_pass = mean_rel_diff < 0.01
        var_pass = var_rel_diff < 0.01
        set_pass = mean_pass and var_pass
        
        print(f"  Mean rel diff: {mean_rel_diff.item():.2%} {'✓' if mean_pass else '✗'}")
        print(f"  Var rel diff: {var_rel_diff.item():.2%} {'✓' if var_pass else '✗'}")
        print(f"  Set {i+1}: {'PASS' if set_pass else 'FAIL'}")
        
        all_pass = all_pass and set_pass
    
    return all_pass


def run_all_tests():
    """Run all comparison tests."""
    print("StandardGP vs BoTorch SingleTaskGP Forward/Posterior Comparison")
    print("=" * 70)
    
    test1_pass = test_forward_comparison()
    test2_pass = test_posterior_comparison()
    test3_pass = test_multiple_parameter_sets()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS:")
    print(f"  Forward() comparison: {'PASS ✅' if test1_pass else 'FAIL ❌'}")
    print(f"  Posterior() comparison: {'PASS ✅' if test2_pass else 'FAIL ❌'}")
    print(f"  Multiple parameter sets: {'PASS ✅' if test3_pass else 'FAIL ❌'}")
    
    overall_pass = test1_pass and test2_pass and test3_pass
    print(f"  Overall: {'PASS ✅' if overall_pass else 'FAIL ❌'}")
    
    if overall_pass:
        print("\n🎉 StandardGP produces identical outputs to BoTorch SingleTaskGP!")
    else:
        print("\n❌ StandardGP has implementation differences from BoTorch SingleTaskGP")
    
    return overall_pass


if __name__ == "__main__":
    success = run_all_tests()
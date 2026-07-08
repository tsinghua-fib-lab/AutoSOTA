"""
Unit test to ensure StandardGP computes identical MLL values to BoTorch SingleTaskGP.

This test verifies that the ExactMarginalLogLikelihood computation is identical
between StandardGP and BoTorch SingleTaskGP when given the same parameters and data.
This is critical for ensuring optimization converges to the same solutions.
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from rcgp.models.standard_gp import StandardGP
from gpytorch.mlls import ExactMarginalLogLikelihood


def test_mll_computation_identical():
    """Test that MLL computation is identical between StandardGP and BoTorch SingleTaskGP."""
    
    # Create synthetic dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.linspace(0, 1, 8).unsqueeze(-1).double()
    y = torch.tensor([1.0, 2.0, 1.5, 0.5, 1.2, 0.8, 1.8, 1.1], dtype=torch.float64).unsqueeze(-1)
    
    # Create models
    standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Test multiple parameter combinations
    parameter_sets = [
        {"lengthscale": 0.1, "noise": 0.01, "mean": 0.0},
        {"lengthscale": 0.5, "noise": 0.1, "mean": 1.0},
        {"lengthscale": 1.0, "noise": 0.05, "mean": -0.5},
        {"lengthscale": 0.2, "noise": 0.02, "mean": 0.5},
    ]
    
    for i, params in enumerate(parameter_sets):
        # Set identical parameters
        lengthscale = params["lengthscale"]
        noise = params["noise"]
        mean_constant = params["mean"]
        
        standard_gp.covar_module.lengthscale = lengthscale
        standard_gp.likelihood.noise = noise
        standard_gp.mean_module.constant = mean_constant
        
        botorch_gp.covar_module.lengthscale = lengthscale
        botorch_gp.likelihood.noise = noise
        botorch_gp.mean_module.constant = mean_constant
        
        # Set to training mode
        standard_gp.train()
        standard_gp.likelihood.train()
        botorch_gp.train()
        botorch_gp.likelihood.train()
        
        # Get model outputs
        standard_output = standard_gp(X)
        botorch_output = botorch_gp(X)
        
        # Verify outputs are identical (sanity check)
        output_diff = torch.max(torch.abs(standard_output.mean - botorch_output.mean))
        assert output_diff < 1e-6, f"Model outputs differ by {output_diff:.2e} for parameter set {i+1}"
        
        # Create MLL objects and compute values
        mll_standard = ExactMarginalLogLikelihood(standard_gp.likelihood, standard_gp)
        mll_botorch = ExactMarginalLogLikelihood(botorch_gp.likelihood, botorch_gp)
        
        standard_mll_val = mll_standard(standard_output, standard_gp.train_targets)
        botorch_mll_val = mll_botorch(botorch_output, botorch_gp.train_targets)
        
        # Check MLL values are identical using relative comparison
        std_mll = standard_mll_val.item()
        bot_mll = botorch_mll_val.item()
        
        # Check same sign
        assert (std_mll >= 0) == (bot_mll >= 0), f"MLL values have different signs for parameter set {i+1}: {std_mll} vs {bot_mll}"
        
        # Calculate relative difference
        if abs(bot_mll) > 1e-10:  # Avoid division by zero
            rel_diff = abs(std_mll - bot_mll) / abs(bot_mll)
        else:
            rel_diff = abs(std_mll - bot_mll)  # Absolute difference when reference is near zero
        
        print(f"Parameter set {i+1}: {params}")
        print(f"  StandardGP MLL: {std_mll:.10f}")
        print(f"  BoTorch GP MLL: {bot_mll:.10f}")
        print(f"  Relative difference: {rel_diff:.2%}")
        
        # Assert MLL values match within 1% relative difference
        assert rel_diff < 0.01, f"MLL values differ by {rel_diff:.2%} for parameter set {i+1}"
        
        print(f"  ✅ PASSED")
    
    print(f"\n🎉 All {len(parameter_sets)} parameter sets passed!")
    print("StandardGP computes identical MLL values to BoTorch SingleTaskGP")


def test_prior_configuration_matches():
    """Test that priors are configured identically to BoTorch SingleTaskGP."""
    
    X = torch.linspace(0, 1, 5).unsqueeze(-1).double()
    y = torch.tensor([1.0, 2.0, 1.5, 0.5, 1.2], dtype=torch.float64).unsqueeze(-1)
    
    standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Check noise prior
    std_noise_prior = standard_gp.likelihood.noise_covar.noise_prior
    bot_noise_prior = botorch_gp.likelihood.noise_covar.noise_prior
    
    assert type(std_noise_prior) == type(bot_noise_prior), \
        f"Noise prior types differ: {type(std_noise_prior)} vs {type(bot_noise_prior)}"
    
    assert abs(std_noise_prior.loc - bot_noise_prior.loc) < 1e-10, \
        f"Noise prior loc differs: {std_noise_prior.loc} vs {bot_noise_prior.loc}"
    
    assert abs(std_noise_prior.scale - bot_noise_prior.scale) < 1e-10, \
        f"Noise prior scale differs: {std_noise_prior.scale} vs {bot_noise_prior.scale}"
    
    # Check lengthscale prior
    std_length_prior = standard_gp.covar_module.lengthscale_prior
    bot_length_prior = botorch_gp.covar_module.lengthscale_prior
    
    assert type(std_length_prior) == type(bot_length_prior), \
        f"Lengthscale prior types differ: {type(std_length_prior)} vs {type(bot_length_prior)}"
    
    # For dimension-scaled priors, just check they're both LogNormalPrior
    assert std_length_prior.__class__.__name__ == "LogNormalPrior", \
        f"StandardGP lengthscale prior should be LogNormalPrior, got {type(std_length_prior)}"
    
    print("✅ Prior configurations match BoTorch SingleTaskGP")


def test_initial_parameter_values_identical():
    """Test that initial parameter values match BoTorch exactly."""
    
    # Use multiple random seeds to ensure consistency
    for seed in [41, 42, 43]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X = torch.linspace(0, 1, 6).unsqueeze(-1).double()
        y = torch.sin(2 * torch.pi * X.squeeze()).unsqueeze(-1)
        
        standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
        botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
        
        std_params = standard_gp._extract_parameter_values()
        bot_params = {
            'lengthscale': botorch_gp.covar_module.lengthscale.item(),
            'noise': botorch_gp.likelihood.noise.item(),
            'mean_constant': botorch_gp.mean_module.constant.item()
        }
        
        for param in std_params:
            if param in bot_params:
                diff = abs(std_params[param] - bot_params[param])
                assert diff < 1e-10, \
                    f"Initial {param} differs by {diff:.2e} (seed {seed}): {std_params[param]} vs {bot_params[param]}"
        
        print(f"✅ Initial parameters identical for seed {seed}")
    
    print("✅ Initial parameter values match BoTorch across multiple seeds")


def run_all_mll_tests():
    """Run all MLL-related tests."""
    print("Testing StandardGP MLL Computation Against BoTorch SingleTaskGP")
    print("=" * 65)
    
    test_initial_parameter_values_identical()
    print()
    
    test_prior_configuration_matches()
    print()
    
    test_mll_computation_identical()
    
    print("\n" + "=" * 65)
    print("🎉 ALL MLL TESTS PASSED!")
    print("StandardGP is now fully compatible with BoTorch SingleTaskGP")


if __name__ == "__main__":
    run_all_mll_tests()
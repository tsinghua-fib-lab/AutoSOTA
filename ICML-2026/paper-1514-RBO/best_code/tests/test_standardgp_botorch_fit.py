"""
Test that StandardGP can be fitted using BoTorch's fit_gpytorch_mll 
and produces identical results to its own fit_hyperparameters method.
"""

import torch
import numpy as np
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from rcgp.models.standard_gp import StandardGP


def test_standardgp_with_botorch_fit():
    """Test StandardGP optimization using BoTorch's fit_gpytorch_mll."""
    
    # Create synthetic dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_points = 30
    X = torch.linspace(0, 1, n_points).unsqueeze(-1).double()
    true_y = torch.sin(2 * torch.pi * X.squeeze())
    noise = torch.randn(n_points) * 0.1
    y = (true_y + noise).unsqueeze(-1)
    
    print("Testing StandardGP with different optimization methods")
    print("=" * 60)
    print(f"Dataset: {n_points} points, noise_std=0.1")
    
    # Create two identical StandardGP models
    gp_adam = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    gp_botorch = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Get initial parameters (should be identical)
    initial_adam = gp_adam._extract_parameter_values()
    initial_botorch = gp_botorch._extract_parameter_values()
    
    print("\nInitial parameters (should be identical):")
    print(f"Adam GP: {initial_adam}")
    print(f"BoTorch GP: {initial_botorch}")
    
    # Verify initial parameters are the same
    for param in initial_adam:
        if param in initial_botorch:
            assert abs(initial_adam[param] - initial_botorch[param]) < 1e-10, \
                f"Initial {param} mismatch!"
    
    # Fit using Adam optimizer (StandardGP's method)
    print("\nFitting with Adam optimizer...")
    fitted_adam = gp_adam.fit_hyperparameters(
        learning_rate=0.1,
        max_iterations=500,
        verbose=False
    )
    
    # Fit using BoTorch's fit_gpytorch_mll
    print("Fitting with BoTorch's fit_gpytorch_mll...")
    gp_botorch.train()  # Set to training mode
    gp_botorch.likelihood.train()
    mll = ExactMarginalLogLikelihood(gp_botorch.likelihood, gp_botorch)
    fit_gpytorch_mll(mll)
    fitted_botorch = gp_botorch._extract_parameter_values()
    
    # Compare fitted parameters
    print("\nFitted parameters:")
    print(f"Adam: {fitted_adam}")
    print(f"BoTorch: {fitted_botorch}")
    
    print("\nParameter differences:")
    all_pass = True
    for param in fitted_adam:
        if param in fitted_botorch:
            abs_diff = abs(fitted_adam[param] - fitted_botorch[param])
            rel_diff = abs_diff / max(abs(fitted_adam[param]), abs(fitted_botorch[param]))
            
            # Use 5% threshold for optimizer differences
            threshold = 0.05
            status = "✓" if rel_diff < threshold else "✗"
            print(f"  {param}: {rel_diff:.2%} {status}")
            
            if rel_diff >= threshold:
                all_pass = False
    
    # Compare final MLL values
    gp_adam.eval()
    gp_botorch.eval()
    
    mll_adam = ExactMarginalLogLikelihood(gp_adam.likelihood, gp_adam)
    mll_botorch = ExactMarginalLogLikelihood(gp_botorch.likelihood, gp_botorch)
    
    with torch.no_grad():
        output_adam = gp_adam(X)
        final_mll_adam = mll_adam(output_adam, y.squeeze()).item()
        
        output_botorch = gp_botorch(X)
        final_mll_botorch = mll_botorch(output_botorch, y.squeeze()).item()
    
    print(f"\nFinal MLL values:")
    print(f"  Adam: {final_mll_adam:.6f}")
    print(f"  BoTorch: {final_mll_botorch:.6f}")
    print(f"  Difference: {abs(final_mll_adam - final_mll_botorch):.6f}")
    
    # The MLL values should be very close (both should find similar optima)
    mll_close = abs(final_mll_adam - final_mll_botorch) < 0.01
    
    if all_pass and mll_close:
        print("\n✅ Test PASSED: StandardGP works correctly with both optimizers!")
    else:
        print("\n❌ Test FAILED: Significant differences between optimizers")
    
    return all_pass and mll_close


def test_kernel_structure():
    """Verify what kernel structure StandardGP is using."""
    
    X = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64)
    y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    
    gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    
    print("\nKernel Structure Test:")
    print("=" * 60)
    print(f"Kernel type: {gp.covar_module.__class__.__name__}")
    print(f"Kernel: {gp.covar_module}")
    
    # Check what parameters are available
    params = gp._extract_parameter_values()
    print(f"\nAvailable parameters: {list(params.keys())}")
    
    # Check if we have outputscale
    has_outputscale = 'outputscale' in params
    print(f"Has outputscale parameter: {has_outputscale}")
    
    if not has_outputscale:
        print("Note: Using plain RBFKernel without ScaleKernel wrapper")
        print("      (This matches BoTorch's SingleTaskGP)")
    else:
        print("Note: Using ScaleKernel(RBFKernel) with outputscale")
    
    return has_outputscale


def test_standardgp_vs_botorch_both_with_botorch_fit():
    """Compare StandardGP vs BoTorch SingleTaskGP, both fitted with BoTorch's fit_gpytorch_mll."""
    
    # Create synthetic dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_points = 30
    X = torch.linspace(0, 1, n_points).unsqueeze(-1).double()
    true_y = torch.sin(2 * torch.pi * X.squeeze())
    noise = torch.randn(n_points) * 0.1
    y = (true_y + noise).unsqueeze(-1)
    
    print("\nComparing StandardGP vs BoTorch SingleTaskGP (both with BoTorch fit)")
    print("=" * 60)
    print(f"Dataset: {n_points} points, noise_std=0.1")
    
    # Create StandardGP without outcome standardization
    standard_gp = StandardGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Create BoTorch SingleTaskGP without outcome standardization
    from botorch.models import SingleTaskGP
    botorch_gp = SingleTaskGP(train_X=X, train_Y=y, outcome_transform=None)
    
    # Get initial parameters
    initial_standard = standard_gp._extract_parameter_values()
    initial_botorch = {}
    if hasattr(botorch_gp.covar_module, 'base_kernel'):
        initial_botorch['lengthscale'] = botorch_gp.covar_module.base_kernel.lengthscale.item()
    else:
        initial_botorch['lengthscale'] = botorch_gp.covar_module.lengthscale.item()
    initial_botorch['noise'] = botorch_gp.likelihood.noise.item()
    if hasattr(botorch_gp.mean_module, 'constant'):
        initial_botorch['mean_constant'] = botorch_gp.mean_module.constant.item()
    
    print("\nInitial parameters:")
    print(f"StandardGP: {initial_standard}")
    print(f"BoTorch GP: {initial_botorch}")
    
    # Fit StandardGP using BoTorch's fit_gpytorch_mll
    print("\nFitting StandardGP with BoTorch's fit_gpytorch_mll...")
    standard_gp.train()  # Set to training mode
    standard_gp.likelihood.train()
    mll_standard = ExactMarginalLogLikelihood(standard_gp.likelihood, standard_gp)
    fit_gpytorch_mll(mll_standard)
    fitted_standard = standard_gp._extract_parameter_values()
    
    # Fit BoTorch GP using BoTorch's fit_gpytorch_mll
    print("Fitting BoTorch SingleTaskGP with BoTorch's fit_gpytorch_mll...")
    botorch_gp.train()  # Set to training mode
    botorch_gp.likelihood.train()
    mll_botorch = ExactMarginalLogLikelihood(botorch_gp.likelihood, botorch_gp)
    fit_gpytorch_mll(mll_botorch)
    fitted_botorch = {}
    if hasattr(botorch_gp.covar_module, 'base_kernel'):
        fitted_botorch['lengthscale'] = botorch_gp.covar_module.base_kernel.lengthscale.item()
    else:
        fitted_botorch['lengthscale'] = botorch_gp.covar_module.lengthscale.item()
    fitted_botorch['noise'] = botorch_gp.likelihood.noise.item()
    if hasattr(botorch_gp.mean_module, 'constant'):
        fitted_botorch['mean_constant'] = botorch_gp.mean_module.constant.item()
    
    print("\nFitted parameters:")
    print(f"StandardGP: {fitted_standard}")
    print(f"BoTorch GP: {fitted_botorch}")
    
    print("\nParameter differences:")
    all_pass = True
    for param in fitted_standard:
        if param in fitted_botorch:
            abs_diff = abs(fitted_standard[param] - fitted_botorch[param])
            rel_diff = abs_diff / max(abs(fitted_standard[param]), abs(fitted_botorch[param]))
            
            # All parameters should match within 1%
            threshold = 0.01
            
            status = "✓" if rel_diff < threshold else "✗"
            print(f"  {param}: {rel_diff:.2%} {status}")
            
            if rel_diff >= threshold:
                all_pass = False
    
    # Compare final MLL values
    standard_gp.eval()
    botorch_gp.eval()
    
    with torch.no_grad():
        output_standard = standard_gp(X)
        final_mll_standard = mll_standard(output_standard, y.squeeze()).item()
        
        output_botorch = botorch_gp(X)
        final_mll_botorch = mll_botorch(output_botorch, y.squeeze()).item()
    
    print(f"\nFinal MLL values:")
    print(f"  StandardGP: {final_mll_standard:.6f}")
    print(f"  BoTorch GP: {final_mll_botorch:.6f}")
    print(f"  Difference: {abs(final_mll_standard - final_mll_botorch):.6f}")
    
    mll_close = abs(final_mll_standard - final_mll_botorch) < 0.01
    
    if all_pass and mll_close:
        print("\n✅ Test PASSED: StandardGP and BoTorch SingleTaskGP produce identical results!")
    else:
        print("\n❌ Test FAILED: Differences between StandardGP and BoTorch SingleTaskGP")
    
    return all_pass and mll_close


def run_all_tests():
    """Run all tests."""
    print("StandardGP Optimization Method Comparison Tests")
    print("=" * 60)
    
    # Test kernel structure
    has_outputscale = test_kernel_structure()
    
    print("\n")
    
    # Test optimization methods (StandardGP with Adam vs BoTorch fit)
    test1_passed = test_standardgp_with_botorch_fit()
    
    # Test StandardGP vs BoTorch GP (both with BoTorch fit)
    test2_passed = test_standardgp_vs_botorch_both_with_botorch_fit()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Kernel: {'ScaleKernel(RBFKernel)' if has_outputscale else 'RBFKernel (plain)'}")
    print(f"  Outputscale parameter: {'Yes' if has_outputscale else 'No'}")
    print(f"  StandardGP Adam vs BoTorch fit: {'PASSED ✅' if test1_passed else 'FAILED ❌'}")
    print(f"  StandardGP vs BoTorch (both BoTorch fit): {'PASSED ✅' if test2_passed else 'FAILED ❌'}")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = run_all_tests()
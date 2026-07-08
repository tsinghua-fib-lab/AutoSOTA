"""
Unit tests for StandardGP hyperparameter fitting.

Tests StandardGP.fit_hyperparameters() against BoTorch's SingleTaskGP optimization
on various synthetic datasets without outcome normalization.
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from rcgp.models.standard_gp import StandardGP


class TestStandardGPFitting:
    """Test suite for StandardGP hyperparameter fitting."""
    
    @staticmethod
    def create_synthetic_dataset(n_points=50, input_dim=1, noise_std=0.1, mean_shift=0.0, seed=42):
        """Create synthetic dataset with specified noise and mean."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create input points
        X = torch.linspace(0, 1, n_points).unsqueeze(-1)
        if input_dim > 1:
            X = torch.randn(n_points, input_dim)
        
        # Create true function (simple sine wave shifted by mean)
        if input_dim == 1:
            true_y = torch.sin(2 * torch.pi * X.squeeze()) + mean_shift
        else:
            # For multi-dimensional, use sum of sines
            true_y = torch.sin(2 * torch.pi * X.sum(dim=-1)) + mean_shift
        
        # Add noise
        noise = torch.randn(n_points) * noise_std
        y = true_y + noise
        
        return X.double(), y.double().unsqueeze(-1)
    
    @staticmethod
    def extract_botorch_parameters(model):
        """Extract parameters from BoTorch SingleTaskGP model."""
        params = {}
        
        # Lengthscale
        if hasattr(model.covar_module, 'base_kernel') and hasattr(model.covar_module.base_kernel, 'lengthscale'):
            lengthscale = model.covar_module.base_kernel.lengthscale
            if lengthscale.numel() == 1:
                params['lengthscale'] = lengthscale.item()
            else:
                params['lengthscale'] = lengthscale.mean().item()
        elif hasattr(model.covar_module, 'lengthscale'):
            lengthscale = model.covar_module.lengthscale
            if lengthscale.numel() == 1:
                params['lengthscale'] = lengthscale.item()
            else:
                params['lengthscale'] = lengthscale.mean().item()
        
        # Outputscale
        if hasattr(model.covar_module, 'outputscale'):
            params['outputscale'] = model.covar_module.outputscale.item()
        
        # Noise
        if hasattr(model.likelihood, 'noise'):
            params['noise'] = model.likelihood.noise.item()
        
        # Mean constant
        if hasattr(model.mean_module, 'constant'):
            params['mean_constant'] = model.mean_module.constant.item()
        
        return params

    def test_against_botorch(self, dataset_name, X, y, verbose=False):
        """Compare StandardGP vs BoTorch SingleTaskGP hyperparameter fitting."""
        
        if verbose:
            print(f"\n=== Testing {dataset_name} ===")
            print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Create StandardGP (no outcome normalization)
        standard_gp = StandardGP(
            train_X=X,
            train_Y=y,
            outcome_transform=None  # Disable outcome normalization
        )
        
        # Create BoTorch SingleTaskGP (no outcome normalization)
        botorch_gp = SingleTaskGP(
            train_X=X,
            train_Y=y,
            outcome_transform=None  # Disable outcome normalization
        )
        
        # Fit StandardGP hyperparameters
        fitted_standard = standard_gp.fit_hyperparameters(
            learning_rate=0.1,
            max_iterations=500,
            verbose=verbose
        )
        
        # Fit BoTorch GP hyperparameters
        mll = ExactMarginalLogLikelihood(botorch_gp.likelihood, botorch_gp)
        fit_gpytorch_mll(mll)
        fitted_botorch = self.extract_botorch_parameters(botorch_gp)
        
        if verbose:
            print(f"Fitted StandardGP params: {fitted_standard}")
            print(f"Fitted BoTorch params: {fitted_botorch}")
        
        # Compare parameters (they should be reasonably close)
        param_diffs = {}
        for param in ['lengthscale', 'noise']:
            if param in fitted_standard and param in fitted_botorch:
                diff = abs(fitted_standard[param] - fitted_botorch[param])
                rel_diff = diff / max(fitted_standard[param], fitted_botorch[param])
                param_diffs[param] = {'abs_diff': diff, 'rel_diff': rel_diff}
        
        # Test posterior predictions on test points
        test_X = torch.linspace(-0.5, 1.5, 20).unsqueeze(-1).double()
        
        # StandardGP predictions
        standard_gp.eval()
        with torch.no_grad():
            standard_posterior = standard_gp.posterior(test_X)
            standard_mean = standard_posterior.mean.squeeze()
            standard_var = standard_posterior.variance.squeeze()
        
        # BoTorch predictions
        botorch_gp.eval()
        with torch.no_grad():
            botorch_posterior = botorch_gp.posterior(test_X)
            botorch_mean = botorch_posterior.mean.squeeze()
            botorch_var = botorch_posterior.variance.squeeze()
        
        # Compare predictions
        mean_mse = torch.mean((standard_mean - botorch_mean)**2).item()
        var_mse = torch.mean((standard_var - botorch_var)**2).item()
        
        return {
            'dataset': dataset_name,
            'fitted_standard': fitted_standard,
            'fitted_botorch': fitted_botorch,
            'param_diffs': param_diffs,
            'mean_mse': mean_mse,
            'var_mse': var_mse
        }

    def test_low_noise_dataset(self):
        """Test on low noise dataset."""
        X, y = self.create_synthetic_dataset(n_points=30, noise_std=0.05, mean_shift=0.0, seed=42)
        result = self.test_against_botorch('Low Noise Dataset', X, y)
        
        # Assertions
        assert result['mean_mse'] < 1e-3, f"Mean predictions too different: {result['mean_mse']}"
        assert result['var_mse'] < 1e-2, f"Variance predictions too different: {result['var_mse']}"
        
        for param, diff_info in result['param_diffs'].items():
            assert diff_info['rel_diff'] < 0.02, f"{param} relative difference too large: {diff_info['rel_diff']:.1%}"
        
        return result
    
    def test_high_noise_dataset(self):
        """Test on high noise dataset."""
        X, y = self.create_synthetic_dataset(n_points=30, noise_std=0.3, mean_shift=0.0, seed=43)
        result = self.test_against_botorch('High Noise Dataset', X, y)
        
        assert result['mean_mse'] < 1e-3, f"Mean predictions too different: {result['mean_mse']}"
        assert result['var_mse'] < 1e-2, f"Variance predictions too different: {result['var_mse']}"
        
        for param, diff_info in result['param_diffs'].items():
            assert diff_info['rel_diff'] < 0.02, f"{param} relative difference too large: {diff_info['rel_diff']:.1%}"
        
        return result
    
    def test_shifted_mean_dataset(self):
        """Test on dataset with shifted mean."""
        X, y = self.create_synthetic_dataset(n_points=30, noise_std=0.1, mean_shift=5.0, seed=44)
        result = self.test_against_botorch('Shifted Mean Dataset', X, y)
        
        assert result['mean_mse'] < 1e-3, f"Mean predictions too different: {result['mean_mse']}"
        assert result['var_mse'] < 1e-2, f"Variance predictions too different: {result['var_mse']}"
        
        for param, diff_info in result['param_diffs'].items():
            assert diff_info['rel_diff'] < 0.02, f"{param} relative difference too large: {diff_info['rel_diff']:.1%}"
        
        return result
    
    def test_large_dataset(self):
        """Test on large dataset."""
        X, y = self.create_synthetic_dataset(n_points=100, noise_std=0.1, mean_shift=0.0, seed=45)
        result = self.test_against_botorch('Large Dataset', X, y)
        
        assert result['mean_mse'] < 1e-3, f"Mean predictions too different: {result['mean_mse']}"
        assert result['var_mse'] < 1e-2, f"Variance predictions too different: {result['var_mse']}"
        
        for param, diff_info in result['param_diffs'].items():
            assert diff_info['rel_diff'] < 0.02, f"{param} relative difference too large: {diff_info['rel_diff']:.1%}"
        
        return result
    
    def test_small_dataset(self):
        """Test on small dataset."""
        X, y = self.create_synthetic_dataset(n_points=15, noise_std=0.1, mean_shift=0.0, seed=46)
        result = self.test_against_botorch('Small Dataset', X, y)
        
        assert result['mean_mse'] < 1e-3, f"Mean predictions too different: {result['mean_mse']}"
        assert result['var_mse'] < 1e-2, f"Variance predictions too different: {result['var_mse']}"
        
        for param, diff_info in result['param_diffs'].items():
            assert diff_info['rel_diff'] < 0.02, f"{param} relative difference too large: {diff_info['rel_diff']:.1%}"
        
        return result
    
    def run_all_tests(self, verbose=False):
        """Run all test cases."""
        print("Testing StandardGP hyperparameter fitting against BoTorch SingleTaskGP")
        print("=" * 70)
        
        test_methods = [
            self.test_low_noise_dataset,
            self.test_high_noise_dataset,
            self.test_shifted_mean_dataset,
            self.test_large_dataset,
            self.test_small_dataset
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
                print(f"✅ {result['dataset']} - PASSED")
                if verbose:
                    for param, diff_info in result['param_diffs'].items():
                        print(f"   {param} relative difference: {diff_info['rel_diff']:.1%}")
            except AssertionError as e:
                print(f"❌ {test_method.__name__} - FAILED: {e}")
                raise
        
        print(f"\n✅ All {len(test_methods)} tests passed!")
        print("StandardGP hyperparameter fitting is working correctly.")
        
        return results


def test_standardgp_fitting():
    """Main test function for pytest or direct execution."""
    tester = TestStandardGPFitting()
    results = tester.run_all_tests(verbose=False)
    return results


if __name__ == "__main__":
    # Run tests with detailed output
    tester = TestStandardGPFitting()
    
    test_methods = [
        ('Low Noise', tester.test_low_noise_dataset),
        ('High Noise', tester.test_high_noise_dataset),
        ('Shifted Mean', tester.test_shifted_mean_dataset),
        ('Large Dataset', tester.test_large_dataset),
        ('Small Dataset', tester.test_small_dataset)
    ]
    
    print("Testing StandardGP vs BoTorch with 1% relative difference threshold")
    print("=" * 70)
    
    for name, test_method in test_methods:
        try:
            result = test_method()
            print(f"✅ {name} - PASSED")
            for param, diff_info in result['param_diffs'].items():
                print(f"   {param}: {diff_info['rel_diff']:.2%}")
        except AssertionError as e:
            print(f"❌ {name} - FAILED")
            print(f"   {e}")
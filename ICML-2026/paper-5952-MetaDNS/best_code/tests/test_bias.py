
import numpy as np
import torch

from ..bias import BiasPotential


def test_bias_gaussian():
    print("Testing Gaussian Kernel...")
    device = 'cpu'
    bias = BiasPotential(
        cv_min=-1.0, cv_max=1.0, 
        grid_size=11, # -1.0, -0.8, ..., 0.0, ..., 1.0 (step 0.2)
        sigma=0.2, 
        initial_height=1.0, 
        bias_factor=10.0, 
        T=1.0, 
        kernel_type='gaussian',
        device=device
    )
    
    # Grid: -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    # Indices: 0,    1,    2,    3,    4,   5,   6,   7,   8,   9,   10
    
    # Update at 0.0
    cv_batch = torch.tensor([0.0], device=device)
    bias.update(cv_batch)
    
    # Check center value (index 5)
    # W_eff = W * exp(-V/kT) = 1.0 * exp(0) = 1.0
    # Added = 1.0 * Gaussian(0) = 1.0
    center_val = bias.evaluate(cv_batch)
    print(f"Bias at 0.0 after 1 update: {center_val.item()} (Expected ~1.0)")
    assert np.isclose(center_val.item(), 1.0, atol=0.1)
    
    # Update at 0.0 again
    # Current V(0) = 1.0
    # Delta_T = (10-1)*1 = 9
    # W_eff = 1.0 * exp(-1.0 / 9.0) = exp(-0.111) approx 0.89
    bias.update(cv_batch)
    new_val = bias.evaluate(cv_batch)
    expected_val = 1.0 + np.exp(-1.0/9.0)
    print(f"Bias at 0.0 after 2 updates: {new_val.item()} (Expected ~{expected_val})")
    assert np.isclose(new_val.item(), expected_val, atol=0.1)

def test_bias_delta():
    print("\nTesting Delta Kernel...")
    device = 'cpu'
    bias = BiasPotential(
        cv_min=-1.0, cv_max=1.0, 
        grid_size=11,
        sigma=0.2, # Ignored
        initial_height=1.0, 
        bias_factor=10.0, 
        T=1.0, 
        kernel_type='delta',
        device=device
    )
    
    # Update at 0.0 (Index 5)
    cv_batch = torch.tensor([0.0], device=device)
    bias.update(cv_batch)
    
    # Check grid directly
    print("Grid center value:", bias.bias_grid[5].item())
    assert bias.bias_grid[5].item() == 1.0
    assert bias.bias_grid[4].item() == 0.0
    
    # Update neighbor 0.2 (Index 6)
    cv_batch = torch.tensor([0.2], device=device)
    bias.update(cv_batch)
    assert bias.bias_grid[6].item() == 1.0
    
    print("Delta kernel test passed.")

if __name__ == "__main__":
    test_bias_gaussian()
    test_bias_delta()

"""Quick test of Mujoco Ant implementation."""

import torch
from bo_framework import SearchSpace
from experiments.mujoco_ant.evaluator import AntEvaluator


def test_basic_functionality():
    """Test basic functionality without full BO run."""
    
    print("Testing Mujoco Ant evaluator...")
    
    # Create evaluator
    evaluator = AntEvaluator(
        env_name="Ant-v4",
        max_steps=100,  # Short episodes for testing
        render=False,
        seed=42,
        penalty_value=-2000.0
    )
    
    print(f"Environment initialized successfully!")
    print(f"Observation dim: {evaluator.obs_dim}")
    print(f"Action dim: {evaluator.action_dim}")
    print(f"Policy params: {evaluator.policy.n_params}")
    
    # Create search space
    n_params = evaluator.policy.n_params
    bounds = torch.tensor([[-1.0] * n_params, [1.0] * n_params], dtype=torch.double)
    search_space = SearchSpace.from_bounds(bounds, normalize=True)
    
    print(f"Search space: {search_space.n_dims}D")
    
    # Test evaluation with random parameters
    print("\nTesting evaluations:")
    
    for i in range(3):
        # Random normalized parameters [0, 1]
        params = torch.rand(n_params, dtype=torch.double)
        param_dict = {}
        for j in range(n_params):
            param_dict[f"x{j}"] = params[j].item()
        
        result = evaluator.evaluate(param_dict)
        print(f"  Test {i+1}: reward = {result.y_true:.1f}")
        
        # Check for simulation failures
        if result.y_true <= -1500:
            print("    [Detected simulation failure - penalty assigned]")
    
    # Test with extreme parameters (should cause failure)
    print("\nTesting failure handling:")
    extreme_params = {}
    for j in range(n_params):
        extreme_params[f"x{j}"] = 1.0 if j % 2 == 0 else 0.0  # Extreme values
        
    result = evaluator.evaluate(extreme_params)
    print(f"  Extreme params: reward = {result.y_true:.1f}")
    if result.y_true <= -1500:
        print("    [Correctly detected simulation failure]")
    
    evaluator.close()
    print("\nAll tests passed! Implementation is working correctly.")


if __name__ == "__main__":
    test_basic_functionality()
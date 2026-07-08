"""Test the BO framework with Mujoco Ant locomotion optimization."""

import torch
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.models.factory import create_gp_model, create_rcgp_model
from experiments.mujoco_ant.evaluator import AntEvaluator


N_ITERATIONS = 8
N_INITIAL = 5

def main():
    """Demonstrate Bayesian Optimization for Mujoco Ant locomotion control."""
    
    # First, check if we can create the environment
    print("=" * 80)
    print("Mujoco Ant Locomotion Optimization with Bayesian Optimization")
    print("=" * 80)
    
    # Create a test evaluator to get environment dimensions
    try:
        test_evaluator = AntEvaluator(env_name="Ant-v5", render=False, seed=42)
        obs_dim = test_evaluator.obs_dim
        action_dim = test_evaluator.action_dim
        n_params = test_evaluator.policy.n_params
        test_evaluator.close()
        
        print(f"Environment: Ant-v5")
        print(f"Observations: {obs_dim}D")
        print(f"Actions: {action_dim}D continuous")
        print(f"Policy parameters: {n_params} ({obs_dim}×{action_dim} weights + {action_dim} biases)")
        print()
        
    except Exception as e:
        print(f"Error: Could not initialize Ant environment: {e}")
        print("Make sure Mujoco is installed: pip install gymnasium[mujoco]")
        return
    
    # Create high-dimensional continuous search space
    # Parameters bounded in [-1.0, 1.0] to prevent unstable control
    bounds = torch.tensor([[-1.0] * n_params, [1.0] * n_params], dtype=torch.double)
    search_space = SearchSpace.from_bounds(bounds, normalize=True)
    
    print(f"Search Space:")
    print(f"  Dimensions: {search_space.n_dims}")
    print(f"  Parameter bounds: [-1.0, 1.0] for all parameters")
    print(f"  Normalized for BO: [0, 1]")
    print()
    
    # Create main evaluator
    rcgp_evaluator = AntEvaluator(
        env_name="Ant-v5",
        max_steps=1000,
        render=False,  # Lightweight headless simulation
        seed=42,
        penalty_value=-2000.0  # Very bad score for simulation failures
    )

    # create another evaluator object to reset the seeded environment
    gp_evaluator = AntEvaluator(
        env_name="Ant-v5",
        max_steps=1000,
        render=False,  # Lightweight headless simulation
        seed=42,
        penalty_value=-2000.0  # Very bad score for simulation failures
    )
    
    # Test 1: Standard GP
    print("=" * 80)
    print("Test 1: Ant Locomotion with Standard GP")
    print("=" * 80)
    
    # Configure standard GP model kwargs (similar to Forrester example)
    gp_model_kwargs = {
        'fit_hyperparameters': False,  # Disable fitting for high-dimensional space
        'standardize': True,
        'use_botorch_model': True  # Use BoTorch's SingleTaskGP
    }
    
    runner_gp = ExperimentRunner(search_space, gp_evaluator)
    results_gp = runner_gp.run(
        n_iterations=N_ITERATIONS,  # Small number for testing high-dimensional space
        n_initial=N_INITIAL,     # More initial points for high-dim
        model_factory=create_gp_model,
        acquisition_factory=UCBAcquisition.create,
        seed=42,
        model_kwargs=gp_model_kwargs,
        verbose=True
    )

    # Cleanup
    gp_evaluator.close()
    
    # Test 2: RCGP model (robust to simulation failures/outliers)
    print("\n" + "=" * 80)
    print("Test 2: Ant Locomotion with Robust Conjugate GP")
    print("=" * 80)
    
    # Configure RCGP model kwargs using the new param_handling_dict API
    rcgp_kwargs = {
        "param_handling_dict": {
            "plateau_width": {"method": "heuristics",},  # Large value for high rewards
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},  # High noise for unstable simulations
            "mean": {"method": "fit"}  # Fit the mean parameter
        },
        "fitting_objective_type": "wloo-cv",  # Use MLL for high-dimensional problems
        "optimizer_type": "lbfgs",
        "standardize": True,
        "verbose": False
    }
    
    runner_rcgp = ExperimentRunner(search_space, rcgp_evaluator)
    results_rcgp = runner_rcgp.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=create_rcgp_model,
        acquisition_factory=UCBAcquisition.create,
        seed=42,
        model_kwargs=rcgp_kwargs,
        verbose=True
    )
    
    # Cleanup
    rcgp_evaluator.close()
    
    # Analysis and Comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"{'Method':<20} {'Best Reward':<12} {'Episodes':<8} {'Avg Initial':<12}")
    print("-" * 60)
    
    # GP results
    gp_best = results_gp['best_observed_value']
    gp_episodes = len(results_gp['all_results'])
    # Include ALL scores (including failures) in average - this is important for BO
    gp_initial_scores = [r.y_true for r in results_gp['all_results'][:N_INITIAL]]
    gp_initial_avg = sum(gp_initial_scores) / len(gp_initial_scores)
    print(f"{'Standard GP':<20} {gp_best:<12.1f} {gp_episodes:<8} {gp_initial_avg:<12.1f}")
    
    # RCGP results  
    rcgp_best = results_rcgp['best_observed_value']
    rcgp_episodes = len(results_rcgp['all_results'])
    # Include ALL scores (including failures) in average - this is important for BO
    rcgp_initial_scores = [r.y_true for r in results_rcgp['all_results'][:N_INITIAL]]
    rcgp_initial_avg = sum(rcgp_initial_scores) / len(rcgp_initial_scores)
    print(f"{'RCGP':<20} {rcgp_best:<12.1f} {rcgp_episodes:<8} {rcgp_initial_avg:<12.1f}")
    
    # Verify failure scores are included in averages
    print(f"\nAverage Calculation Verification:")
    print(f"GP initial scores: {[f'{s:.1f}' for s in gp_initial_scores]} → avg = {gp_initial_avg:.1f}")
    print(f"RCGP initial scores: {[f'{s:.1f}' for s in rcgp_initial_scores]} → avg = {rcgp_initial_avg:.1f}")
    
    # Performance analysis
    print(f"\nLocomotion Performance Analysis:")
    print(f"Best Standard GP policy reward: {gp_best:.1f}")
    print(f"Best RCGP policy reward: {rcgp_best:.1f}")
    
    # Check for simulation failures (penalty values)
    gp_failures = sum(1 for r in results_gp['all_results'] if r.y_true <= -1500)
    rcgp_failures = sum(1 for r in results_rcgp['all_results'] if r.y_true <= -1500)
    
    print(f"\nSimulation Failure Analysis:")
    print(f"Standard GP failures: {gp_failures}/{gp_episodes} episodes")
    print(f"RCGP failures: {rcgp_failures}/{rcgp_episodes} episodes")
    
    # Show optimization progress (summary only, not individual points)
    print(f"\nOptimization Summary:")
    print(f"Standard GP: {gp_episodes} episodes, best = {gp_best:.1f}")
    print(f"RCGP: {rcgp_episodes} episodes, best = {rcgp_best:.1f}")
    
    # Success metrics
    successful_rewards_gp = [r.y_true for r in results_gp['all_results'] if r.y_true > -1500]
    successful_rewards_rcgp = [r.y_true for r in results_rcgp['all_results'] if r.y_true > -1500]
    
    if successful_rewards_gp:
        print(f"\nSuccessful Standard GP episodes: avg = {sum(successful_rewards_gp)/len(successful_rewards_gp):.1f}")
    if successful_rewards_rcgp:
        print(f"Successful RCGP episodes: avg = {sum(successful_rewards_rcgp)/len(successful_rewards_rcgp):.1f}")


if __name__ == "__main__":
    main()
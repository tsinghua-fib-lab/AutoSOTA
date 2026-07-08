"""Experiment runner for orchestrating BO experiments."""

import torch
from torch.quasirandom import SobolEngine
from typing import Callable, Dict, Any, Optional

from .search_space import SearchSpace
from .evaluator import BaseEvaluator
from .optimizer import BOOptimizer
from .schedulers import BetaScheduler

# Import regret analysis utilities
from utilities.regret_analysis import (
    find_best_points,
    compute_corruption_statistics
)


class ExperimentRunner:
    """Orchestrates BO experiments with data I/O and model management.
    
    Phase 1a: Minimal implementation without checkpointing.
    """
    
    def __init__(self, 
                 search_space: SearchSpace,
                 evaluator: BaseEvaluator):
        """Initialize experiment runner.
        
        Args:
            search_space: SearchSpace defining the optimization domain
            evaluator: Function evaluator
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.bo_optimizer = BOOptimizer(search_space)
    
    def generate_initial_points(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        """Generate initial points using Sobol sequence.
        
        Args:
            n: Number of points to generate
            seed: Random seed for reproducibility
            
        Returns:
            Initial points tensor [n, n_dims] in the search space's coordinate system
        """
        sobol = SobolEngine(self.search_space.n_dims, scramble=True, seed=seed)
        # Generate in [0, 1]
        points_01 = sobol.draw(n).double()
        
        # Map to the actual search space bounds
        # If dimension.normalize=True, bounds are already [0,1]
        # If dimension.normalize=False, we need to scale to actual bounds
        bounds = self.search_space.bounds
        lower = bounds[0]
        upper = bounds[1]
        
        # Scale from [0,1] to the search space bounds
        points = points_01 * (upper - lower) + lower
        
        return points
    
    def run(self, 
            n_iterations: int,
            n_initial: int,
            model_factory: Callable,
            acquisition_factory: Callable,
            seed: Optional[int] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            beta_scheduler: Optional[BetaScheduler] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """Run the complete BO experiment.
        
        Args:
            n_iterations: Number of BO iterations
            n_initial: Number of initial points
            model_factory: Function to create model from (X, Y, **model_kwargs)
            acquisition_factory: Function to create acquisition from (model, search_space, **kwargs)
            seed: Random seed for reproducibility
            model_kwargs: Additional arguments for model factory
            beta_scheduler: Optional scheduler for UCB beta parameter
            verbose: Whether to print progress
            
        Returns:
            Dictionary with complete experiment results including:
                - X: All evaluated points
                - Y_observed: Values seen by BO
                - Y_true: True function values
                - Y_noisy: Values with observation noise
                - corruption_levels: Corruption applied at each point
                - final_model: Last fitted model
                - final_acquisition: Last acquisition function
                - best_observed_*: Best according to BO
                - best_true_*: Best according to true values
        """
        if verbose:
            print(f"Starting BO experiment with {n_initial} initial points and {n_iterations} iterations")
            print("-" * 70)
        
        # Generate and evaluate initial points
        if verbose:
            print("Evaluating initial points...")
        
        initial_results = []
        X_init = self.generate_initial_points(n_initial, seed)
        
        for i, x in enumerate(X_init):
            params = self.search_space.decode_point(x)
            result = self.evaluator.evaluate(params)
            initial_results.append(result)
            
            if verbose:
                if result.corruption != 0.0:
                    print(f"  Initial point {i+1}: f({list(params.values())}) = {result.y_true:.4f} "
                          f"(noisy: {result.y_noisy:.4f}, observed: {result.y_observed:.4f}, corruption: {result.corruption:+.4f})")
                elif result.noise != 0.0:
                    print(f"  Initial point {i+1}: f({list(params.values())}) = {result.y_true:.4f} (noisy: {result.y_noisy:.4f})")
                else:
                    print(f"  Initial point {i+1}: f({list(params.values())}) = {result.y_true:.4f}")
        
        # Run BO optimization
        if verbose:
            print("\nStarting BO iterations...")
            if beta_scheduler is not None:
                print(f"Using beta scheduler: {beta_scheduler.__class__.__name__}")
        
        bo_results = self.bo_optimizer.optimize_loop(
            initial_results,
            n_iterations,
            model_factory,
            acquisition_factory,
            self.evaluator,
            model_kwargs,
            beta_scheduler,
            verbose=verbose
        )
        
        # Find best points using regret_analysis utilities
        eval_results = bo_results['all_results']
        best_observed_info, best_true_info = find_best_points(eval_results)
        
        # Extract legacy format data for backward compatibility
        Y_observed = bo_results['Y_observed']
        Y_true = bo_results['Y_true']
        X = bo_results['X']
        
        # Compile complete results
        results = {
            # New clean interface
            'all_results': bo_results['all_results'],
            
            # Core data (backward compatibility)
            'X': X,
            'Y_observed': Y_observed,
            'Y_true': Y_true,
            'Y_noisy': bo_results['Y_noisy'],
            'corruption_levels': bo_results['corruption_levels'],
            
            # Models for analysis/plotting
            'final_model': bo_results['final_model'],
            'final_acquisition': bo_results['final_acquisition'],
            
            # Best points (observed - what BO thinks is best)
            'best_observed_value': best_observed_info['value'],
            'best_observed_point': X[best_observed_info['index']],
            'best_observed_params': best_observed_info['params'],
            
            # Best points (true - what is actually best)
            'best_true_value': best_true_info['value'],
            'best_true_point': X[best_true_info['index']],
            'best_true_params': best_true_info['params'],
            
            # Experiment metadata
            'n_iterations': n_iterations,
            'n_initial': n_initial,
            'seed': seed
        }
        
        if verbose:
            print("\n" + "=" * 70)
            print("Optimization complete!")
            print(f"Best observed value (BO perspective): {results['best_observed_value']:.6f}")
            print(f"Best observed params: {results['best_observed_params']}")
            print(f"Best true value (actual best): {results['best_true_value']:.6f}")
            print(f"Best true params: {results['best_true_params']}")
            
            # Show corruption statistics using regret_analysis utilities
            corruption_stats = compute_corruption_statistics(eval_results)
            if corruption_stats['n_corrupted'] > 0:
                print(f"Corruption: {corruption_stats['n_corrupted']} points corrupted, "
                      f"total magnitude: {corruption_stats['total_corruption']:.2f}")
            
            print("=" * 70)
        
        return results
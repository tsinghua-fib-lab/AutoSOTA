"""Bayesian Optimization loop implementation."""

import torch
from typing import Callable, Dict, Any, Optional, Tuple, List
from tqdm import tqdm

from .search_space import SearchSpace
from .evaluator import BaseEvaluator
from .evaluation_result import EvaluationResult
from .schedulers import BetaScheduler


class BOOptimizer:
    """Handles the Bayesian Optimization loop.
    
    This class implements the core BO loop: fit model → create acquisition → 
    optimize → evaluate → update data.
    """
    
    def __init__(self, search_space: SearchSpace):
        """Initialize with search space.
        
        Args:
            search_space: SearchSpace defining the optimization domain
        """
        self.search_space = search_space
    
    def _extract_X_tensor(self, results: List[EvaluationResult]) -> torch.Tensor:
        """Extract input tensor from EvaluationResult list."""
        X_points = []
        for result in results:
            if isinstance(result.x, dict):
                # Convert dict to tensor using search space
                x_point = self.search_space.encode_point(result.x)
            else:
                # Assume it's already a tensor
                x_point = result.x.clone() if hasattr(result.x, 'clone') else torch.tensor(result.x)
            X_points.append(x_point)
        
        return torch.stack(X_points) if X_points else torch.empty(0, self.search_space.n_dims)
    
    def _extract_Y_tensor(self, results: List[EvaluationResult], field: str) -> torch.Tensor:
        """Extract Y values tensor from EvaluationResult list."""
        Y_values = [getattr(result, field) for result in results]
        return torch.tensor(Y_values, dtype=torch.double) if Y_values else torch.empty(0)
    
    def optimize_loop(self, 
                     initial_results: List[EvaluationResult],
                     n_iterations: int,
                     model_factory: Callable,
                     acquisition_factory: Callable,
                     evaluator: BaseEvaluator,
                     model_kwargs: Optional[Dict[str, Any]] = None,
                     beta_scheduler: Optional[BetaScheduler] = None,
                     verbose: bool = False) -> Dict[str, Any]:
        """Run the BO optimization loop.
        
        Args:
            initial_results: List of EvaluationResult objects for initial points
            n_iterations: Number of BO iterations
            model_factory: Function to create model from (X, Y, **kwargs)
            acquisition_factory: Function to create acquisition from (model, search_space, **kwargs)
            evaluator: Function evaluator (handles noise/corruption internally)
            model_kwargs: Additional arguments for model_factory
            beta_scheduler: Optional scheduler for UCB beta parameter
            verbose: Whether to print progress
            
        Returns:
            Dictionary with complete optimization history:
                - X: All evaluated points
                - Y_observed: Values seen by BO (for model training)
                - Y_true: True function values
                - Y_noisy: Values with observation noise
                - corruption_levels: Corruption applied at each point
                - final_model: Last fitted model
                - final_acquisition: Last acquisition function
        """
        # Extract X and Y tensors from initial results
        X_current = self._extract_X_tensor(initial_results)
        Y_current = self._extract_Y_tensor(initial_results, 'y_observed')
        
        # Ensure proper tensor format
        X_current = X_current.double()
        Y_current = Y_current.double()
        
        if Y_current.dim() == 1:
            Y_current = Y_current.unsqueeze(-1)
        
        model_kwargs = model_kwargs or {}
        
        # Track complete history - start with initial results
        all_results = initial_results.copy()  # List of all EvaluationResult objects
        
        final_model = None
        final_acquisition = None
        
        # Create progress bar that always shows
        pbar = tqdm(range(n_iterations), desc="BO Iterations", unit="iter")
        
        for iteration in pbar:
            if verbose:
                print(f"  Iteration {iteration + 1}/{n_iterations}", end="")
            
            # Create model with observed data
            final_model = model_factory(X_current, Y_current, **model_kwargs)
            
            # Get beta from scheduler if provided
            acquisition_kwargs = {}
            if beta_scheduler is not None:
                beta = beta_scheduler.get_beta(iteration, n_iterations, final_model)
                acquisition_kwargs['beta'] = beta

            # Create acquisition function with optional beta
            final_acquisition = acquisition_factory(final_model, self.search_space, **acquisition_kwargs)
            
            # Optimize acquisition function using its optimize() method
            x_new = final_acquisition.optimize(num_restarts=10, raw_samples=512)
            
            # Evaluate new point
            params = self.search_space.decode_point(x_new)
            result = evaluator.evaluate(params)
            
            # Add to complete results list
            all_results.append(result)
            
            # Extract values from EvaluationResult
            y_true = result.y_true
            y_observed = result.y_observed
            noise = result.noise
            corruption = result.corruption
            
            # Update progress bar with current best value
            current_best = max(r.y_true for r in all_results)
            pbar.set_postfix({'best_true': f'{current_best:.2f}', 'current': f'{y_true:.2f}'})
            
            # Verbose output
            if verbose:
                if corruption != 0.0:
                    print(f" → f({list(params.values())}) = {y_true:.4f} "
                          f"(noisy: {result.y_noisy:.4f}, observed: {y_observed:.4f}, corruption: {corruption:+.4f})")
                elif noise != 0.0:
                    print(f" → f({list(params.values())}) = {y_true:.4f} (noisy: {result.y_noisy:.4f})")
                else:
                    print(f" → f({list(params.values())}) = {y_true:.4f}")
            
            # No need to maintain separate histories anymore - data is in all_results
            
            # Update data with observed value (what BO sees)
            X_current = torch.cat([X_current, x_new.unsqueeze(0)])
            Y_current = torch.cat([Y_current, torch.tensor([[y_observed]], dtype=torch.double)])
        
        return {
            # New clean interface
            'all_results': all_results,
            # Backward compatibility - extract tensors from all_results
            'X': self._extract_X_tensor(all_results),
            'Y_observed': self._extract_Y_tensor(all_results, 'y_observed'),
            'Y_true': self._extract_Y_tensor(all_results, 'y_true'),
            'Y_noisy': self._extract_Y_tensor(all_results, 'y_noisy'),
            'corruption_levels': self._extract_Y_tensor(all_results, 'corruption'),
            'final_model': final_model,
            'final_acquisition': final_acquisition
        }
"""
Parameter fitting utilities for GP models using Adam optimizer.

This module provides robust parameter fitting for both StandardGP and RobustConjugateGP
models, with special handling for RCGP's sigma parameter linkage.
"""

import torch
import numpy as np
from typing import Union, Dict, Optional, Any
from torch.optim import Adam
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from .models.standard_gp import StandardGP
from .models.robust_gp import RobustConjugateGP
from .weighting.plateau_imq import PlateauIMQ


class ParameterFitter:
    """
    Parameter fitter using Adam optimizer for stable hyperparameter optimization.
    
    Handles both StandardGP and RobustConjugateGP models with proper constraint handling
    and RCGP sigma parameter linking.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 500,
        convergence_threshold: float = 1e-6,
        patience: int = 50,
        verbose: bool = True,
    ):
        """
        Initialize parameter fitter.
        
        Args:
            learning_rate: Adam learning rate
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence tolerance for loss change
            patience: Early stopping patience
            verbose: Whether to print optimization progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.verbose = verbose
        
    def fit_parameters(
        self,
        model: Union[StandardGP, RobustConjugateGP],
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Fit model parameters using Adam optimizer.
        
        Args:
            model: GP model to fit
            fixed_params: Dictionary of parameters to keep fixed (e.g., {'plateau_width': 1.96})
            
        Returns:
            Dictionary of fitted parameter values
        """
        if fixed_params is None:
            fixed_params = {}
            
        # Set model to training mode
        model.train()
        
        # Create marginal log likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Collect parameters to optimize (excluding fixed ones)
        params_to_optimize = []
        param_names = []
        
        # Add likelihood parameters
        if 'noise' not in fixed_params:
            params_to_optimize.append(model.likelihood.raw_noise)
            param_names.append('likelihood.noise')
            
        # Add mean parameters
        if 'mean_constant' not in fixed_params:
            params_to_optimize.append(model.mean_module.raw_constant)
            param_names.append('mean_module.constant')
            
        # Add kernel parameters (handle both ScaleKernel and plain kernel)
        if 'lengthscale' not in fixed_params:
            if hasattr(model.covar_module, 'base_kernel'):
                # ScaleKernel wrapper (custom models)
                params_to_optimize.append(model.covar_module.base_kernel.raw_lengthscale)
            else:
                # Plain kernel (BoTorch SingleTaskGP)
                params_to_optimize.append(model.covar_module.raw_lengthscale)
            param_names.append('covar_module.lengthscale')
            
        if 'outputscale' not in fixed_params:
            if hasattr(model.covar_module, 'raw_outputscale'):
                # ScaleKernel wrapper (custom models)
                params_to_optimize.append(model.covar_module.raw_outputscale)
                param_names.append('covar_module.outputscale')
            # BoTorch SingleTaskGP doesn't have outputscale - skip it
            
        # Add RCGP-specific parameters
        is_rcgp = isinstance(model, RobustConjugateGP)
        if is_rcgp and isinstance(model.weighting_function, PlateauIMQ):
            # Add c parameter if not fixed
            if 'c' not in fixed_params:
                # Make c parameter learnable
                if not isinstance(model.weighting_function.c, torch.nn.Parameter):
                    model.weighting_function.c = torch.nn.Parameter(
                        torch.tensor(model.weighting_function.c, dtype=torch.double)
                    )
                params_to_optimize.append(model.weighting_function.c)
                param_names.append('weighting_function.c')
                
        # Apply fixed parameter constraints
        for param_name, value in fixed_params.items():
            self._set_fixed_parameter(model, param_name, value)
            
        if not params_to_optimize:
            if self.verbose:
                print("No parameters to optimize - all are fixed")
            return self._extract_parameter_values(model)
            
        # Create optimizer
        optimizer = Adam(params_to_optimize, lr=self.learning_rate)
        
        # Optimization loop
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        if self.verbose:
            print(f"Starting parameter optimization with {len(params_to_optimize)} parameters...")
            print(f"Parameters to optimize: {param_names}")
            
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Update RCGP sigma to match likelihood noise (before computing loss)
            if is_rcgp:
                self._update_rcgp_sigma(model)
                
            # Compute negative marginal log likelihood
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            
            # Check for numerical issues
            if not torch.isfinite(loss):
                if self.verbose:
                    print(f"Non-finite loss at iteration {iteration}: {loss.item()}")
                break
                
            # Backward pass
            try:
                loss.backward()
            except Exception as e:
                if self.verbose:
                    print(f"Error during backward pass at iteration {iteration}: {e}")
                break
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            
            # Check for invalid gradients
            valid_grads = True
            for param in params_to_optimize:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    valid_grads = False
                    break
                    
            if not valid_grads:
                if self.verbose:
                    print(f"Invalid gradients at iteration {iteration}")
                break
            
            optimizer.step()
            
            # Ensure parameters stay within valid bounds for BoTorch models
            self._ensure_parameter_bounds(model)
            
            # Track convergence
            current_loss = loss.item()
            losses.append(current_loss)
            
            # Check for improvement
            if current_loss < best_loss - self.convergence_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Print progress
            if self.verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1:3d}: Loss = {current_loss:.6f}")
                
            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration + 1}")
                break
                
        # Final update of RCGP sigma
        if is_rcgp:
            self._update_rcgp_sigma(model)
            
        # Extract final parameter values
        final_params = self._extract_parameter_values(model)
        
        if self.verbose:
            print(f"Optimization completed. Final loss: {best_loss:.6f}")
            print("Final parameter values:")
            for name, value in final_params.items():
                print(f"  {name}: {value:.6f}")
                
        return final_params
        
    def _ensure_parameter_bounds(self, model):
        """Ensure parameters stay within valid bounds for BoTorch models."""
        # For BoTorch models, we need to respect the constraints
        # This prevents the LogNormal prior violations
        
        # Ensure noise stays positive with minimum bound
        if hasattr(model.likelihood, 'noise_covar') and hasattr(model.likelihood.noise_covar, 'raw_noise_constraint'):
            constraint = model.likelihood.noise_covar.raw_noise_constraint
            if hasattr(constraint, 'lower_bound'):
                min_noise = constraint.lower_bound
                with torch.no_grad():
                    model.likelihood.raw_noise.clamp_(min=min_noise + 1e-6)
        
        # Ensure lengthscale stays positive with minimum bound
        covar_module = model.covar_module
        if hasattr(covar_module, 'base_kernel'):
            covar_module = covar_module.base_kernel
            
        if hasattr(covar_module, 'raw_lengthscale_constraint'):
            constraint = covar_module.raw_lengthscale_constraint
            if hasattr(constraint, 'lower_bound'):
                min_lengthscale = constraint.lower_bound
                with torch.no_grad():
                    covar_module.raw_lengthscale.clamp_(min=min_lengthscale + 1e-6)
        
    def _set_fixed_parameter(self, model: Union[StandardGP, RobustConjugateGP], param_name: str, value: float):
        """Set a parameter to a fixed value and disable gradients."""
        if param_name == 'noise':
            model.likelihood.noise = torch.tensor(value, dtype=torch.double)
            model.likelihood.raw_noise.requires_grad_(False)
        elif param_name == 'mean_constant':
            model.mean_module.constant = torch.tensor(value, dtype=torch.double)
            model.mean_module.raw_constant.requires_grad_(False)
        elif param_name == 'lengthscale':
            if hasattr(model.covar_module, 'base_kernel'):
                model.covar_module.base_kernel.lengthscale = torch.tensor(value, dtype=torch.double)
                model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
            else:
                model.covar_module.lengthscale = torch.tensor(value, dtype=torch.double)
                model.covar_module.raw_lengthscale.requires_grad_(False)
        elif param_name == 'outputscale':
            if hasattr(model.covar_module, 'raw_outputscale'):
                model.covar_module.outputscale = torch.tensor(value, dtype=torch.double)
                model.covar_module.raw_outputscale.requires_grad_(False)
        elif param_name == 'plateau_width':
            if isinstance(model, RobustConjugateGP):
                model.weighting_function.plateau_width = value
        elif param_name == 'c':
            if isinstance(model, RobustConjugateGP) and isinstance(model.weighting_function, PlateauIMQ):
                if isinstance(model.weighting_function.c, torch.nn.Parameter):
                    model.weighting_function.c.data.fill_(value)
                    model.weighting_function.c.requires_grad_(False)
                else:
                    model.weighting_function.c = value
                    
    def _update_rcgp_sigma(self, model: RobustConjugateGP):
        """Update RCGP sigma parameter to match likelihood noise standard deviation."""
        if isinstance(model, RobustConjugateGP):
            # Ensure sigma = sqrt(likelihood.noise)
            noise_std = torch.sqrt(model.likelihood.noise).item()
            model.weighting_function.sigma = noise_std
            model.weighting_function.beta = noise_std / np.sqrt(2)
            
            # No need to update c parameter - it's used directly
                
    def _extract_parameter_values(self, model: Union[StandardGP, RobustConjugateGP]) -> Dict[str, float]:
        """Extract current parameter values from model."""
        params = {}
        
        # Likelihood parameters
        params['noise'] = model.likelihood.noise.item()
        
        # Mean parameters
        params['mean_constant'] = model.mean_module.constant.item()
        
        # Kernel parameters (handle both ScaleKernel and plain kernel)
        if hasattr(model.covar_module, 'base_kernel'):
            params['lengthscale'] = model.covar_module.base_kernel.lengthscale.item()
        else:
            params['lengthscale'] = model.covar_module.lengthscale.item()
            
        if hasattr(model.covar_module, 'outputscale'):
            params['outputscale'] = model.covar_module.outputscale.item()
        
        # RCGP-specific parameters
        if isinstance(model, RobustConjugateGP):
            params['sigma'] = model.weighting_function.sigma
            params['beta'] = model.weighting_function.beta
            if isinstance(model.weighting_function, PlateauIMQ):
                params['plateau_width'] = model.weighting_function.plateau_width
                if isinstance(model.weighting_function.c, torch.nn.Parameter):
                    params['c'] = model.weighting_function.c.item()
                else:
                    params['c'] = model.weighting_function.c
                
        return params


def fit_gp_parameters(
    model: Union[StandardGP, RobustConjugateGP],
    fixed_params: Optional[Dict[str, Any]] = None,
    learning_rate: float = 0.01,
    max_iterations: int = 500,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Convenience function for fitting GP parameters.
    
    Args:
        model: GP model to fit
        fixed_params: Dictionary of parameters to keep fixed
        learning_rate: Adam learning rate
        max_iterations: Maximum optimization iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary of fitted parameter values
    """
    fitter = ParameterFitter(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    
    return fitter.fit_parameters(model, fixed_params)
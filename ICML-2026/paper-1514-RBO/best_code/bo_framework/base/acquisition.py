"""Base acquisition function abstractions with self-optimization capability."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional
from botorch.optim import optimize_acqf


class BaseAcquisitionFunction(nn.Module, ABC):
    """Base class for acquisition functions with self-optimization capability.
    
    This abstract class provides a unified interface for all acquisition functions,
    enabling them to optimize themselves over the search space. Inherits from 
    torch.nn.Module for compatibility with BoTorch optimization routines.
    """
    
    def __init__(self, model, search_space):
        """Initialize with model and search space context.
        
        Args:
            model: GP model with posterior() method
            search_space: SearchSpace for optimization bounds and structure
        """
        super().__init__()
        self.model = model
        self.search_space = search_space
    
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate acquisition function at points X.
        
        This is the standard forward method for nn.Module compatibility.
        
        Args:
            X: Points to evaluate. Shape depends on context:
               - [n_points, n_dims] for standard evaluation
               - [batch, q, n_dims] for BoTorch optimization (q=1 typically)
            
        Returns:
            Acquisition values with appropriate shape:
               - [n_points] for standard evaluation
               - [batch, q] for BoTorch optimization
        """
        pass
    
    @abstractmethod
    def optimize(self, num_restarts: int = 10, raw_samples: int = 512) -> torch.Tensor:
        """Optimize the acquisition function over the search space.
        
        Args:
            num_restarts: Number of random restarts for optimization
            raw_samples: Number of samples for initialization heuristic
            
        Returns:
            Optimal point in the search space [n_dims]
        """
        pass
    
    def _optimize_continuous(self, num_restarts: int = 10, raw_samples: int = 512) -> torch.Tensor:
        """Optimize over continuous space using optimize_acqf.
        
        Args:
            num_restarts: Number of random restarts
            raw_samples: Number of samples for initialization
            
        Returns:
            Optimal point [n_dims]
        """
        candidate, _ = optimize_acqf(
            acq_function=self,
            bounds=self.search_space.bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        
        return candidate.squeeze(0)  # Remove batch dimension
    
    def _optimize_mixed(self, num_restarts: int = 10, raw_samples: int = 512) -> torch.Tensor:
        """Optimize over mixed space using grid search over discrete variables.
        
        For each combination of discrete variables, optimizes continuous variables.
        
        Args:
            num_restarts: Number of random restarts for continuous optimization
            raw_samples: Number of samples for initialization
            
        Returns:
            Optimal point [n_dims]
        """
        from botorch.acquisition import FixedFeatureAcquisitionFunction
        
        best_candidate = None
        best_acq_value = float('-inf')
        
        # Get all discrete combinations (categorical and ordinal)
        discrete_combos = self.search_space.get_discrete_combinations(include_ordinals=True)
        
        for fixed_features in discrete_combos:
            if not fixed_features:
                # No discrete variables, just optimize continuous
                return self._optimize_continuous(num_restarts, raw_samples)
            
            # Create acquisition with fixed discrete features
            fixed_acq = FixedFeatureAcquisitionFunction(
                acq_function=self,
                d=self.search_space.n_dims,
                columns=list(fixed_features.keys()),
                values=list(fixed_features.values())
            )
            
            # Get bounds for continuous dimensions only
            continuous_dims = self.search_space.continuous_dims
            if continuous_dims:
                # Extract continuous bounds
                full_bounds = self.search_space.bounds
                cont_lower = [full_bounds[0, i].item() for i in continuous_dims]
                cont_upper = [full_bounds[1, i].item() for i in continuous_dims]
                cont_bounds = torch.tensor([cont_lower, cont_upper], dtype=torch.double)
                
                # Optimize continuous variables
                from botorch.optim import optimize_acqf
                candidate, acq_val = optimize_acqf(
                    acq_function=fixed_acq,
                    bounds=cont_bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
                candidate = candidate.squeeze(0)
                acq_val = acq_val.item()
            else:
                # No continuous variables, just evaluate this discrete combo
                candidate = torch.zeros(self.search_space.n_dims, dtype=torch.double)
                for dim_idx, val in fixed_features.items():
                    candidate[dim_idx] = val
                with torch.no_grad():
                    acq_val = self.forward(candidate.unsqueeze(0)).item()
            
            if acq_val > best_acq_value:
                best_acq_value = acq_val
                # Reconstruct full candidate with fixed features
                full_candidate = torch.zeros(self.search_space.n_dims, dtype=torch.double)
                
                # Fill in continuous values
                if continuous_dims:
                    for i, dim_idx in enumerate(continuous_dims):
                        full_candidate[dim_idx] = candidate[i]
                
                # Fill in discrete values
                for dim_idx, val in fixed_features.items():
                    full_candidate[dim_idx] = val
                
                best_candidate = full_candidate
        
        return best_candidate


class UCBAcquisition(BaseAcquisitionFunction):
    """Upper Confidence Bound acquisition function.
    
    UCB balances exploration and exploitation using:
    UCB(x) = μ(x) + β * σ(x)
    
    where μ is the posterior mean, σ is the posterior standard deviation,
    and β controls the exploration-exploitation trade-off.
    """
    
    def __init__(self, model, search_space, beta: float = 2.0):
        """Initialize UCB acquisition.
        
        Args:
            model: GP model with posterior() method
            search_space: SearchSpace for optimization
            beta: Exploration weight (higher = more exploration)
        """
        super().__init__(model, search_space)
        self.beta = beta
    
    @classmethod
    def create(cls, model, search_space, beta: float = 2.0):
        """Factory method to create UCB acquisition.
        
        Args:
            model: GP model with posterior() method
            search_space: SearchSpace for optimization
            beta: Exploration weight (higher = more exploration)
            
        Returns:
            UCBAcquisition instance
        """
        return cls(model, search_space, beta=beta)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute UCB acquisition values.
        
        Handles both standard evaluation and BoTorch optimization shapes.
        
        Args:
            X: Points to evaluate
               - [n_points, n_dims] for standard evaluation
               - [batch, q, n_dims] for BoTorch optimization
            
        Returns:
            UCB values with appropriate shape
        """
        # Handle BoTorch's batch x q x d format
        if X.dim() == 3:
            batch_shape = X.shape[:-2]
            X_2d = X.reshape(-1, X.shape[-1])  # Flatten to 2D for posterior
        else:
            batch_shape = None
            X_2d = X
        
        # Ensure correct dtype
        X_2d = X_2d.double()
        
        # Get posterior (no torch.no_grad() here - we need gradients for optimization!)
        posterior = self.model.posterior(X_2d)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
        
        # Compute UCB
        ucb = mean + self.beta * std
        
        # Reshape if needed for BoTorch
        if batch_shape is not None:
            ucb = ucb.reshape(*batch_shape)
        
        return ucb
    
    def optimize(self, num_restarts: int = 10, raw_samples: int = 512) -> torch.Tensor:
        """Optimize UCB over the search space.
        
        Args:
            num_restarts: Number of random restarts
            raw_samples: Number of samples for initialization
            
        Returns:
            Optimal point [n_dims]
        """
        # Check if we have mixed variables
        if self.search_space.categorical_dims or self.search_space.ordinal_dims:
            return self._optimize_mixed(num_restarts, raw_samples)
        else:
            return self._optimize_continuous(num_restarts, raw_samples)


class EIAcquisition(BaseAcquisitionFunction):
    """Expected Improvement acquisition function.
    
    EI computes the expected improvement over the current best value:
    EI(x) = E[max(f(x) - f*, 0)]
    
    where f* is the best observed value.
    """
    
    def __init__(self, model, search_space, best_f: Optional[float] = None):
        """Initialize EI acquisition.
        
        Args:
            model: GP model with posterior() method
            search_space: SearchSpace for optimization
            best_f: Best observed value (if None, computed from model's training data)
        """
        super().__init__(model, search_space)
        
        if best_f is None:
            # Try to infer from model's training data
            if hasattr(model, 'train_targets'):
                self.best_f = model.train_targets.max().item()
            else:
                raise ValueError("best_f must be provided or model must have train_targets")
        else:
            self.best_f = best_f
    
    @classmethod
    def create(cls, model, search_space, best_f: Optional[float] = None):
        """Factory method to create EI acquisition.
        
        Args:
            model: GP model with posterior() method
            search_space: SearchSpace for optimization
            best_f: Best observed value (if None, computed from model's training data)
            
        Returns:
            EIAcquisition instance
        """
        return cls(model, search_space, best_f=best_f)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute EI acquisition values.
        
        Handles both standard evaluation and BoTorch optimization shapes.
        
        Args:
            X: Points to evaluate
               - [n_points, n_dims] for standard evaluation
               - [batch, q, n_dims] for BoTorch optimization
            
        Returns:
            EI values with appropriate shape
        """
        from torch.distributions import Normal
        
        # Handle BoTorch's batch x q x d format
        if X.dim() == 3:
            batch_shape = X.shape[:-2]
            X_2d = X.reshape(-1, X.shape[-1])  # Flatten to 2D for posterior
        else:
            batch_shape = None
            X_2d = X
        
        # Ensure correct dtype
        X_2d = X_2d.double()
        
        # Get posterior (no torch.no_grad() here - we need gradients for optimization!)
        posterior = self.model.posterior(X_2d)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
        
        # Compute EI
        improvement = mean - self.best_f
        
        # Avoid numerical issues
        mask = std > 1e-6
        ei = torch.zeros_like(mean)
        
        if mask.any():
            # Standard normal distribution
            normal = Normal(torch.zeros(1, device=mean.device), torch.ones(1, device=mean.device))
            z = improvement[mask] / std[mask]
            
            # EI = (μ - f*) * Φ(z) + σ * φ(z)
            ei[mask] = improvement[mask] * normal.cdf(z) + std[mask] * torch.exp(normal.log_prob(z))
        
        # Reshape if needed for BoTorch
        if batch_shape is not None:
            ei = ei.reshape(*batch_shape)
        
        return ei
    
    def optimize(self, num_restarts: int = 10, raw_samples: int = 512) -> torch.Tensor:
        """Optimize EI over the search space.
        
        Args:
            num_restarts: Number of random restarts
            raw_samples: Number of samples for initialization
            
        Returns:
            Optimal point [n_dims]
        """
        # Check if we have mixed variables
        if self.search_space.categorical_dims or self.search_space.ordinal_dims:
            return self._optimize_mixed(num_restarts, raw_samples)
        else:
            return self._optimize_continuous(num_restarts, raw_samples)


class RobustUCBAcquisition(UCBAcquisition):
    """Robust Upper Confidence Bound acquisition function for RCGP models.
    
    This acquisition function implements the theoretical RCGP-UCB algorithm by
    inflating confidence bounds based on detected corruptions. It extends standard
    UCB with an additional robustness term:
    
    Robust-UCB(x) = μ(x) + β * σ(x) + C1 * t_c * σ(x)
    
    where:
    - μ(x), σ(x) are posterior mean and standard deviation
    - β is the standard exploration parameter
    - C1 is the robustness inflation parameter
    - t_c is the number of detected corrupted training points
    """
    
    def __init__(self, model, search_space, beta: float = 2.0, C1: float = 1.0):
        """Initialize Robust UCB acquisition.
        
        Args:
            model: RobustConjugateGP model with weighting function
            search_space: SearchSpace for optimization
            beta: Standard exploration weight
            C1: Robustness inflation parameter
            
        Raises:
            TypeError: If model is not RobustConjugateGP with PlateauIMQ weighting
        """
        super().__init__(model, search_space, beta)
        self.C1 = C1
        
        # Validate model type
        from rcgp.models.robust_gp import RobustConjugateGP
        from rcgp.weighting.plateau_imq import PlateauIMQ
        
        if not isinstance(model, RobustConjugateGP):
            raise TypeError("RobustUCBAcquisition requires a RobustConjugateGP model.")
        
        if not isinstance(model.weighting_function, PlateauIMQ):
            raise TypeError("RobustUCBAcquisition currently requires PlateauIMQ weighting.")
    
    @classmethod
    def create(cls, model, search_space, beta: float = 2.0, C1: float = 1.0):
        """Factory method to create Robust UCB acquisition.
        
        Args:
            model: RobustConjugateGP model with weighting function
            search_space: SearchSpace for optimization
            beta: Standard exploration weight
            C1: Robustness inflation parameter
            
        Returns:
            RobustUCBAcquisition instance
        """
        return cls(model, search_space, beta=beta, C1=C1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Robust UCB acquisition values.
        
        Extends standard UCB with robustness inflation based on detected corruptions.
        
        Args:
            X: Points to evaluate
               - [n_points, n_dims] for standard evaluation
               - [batch, q, n_dims] for BoTorch optimization
            
        Returns:
            Robust UCB values with appropriate shape
        """
        # Handle BoTorch's batch x q x d format
        if X.dim() == 3:
            batch_shape = X.shape[:-2]
            X_2d = X.reshape(-1, X.shape[-1])  # Flatten to 2D for posterior
        else:
            batch_shape = None
            X_2d = X
        
        # Ensure correct dtype
        X_2d = X_2d.double()
        
        # Get posterior (no torch.no_grad() here - we need gradients for optimization!)
        posterior = self.model.posterior(X_2d)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)
        
        # Standard UCB components
        ucb = mean + self.beta * std
        
        # Robust inflation term: C1 * t_c * σ(x)
        # Calculate t_c: number of points outside the PlateauIMQ plateau
        
        # Ensure model is in eval mode to use latest training data/parameters
        self.model.eval()
        
        try:
            # Get corruption indicators from PlateauIMQ weighting function
            if hasattr(self.model, 'train_inputs') and hasattr(self.model, 'train_targets'):
                train_X = self.model.train_inputs[0]
                train_Y = self.model.train_targets
                
                # Use PlateauIMQ's is_in_plateau method to identify corruptions
                is_corrupted = ~self.model.weighting_function.is_in_plateau(train_X, train_Y)
                t_c = is_corrupted.sum().item()
                
                # Add robustness inflation
                robust_inflation = self.C1 * t_c * std
                ucb = ucb + robust_inflation
                
            else:
                # Fallback if no training data available
                import warnings
                warnings.warn("No training data available for robustness calculation, using standard UCB")
                
        except Exception as e:
            import warnings
            warnings.warn(f"Error calculating robustness term: {e}, using standard UCB")
        
        # Reshape if needed for BoTorch
        if batch_shape is not None:
            ucb = ucb.reshape(*batch_shape)
        
        return ucb
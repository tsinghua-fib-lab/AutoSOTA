"""
Adaptive Double Robust Conjugate Gaussian Process (A2RCGP) implementation.

This module implements A2RCGP using a lagged centering approach (N-1). 
It assumes data is provided in chronological order.
"""

from typing import Optional, Dict, Any, Callable
import torch
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.types import _DefaultType, DEFAULT
from copy import deepcopy # Used for cloning configurations

from .robust_gp import RobustConjugateGP
from ..weighting import WeightingFunction


class InnerPosteriorMeanCenterFunction:
    """
    Centering function that uses inner RCGP posterior mean.

    Bridges between inner and outer standardized spaces:
    1. Gets inner posterior (auto-untransformed to original space)
    2. Transforms to outer's standardized space for weight computation
    """

    def __init__(self, inner_model: RobustConjugateGP, outer_outcome_transform=None):
        """
        Initialize with inner model reference.

        Args:
            inner_model: The fitted inner RCGP model (has its own outcome_transform)
            outer_outcome_transform: The outer model's outcome transform
        """
        # Store reference to the inner model
        self.inner_model = inner_model
        self.outer_outcome_transform = outer_outcome_transform

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get inner RCGP posterior mean transformed to outer's standardized space.

        This ensures the weighting function compares values in the same space:
        |y_outer_std - inner_pred_outer_std|
        """
        with torch.no_grad():
            # Ensure the inner model is in eval mode
            self.inner_model.eval()

            # Get posterior mean in ORIGINAL space (auto-untransformed by inner model)
            posterior = self.inner_model.posterior(x)
            mean_original = posterior.mean.squeeze(-1)

            # Transform to OUTER's standardized space
            if self.outer_outcome_transform is not None:
                self.outer_outcome_transform.eval()
                # Apply outer's transform: (y - mean_outer) / std_outer
                mean_outer_std, _ = self.outer_outcome_transform(mean_original.unsqueeze(-1))
                return mean_outer_std.squeeze(-1)

            return mean_original


def _create_inner_centering_function(inner_model: RobustConjugateGP, outer_outcome_transform=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create centering function that uses inner RCGP posterior mean.
    
    Args:
        inner_model: The fitted inner RCGP model (working in standardized space)
        outer_outcome_transform: The outer model's outcome transform for untransformation
        
    Returns:
        Pickleable centering function that returns inner model's posterior mean
    """
    return InnerPosteriorMeanCenterFunction(inner_model, outer_outcome_transform)


class A2RCGP(RobustConjugateGP):
    """
    Adaptive Double Robust Conjugate Gaussian Process (Lagged Implementation).

    Assumes chronological data (D_T).

    1. Inner RCGP (Lagged/Anchor): Trained on D_{T-1} with its own outcome_transform.
    2. Outer RCGP (Adaptive): Trained on D_T with its own outcome_transform.
       Uses the lagged inner posterior mean as center (bridged to outer's standardized space).

    Both models work in their own standardized spaces for numerical stability and
    parameter compatibility (c, plateau_width calibrated for std=1).
    """
    
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        inner_weighting_function: WeightingFunction,
        outer_weighting_function: WeightingFunction,
        inner_likelihood: Optional[GaussianLikelihood] = None,
        outer_likelihood: Optional[GaussianLikelihood] = None,
        inner_mean_module: Optional[gpytorch.means.Mean] = None,
        outer_mean_module: Optional[gpytorch.means.Mean] = None,
        inner_covar_module: Optional[gpytorch.kernels.Kernel] = None,
        outer_covar_module: Optional[gpytorch.kernels.Kernel] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ):
        """
        Initialize A2RCGP with independent standardization.
        Inner and outer models each work in their own standardized spaces.
        """
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        
        # Store Raw Y for use in condition_on_observations
        self._raw_train_Y = train_Y.clone()
        
        # Store original configurations for reconstruction
        self._original_outer_weighting = outer_weighting_function
        self._original_inner_weighting = inner_weighting_function
        
        # Handle outcome transform for outer model
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(m=train_Y.shape[-1])

        # Initialize outer RCGP (parent class) first - this fits the outcome transform on D_T
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            weighting_function=outer_weighting_function,
            likelihood=outer_likelihood,
            mean_module=outer_mean_module,
            covar_module=outer_covar_module,
            outcome_transform=outcome_transform
        )

        # Prepare lagged data for inner model (D_{T-1})
        N = train_X.shape[0]
        if N > 1:
            # Use all but the last point
            inner_train_X = train_X[:-1]
            inner_train_Y = train_Y[:-1]  # Raw Y for inner model
        else:
            # N=0 or N=1. Inner data is empty.
            inner_train_X = train_X[:0]
            inner_train_Y = train_Y[:0]

        # Create inner's OWN outcome transform (fitted on D_{T-1})
        # This ensures inner and outer have independent standardizations
        inner_outcome_transform = Standardize(m=train_Y.shape[-1]) if outcome_transform is not None else None

        # Create inner RCGP with its own outcome_transform
        # This prevents drift when D_T changes substantially from D_{T-1}
        self.inner_rcgp = RobustConjugateGP(
            train_X=inner_train_X,
            train_Y=inner_train_Y,
            weighting_function=inner_weighting_function,
            likelihood=inner_likelihood,
            mean_module=inner_mean_module,
            covar_module=inner_covar_module,
            outcome_transform=inner_outcome_transform  # Inner's own transform
        )
        
        # Don't set centering function yet - will be set after fitting
        
    def fit(self, inner_param_dict, outer_param_dict, objective_type="wloo-cv", **kwargs):
        """
        Unified fitting method for A2RCGP.
        
        Args:
            inner_param_dict: Parameter handling dictionary for inner RCGP
            outer_param_dict: Parameter handling dictionary for outer RCGP
            objective_type: Type of fitting objective ("mll", "loo-cv", "wloo-cv")
            **kwargs: Additional arguments (optimizer_type, verbose, etc.)
        """
        verbose = kwargs.get('verbose', False)
        
        # Handle fitting if inner model has no data (N=0 or N=1)
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            if verbose:
                print("Inner RCGP has no training data (N<=1). Skipping inner fitting.")
        else:
            # 1. Fit inner model (parameters should be pre-initialized)
            if verbose:
                print("Fitting inner RCGP model...")
            self.inner_rcgp.fit(inner_param_dict, objective_type=objective_type, **kwargs)
            if verbose:
                print("Inner RCGP fitted.")
            
            # Detach inner model from computation graph to prevent backpropagation
            for param in self.inner_rcgp.parameters():
                param.requires_grad_(False)
        
        # 2. Update centering function (inner model parameters changed)
        center_fn = _create_inner_centering_function(self.inner_rcgp, self.outcome_transform)
        self.weighting_function.set_center_function(center_fn)
        
        # 3. Fit outer model
        if verbose:
            print("Fitting outer RCGP model...")
        super().fit(outer_param_dict, objective_type=objective_type, **kwargs)
        if verbose:
            print("A2RCGP fitting completed.")
    
    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs):
        """
        Condition the model on new observations by reconstructing the model 
        with the full dataset, ensuring the lagged structure is maintained.

        Args:
            X: New input observations [n_new, d] (RAW)
            Y: New output observations [n_new, 1] or [n_new] (RAW)
            **kwargs: Additional arguments (ignored in this reconstruction approach)
            
        Returns:
            New A2RCGP model conditioned on observations
        """
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)

        # 1. Reconstruct the full raw dataset (maintaining chronological order).
        new_train_X = torch.cat([self.train_inputs[0], X], dim=0)
        # We must use the stored raw Y data.
        new_train_Y = torch.cat([self._raw_train_Y, Y], dim=0)

        # 2. Transfer Hyperparameters (Modules)
        # We extract the current state (HPs) from the fitted modules.

        # Check if outcome_transform exists and should be passed.
        has_transform = hasattr(self, "outcome_transform") and self.outcome_transform is not None

        # 3. Initialize a new A2RCGP from scratch using the full raw data and HPs.
        # The __init__ method correctly handles the lagging and standardization.
        
        new_a2rcgp = type(self)(
            train_X=new_train_X,
            train_Y=new_train_Y,
            # Pass the original configurations for weighting functions
            inner_weighting_function=self._original_inner_weighting,
            outer_weighting_function=self._original_outer_weighting,
            # Pass the fitted modules (Likelihoods, Means, Covariances)
            inner_likelihood=self.inner_rcgp.likelihood,
            outer_likelihood=self.likelihood,
            inner_mean_module=self.inner_rcgp.mean_module,
            outer_mean_module=self.mean_module,
            inner_covar_module=self.inner_rcgp.covar_module,
            outer_covar_module=self.covar_module,
            # Pass the existing outcome transform instance (maintains mean/std state)
            outcome_transform=self.outcome_transform if has_transform else None
        )
        
        return new_a2rcgp
    
    def get_inner_weights(self) -> Tensor:
        """Get current observation weights from inner RCGP (T-1 points)."""
        return self.inner_rcgp.get_weights()
    
    def get_outer_weights(self) -> Tensor:
        """Get current observation weights from outer RCGP (T points)."""
        return super().get_weights()
    
    def detect_corruptions(self, threshold_factor: float = 1.0) -> Dict[str, Tensor]:
        """
        Detect corrupted observations using both inner and outer models.
        
        Note: Inner model weights are only available for the first T-1 points.
        """
        inner_corruptions = self.inner_rcgp.detect_corruptions(threshold_factor)
        
        # Pad inner corruptions tensor if it's shorter than the outer one
        N_outer = self.train_inputs[0].shape[0]
        if inner_corruptions.shape[0] < N_outer:
            padding = torch.full(
                (N_outer - inner_corruptions.shape[0],), 
                float('nan'), # Indicate missing information for the last point(s)
                dtype=inner_corruptions.dtype,
                device=inner_corruptions.device
            )
            inner_corruptions = torch.cat([inner_corruptions, padding], dim=0)

        return {
            'inner': inner_corruptions,
            'outer': super().detect_corruptions(threshold_factor)
        }
    
    def get_n_outside_plateau(self) -> int:
        """
        Get the number of observations outside the plateau (corrupted points)
        detected by the outer model.
        """
        outer_corruptions = super().detect_corruptions()
        return int(outer_corruptions.sum().item())
    
    def __repr__(self) -> str:
        inner_w_name = type(self._original_inner_weighting).__name__
        outer_w_name = type(self._original_outer_weighting).__name__
        N = self.train_inputs[0].shape[0]
        return (
            f"A2RCGP(Lagged)(\n" 
            f"  inner_weighting={inner_w_name},\n"
            f"  outer_weighting={outer_w_name},\n"
            f"  n_data={N} (Inner N={max(0, N-1)})\n"
            f")"
        )
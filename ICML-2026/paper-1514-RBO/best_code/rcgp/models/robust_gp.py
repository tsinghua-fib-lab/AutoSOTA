"""
Robust Conjugate Gaussian Process implementation.

This module implements the RobustConjugateGP model that extends BoTorch's
SingleTaskGP with robust weighting functions while maintaining conjugacy.
"""

from typing import Optional
import torch
import numpy as np
import math
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import validate_input_scaling
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.types import _DefaultType, DEFAULT

from ..weighting import WeightingFunction


class RobustConjugateGP(ExactGP, GPyTorchModel):
    """
    Robust Conjugate Gaussian Process.

    This model extends standard GP with robust weighting functions to handle
    adversarial corruptions while maintaining conjugacy for efficient inference.
    """

    _num_outputs = 1  # Required by GPyTorchModel

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        weighting_function: WeightingFunction,
        likelihood: Optional[GaussianLikelihood] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ):
        # 1. Handle Input Validation and Outcome Transform
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(m=train_Y.shape[-1])

        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)

        # 2. Initialize Likelihood
        if likelihood is None:
            # Use noise prior only when standardizing (calibrated for std=1 data)
            # When not standardizing, remove prior to allow sigma to match data scale
            if outcome_transform is not None:
                # Standardized: use tuned prior (critical for performance)
                noise_prior = gpytorch.priors.LogNormalPrior(-4.0, 1.0)
                initial_noise = 0.0067379469983279705  # BoTorch's exact initial value
            else:
                # Not standardized: no prior, higher initial value for larger scale
                noise_prior = None
                initial_noise = 0.1  # More reasonable for non-standardized data

            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=gpytorch.constraints.GreaterThan(
                    1e-4,
                    initial_value=initial_noise,
                    transform=None
                )
            )

        # 3. Initialize the Parent ExactGP Class
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)
        
        # 4. Define Mean and Covariance Modules
        self.mean_module = ConstantMean() if mean_module is None else mean_module

        if covar_module is None:
            # Create the kernel and pass priors/constraints DIRECTLY to the constructor.
            # GPyTorch handles the parameter registration internally.
            ard_num_dims = train_X.shape[-1]
            lengthscale_prior = gpytorch.priors.LogNormalPrior(
                loc=math.sqrt(2) + math.log(ard_num_dims) * 0.5,
                scale=math.sqrt(3)
            )

            # When standardizing: use transform=None (match BoTorch exactly for MLL)
            # When not standardizing: use default transform (enforce constraints properly)
            if outcome_transform is not None:
                # Standardizing: match BoTorch with transform=None
                lengthscale_constraint = gpytorch.constraints.GreaterThan(
                    2.5e-2,
                    initial_value=lengthscale_prior.mode,
                    transform=None
                )
            else:
                # Not standardizing: higher minimum to prevent overfitting at larger scales
                lengthscale_constraint = gpytorch.constraints.GreaterThan(
                    0.2,  # Higher minimum for non-standardized data
                    initial_value=max(lengthscale_prior.mode, 0.3)
                )

            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint
            )

            # When not standardizing, wrap with ScaleKernel to learn output scale
            # When standardizing (std≈1), ScaleKernel is redundant
            if outcome_transform is None:
                self.covar_module = ScaleKernel(base_kernel)
            else:
                self.covar_module = base_kernel
        else:
            self.covar_module = covar_module
            
        # 5. Store Other Attributes
        self.weighting_function = weighting_function
        self.outcome_transform = outcome_transform

        # Ensure model is in float64, as per BoTorch convention
        self.to(train_X.device, dtype=torch.float64)

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Compute the prior latent distribution.
        
        This method always returns the PRIOR distribution, regardless of training mode.
        It's used during hyperparameter optimization (training) and as a building block
        for posterior computation (eval).

        Args:
            x: Input locations [*, n_test, d]

        Returns:
            Prior MultivariateNormal distribution
        """
        # Always compute prior distribution
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if hasattr(covar_x, "evaluate"):
            covar_x = covar_x.evaluate()
            
        # Add jitter for numerical stability
        jitter = 1e-6
        covar_x = covar_x + jitter * torch.eye(
            covar_x.shape[-1], device=covar_x.device, dtype=covar_x.dtype
        )
            
        return MultivariateNormal(mean_x, covar_x)
    
    def _compute_robust_posterior(self, x: Tensor) -> MultivariateNormal:
        """
        Compute robust posterior distribution (internal method).
        
        This contains the core robust GP logic that was previously in forward().
        
        Args:
            x: Test inputs [*, n_test, d]
            
        Returns:
            Robust posterior distribution
        """
        # Get prior distribution
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if hasattr(covar_x, "evaluate"):
            covar_x = covar_x.evaluate()

        # Check if we have training data for conditioning
        if self.train_inputs is None or len(self.train_inputs[0]) == 0:
            return MultivariateNormal(mean_x, covar_x)

        # Compute robust conditioning
        train_x = self.train_inputs[0]
        train_y = self.train_targets

        # Get cached or compute weights and corrections
        weights, J_matrix, gradient_correction = self._get_robust_components(
            train_x, train_y
        )

        # Compute cross-covariance
        covar_x_train = self.covar_module(x, train_x)
        if hasattr(covar_x_train, "evaluate"):
            covar_x_train = covar_x_train.evaluate()

        # Compute training covariance with robust correction
        covar_train_train = self.covar_module(train_x)

        # Add noise and weighting correction: K + σ²·J
        # Evaluate to dense tensor for matrix operations
        if hasattr(covar_train_train, "evaluate"):
            covar_train_train_dense = covar_train_train.evaluate()
        else:
            covar_train_train_dense = covar_train_train
        
        # Form K + σ²·J where J = diag(σ²/(2w²))
        covar_train_train_robust = covar_train_train_dense + self.likelihood.noise * J_matrix

        # Compute corrected training targets
        mean_train = self.mean_module(train_x)
        corrected_targets = (
            train_y - mean_train - self.likelihood.noise * gradient_correction
        )

        # Solve linear system for conditioning
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(covar_train_train_robust)
            # Solve L @ alpha = corrected_targets
            alpha = torch.cholesky_solve(corrected_targets.unsqueeze(-1), L).squeeze(-1)

            # Compute posterior mean
            posterior_mean = mean_x + covar_x_train @ alpha

            # Solve for posterior covariance
            # v = L^{-1} @ covar_x_train.T
            covar_x_train_t = covar_x_train.transpose(-2, -1)
            v = torch.linalg.solve_triangular(L, covar_x_train_t, upper=False)
            posterior_covar = covar_x - v.transpose(-2, -1) @ v

            # Add small jitter for numerical stability
            jitter = 1e-6
            posterior_covar = posterior_covar + jitter * torch.eye(
                posterior_covar.shape[-1], device=posterior_covar.device
            )

        except RuntimeError as e:
            # Fallback to regularized version if Cholesky fails
            print(f"Cholesky decomposition failed: {e}. Using regularized version.")
            jitter = 1e-6
            covar_train_train_robust = covar_train_train_robust + jitter * torch.eye(
                covar_train_train_robust.shape[-1],
                device=covar_train_train_robust.device,
            )

            # Use standard solve
            alpha = torch.linalg.solve(covar_train_train_robust, corrected_targets)
            posterior_mean = mean_x + covar_x_train @ alpha

            covar_x_train_t = covar_x_train.transpose(-2, -1)
            v = torch.linalg.solve(covar_train_train_robust, covar_x_train_t)
            posterior_covar = covar_x - covar_x_train @ v

            # Add jitter for numerical stability
            posterior_covar = posterior_covar + jitter * torch.eye(
                posterior_covar.shape[-1], device=posterior_covar.device
            )

        return MultivariateNormal(posterior_mean, posterior_covar)
        
    def __call__(self, *args, **kwargs) -> MultivariateNormal:
        """
        Forward pass that respects training/eval mode.
        
        Training mode: Returns prior distribution (for hyperparameter optimization)
        Eval mode: Returns robust posterior distribution (for predictions)
        
        This follows the same pattern as BoTorch's ExactGP.
        """
        # Convert args to proper format
        if len(args) == 1:
            x = args[0]
        else:
            raise ValueError("Expected single input tensor")
            
        # Ensure proper tensor format
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        if self.training:
            # Training mode: return prior for hyperparameter optimization
            # Input should match training inputs during optimization
            if self.train_inputs is not None:
                train_x = self.train_inputs[0]
                if x.shape == train_x.shape and torch.allclose(x, train_x, atol=1e-6):
                    # Input matches training data - return prior for MLL computation
                    return self.forward(x)
                else:
                    # Different inputs during training - this might be for gradient computation
                    # Return prior at requested points
                    return self.forward(x)
            else:
                # No training data yet - return prior
                return self.forward(x)
                
        else:
            # Eval mode: return robust posterior
            return self._compute_robust_posterior(x)

    def _get_robust_components(self, train_x: Tensor, train_y: Tensor):
        """
        Compute robust weighting components.
        
        No manual caching - relies on train/eval modes for gradient handling.
        
        Returns:
            Tuple of (weights, J_matrix, gradient_correction)
        """
        # Get current sigma from likelihood (as tensor for differentiability)
        sigma = torch.sqrt(self.likelihood.noise)
        
        # Compute weights and corrections
        weights = self.weighting_function.weight(train_x, train_y, sigma)
        gradient_correction = self.weighting_function.gradient_log_weight(
            train_x, train_y, sigma
        )
        J_matrix = self.weighting_function.compute_J_matrix(weights, sigma)
        
        return weights, J_matrix, gradient_correction

    def posterior(
        self, 
        X: Tensor,
        **kwargs
    ) -> GPyTorchPosterior:
        """
        Compute posterior for BoTorch compatibility.
        
        This method always returns the posterior distribution, regardless of the
        current training mode. It temporarily switches to eval mode if needed.
        
        IMPORTANT: The posterior is automatically untransformed to the original
        outcome scale if an outcome_transform was used.

        Args:
            X: Test points [*, n_test, d]
        
        Returns:
            GPyTorchPosterior object with robust posterior distribution
            in the ORIGINAL outcome scale (untransformed)
        """
        # Store current mode and switch to eval
        was_training = self.training
        self.eval()
        
        try:
            # Get robust posterior distribution (in transformed space if applicable)
            posterior_dist = self(X)
            
            # Create posterior object
            posterior = GPyTorchPosterior(posterior_dist)
            
            # Untransform if we have an outcome transform
            if self.outcome_transform is not None:
                posterior = self.outcome_transform.untransform_posterior(posterior)
            
            return posterior
            
        finally:
            # Restore original training mode
            if was_training:
                self.train()

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs):
        """
        Condition the model on new observations.

        Args:
            X: New input observations [n_new, d]
            Y: New target observations [n_new, 1] or [n_new]

        Returns:
            New RobustConjugateGP with updated training data
        """
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)

        # Important: Y should be in the ORIGINAL scale (not transformed)
        # We need to combine with existing data in original scale
        
        # Get existing data in original scale
        if self.train_inputs is not None and len(self.train_inputs[0]) > 0:
            new_train_X = torch.cat([self.train_inputs[0], X], dim=0)
            
            # Get existing Y in original scale
            if self.outcome_transform is not None:
                # Untransform the existing targets to original scale
                existing_Y = self.outcome_transform.untransform(
                    self.train_targets.unsqueeze(-1)
                )[0]
            else:
                existing_Y = self.train_targets.unsqueeze(-1)
            
            new_train_Y = torch.cat([existing_Y, Y], dim=0)
        else:
            new_train_X = X
            new_train_Y = Y

        # Use the same weighting function (no parameter modification)
        new_weighting_fn = self.weighting_function
        
        # Create new model with appropriate outcome transform
        # The transform will be re-fitted to the new combined data
        if self.outcome_transform is not None:
            # For Standardize transform, we need to pass the number of outputs
            if hasattr(self.outcome_transform, 'means') and hasattr(self.outcome_transform, 'stdvs'):
                # This is likely a Standardize transform
                num_outputs = self._num_outputs if hasattr(self, '_num_outputs') else 1
                new_outcome_transform = type(self.outcome_transform)(m=num_outputs)
            else:
                # Other transform types
                new_outcome_transform = type(self.outcome_transform)()
        else:
            new_outcome_transform = None
            
        new_model = RobustConjugateGP(
            train_X=new_train_X,
            train_Y=new_train_Y,
            weighting_function=new_weighting_fn,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            outcome_transform=new_outcome_transform,
        )

        return new_model

    def get_weights(self) -> Tensor:
        """Get current observation weights."""
        if self.train_inputs is None or len(self.train_inputs[0]) == 0:
            return torch.tensor([])

        train_x = self.train_inputs[0]
        train_y = self.train_targets
        weights, _, _ = self._get_robust_components(train_x, train_y)
        return weights

    def detect_corruptions(self, threshold_factor: float = 1.0) -> Tensor:
        """
        Detect corrupted observations based on weighting function.

        Args:
            threshold_factor: Multiplier for automatic threshold detection

        Returns:
            Boolean tensor indicating detected corruptions [n]
        """
        from ..weighting import PlateauIMQ  # Import specific type for type checking
        
        if not isinstance(self.weighting_function, PlateauIMQ):
            # For non-plateau weighting functions, use weight-based detection
            weights = self.get_weights()
            if len(weights) == 0:
                return torch.tensor([], dtype=torch.bool)

            # Consider low-weight observations as corrupted
            mean_weight = weights.mean()
            threshold = threshold_factor * mean_weight
            return weights < threshold
        else:
            # For plateau weighting functions, use plateau membership
            if self.train_inputs is None:
                return torch.tensor([], dtype=torch.bool)
            train_x = self.train_inputs[0]
            train_y = self.train_targets
            in_plateau = self.weighting_function.is_in_plateau(train_x, train_y)
            return ~in_plateau  # Corrupted = outside plateau
    
    def get_n_outside_plateau(self) -> int:
        """
        Get the number of observations outside the plateau (corrupted points).
        
        This method is used by RCGPScheduler for adaptive beta scheduling.
        
        Returns:
            Number of points detected as corrupted/outside plateau
        """
        corruptions = self.detect_corruptions()
        return int(corruptions.sum().item())

    def update_weighting_function(
        self,
        new_weighting_function: WeightingFunction,
        use_adaptive_centering: bool = False,
    ):
        """
        Update the weighting function and clear cache.

        Args:
            new_weighting_function: New weighting function to use
            use_adaptive_centering: If True, enables adaptive centering that uses
                                   previous posterior mean as center
        """
        self.weighting_function = new_weighting_function
        # No cache clearing needed - removed caching logic
        
        # Note: Adaptive centering removed for simplicity.
        # Can be re-implemented later if needed using the center_fn parameter.

    # Removed update_adaptive_centers and _clear_cache methods
    # No longer needed with simplified caching strategy

    @property
    def num_outputs(self) -> int:
        """Number of outputs."""
        return self._num_outputs if hasattr(self, '_num_outputs') else 1
        
    @property 
    def batch_shape(self) -> torch.Size:
        """The batch shape of the model."""
        if self.train_inputs is not None:
            return self.train_inputs[0].shape[:-2]
        return torch.Size()

    def fit(self, param_handling_dict, objective_type="wloo-cv", optimizer_type="lbfgs", **kwargs):
        """
        Unified fitting method for RCGP models.

        Args:
            param_handling_dict: Dictionary specifying how to handle each parameter
                Format: {"parameter_name": {"method": "manual/heuristics/fit", "value": value}}
            objective_type: Type of fitting objective ("mll", "loo-cv", "wloo-cv")
            optimizer_type: Type of optimizer ("adam", "lbfgs")
            **kwargs: Additional optimizer arguments. Recognized:
                lbfgs_backend: When optimizer_type="lbfgs", selects the L-BFGS
                    driver. "botorch" (default) uses botorch.fit.fit_gpytorch_mll;
                    "scipy" uses optimize_with_scipy_lbfgs (only honored for
                    loo-cv/wloo-cv; mll always uses BoTorch since scipy was
                    never wired for that objective).
        """
        # Import fitting utilities
        try:
            from ..fitting.rcgp_wloo import calculate_robust_heuristics, create_constant_center_fn
            from ..fitting.wloo_mll import RobustLeaveOneOutMLL, WeightedRobustLeaveOneOutMLL
            from ..fitting.scipy_optimizer import optimize_with_scipy_lbfgs
            from botorch.fit import fit_gpytorch_mll
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError:
            raise ImportError("Could not import necessary fitting utilities.")
        
        verbose = kwargs.get('verbose', False)
        
        # Note: Parameters should be initialized before calling fit()
        # This method only handles optimization
        
        # Set to training mode
        self.train()
        
        # Create MLL based on objective type
        if objective_type == 'mll':
            mll = ExactMarginalLogLikelihood(self.likelihood, self)
        elif objective_type == 'loo-cv':
            mll = RobustLeaveOneOutMLL(self.likelihood, self)
        elif objective_type == 'wloo-cv':
            mll = WeightedRobustLeaveOneOutMLL(self.likelihood, self)
        else:
            raise ValueError(f"Invalid objective_type: {objective_type}")
        
        mll.train()
        
        # Optimize based on optimizer type
        if optimizer_type == 'lbfgs':
            lbfgs_backend = kwargs.get('lbfgs_backend', 'botorch')
            if lbfgs_backend not in ('botorch', 'scipy'):
                raise ValueError(
                    f"Invalid lbfgs_backend: {lbfgs_backend!r}. "
                    "Must be 'botorch' or 'scipy'."
                )
            if objective_type in ['loo-cv', 'wloo-cv'] and lbfgs_backend == 'scipy':
                # scipy L-BFGS-B with parameter bounds
                max_iterations = kwargs.get('max_iterations', 1000)
                verbose_opt = kwargs.get('verbose', False)
                optimize_with_scipy_lbfgs(mll, self, max_iterations=max_iterations, verbose=verbose_opt)
            else:
                # BoTorch L-BFGS — default, and the only backend wired for MLL
                fit_gpytorch_mll(mll)
        elif optimizer_type == 'adam':
            # TODO: Implement Adam optimizer
            raise NotImplementedError("Adam optimizer not yet implemented")
        else:
            raise ValueError(f"Invalid optimizer_type: {optimizer_type}")
        
        # Set to eval mode
        self.eval()
        mll.eval()
        
        if verbose:
            print(f"RCGP fitting completed with {objective_type} objective")
    
    def _initialize_parameters(self, param_handling_dict, verbose=False):
        """
        Initialize model parameters based on param_handling_dict.
        """
        try:
            from ..fitting.rcgp_wloo import calculate_robust_heuristics
        except ImportError:
            raise ImportError("Could not import calculate_robust_heuristics.")
        
        # Handle sigma (noise) parameter
        if 'sigma' in param_handling_dict:
            sigma_method = param_handling_dict['sigma']['method']
            if sigma_method == 'manual':
                sigma = param_handling_dict['sigma']['value']
                self.likelihood.noise.data.fill_(sigma ** 2)
                self.likelihood.raw_noise.requires_grad_(False)
                if verbose:
                    print(f"Set noise to manual value: {sigma}")
            elif sigma_method == 'heuristics':
                # Calculate heuristics on standardized data
                # Note: self.train_targets is already standardized (transformed in __init__)
                Y_std = self.train_targets.unsqueeze(-1)
                heuristics = calculate_robust_heuristics(Y_std)
                sigma = heuristics['noise_estimate']
                self.likelihood.noise.data.fill_(sigma ** 2)
                self.likelihood.raw_noise.requires_grad_(False)
                if verbose:
                    print(f"Set noise to heuristics value: {sigma}")
            elif sigma_method == 'fit':
                # Let the parameter be optimized
                self.likelihood.raw_noise.requires_grad_(True)
                if verbose:
                    print("Noise parameter will be fitted")
            else:
                raise ValueError(f"Invalid sigma_method: {sigma_method}")
        
        # Handle mean parameter
        if 'mean' in param_handling_dict:
            mean_method = param_handling_dict['mean']['method']
            if mean_method == 'manual':
                mean = param_handling_dict['mean']['value']
                self.mean_module.constant.data.fill_(mean)
                self.mean_module.raw_constant.requires_grad_(False)
                if verbose:
                    print(f"Set mean to manual value: {mean}")
            elif mean_method == 'heuristics':
                # Calculate heuristics on standardized data
                # Note: self.train_targets is already standardized (transformed in __init__)
                Y_std = self.train_targets.unsqueeze(-1)
                heuristics = calculate_robust_heuristics(Y_std)
                mean = heuristics['center']
                self.mean_module.constant.data.fill_(mean)
                self.mean_module.raw_constant.requires_grad_(False)
                if verbose:
                    print(f"Set mean to heuristics value: {mean}")
            elif mean_method == 'fit':
                # Let the parameter be optimized
                self.mean_module.raw_constant.requires_grad_(True)
                if verbose:
                    print("Mean parameter will be fitted")
            else:
                raise ValueError(f"Invalid mean_method: {mean_method}")

    def eval(self):
        """Put model in evaluation mode for predictions."""
        return super().eval()

    def train(self, mode: bool = True):
        """Put model in training mode for hyperparameter optimization."""
        return super().train(mode)

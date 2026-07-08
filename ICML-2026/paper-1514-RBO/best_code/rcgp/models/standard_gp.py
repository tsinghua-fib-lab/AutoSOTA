"""
Custom Standard Gaussian Process implementation.

This module implements a StandardGP model that closely matches the interface
and structure of RobustConjugateGP but without the robust weighting components.
This is designed to ensure fair comparison with RCGP by eliminating 
implementation differences.

CRITICAL: This implementation is designed to match BoTorch's SingleTaskGP exactly.
The following components MUST be kept identical to BoTorch to ensure proper MLL
computation and optimization convergence:

1. **Noise Prior**: MUST use LogNormalPrior(-4.0, 1.0), NOT GammaPrior!
   - BoTorch uses LogNormalPrior(-4.0, 1.0) by default
   - Using GammaPrior will cause different MLL values even with identical parameters
   - The ExactMarginalLogLikelihood includes prior terms in the computation

2. **Lengthscale Prior**: MUST use dimension-scaled LogNormalPrior
   - loc = sqrt(2) + log(input_dim) * 0.5
   - scale = sqrt(3)
   - This matches BoTorch's get_covar_module_with_dim_scaled_prior()

3. **Kernel Structure**: MUST use plain RBFKernel, NOT ScaleKernel(RBFKernel)
   - BoTorch's SingleTaskGP uses RBFKernel directly
   - Using ScaleKernel would add an outputscale parameter that BoTorch doesn't have

4. **Initial Values**: MUST match BoTorch's exact initialization
   - Noise initial value: 0.0067379469983279705 (BoTorch's specific constant)
   - Lengthscale initial value: prior.mode (dimension-scaled)
   - Mean initial value: 0.0 (ConstantMean default)

5. **Parameter Constraints**: MUST use identical constraints
   - Noise constraint: GreaterThan(1e-4) with transform=None
   - Lengthscale constraint: GreaterThan(2.5e-2) with transform=None

If any of these components differ, the MLL computation will differ, causing
optimization to converge to different solutions even with identical data.

See tests/test_standardgp_mll_computation.py for verification that MLL values
match BoTorch SingleTaskGP within 1% relative error.
"""

from typing import Optional
import torch
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


class StandardGP(ExactGP, GPyTorchModel):
    """
    Standard Gaussian Process implementation that mirrors RobustConjugateGP structure.

    This model provides a baseline standard GP implementation with the same
    interface and structure as RobustConjugateGP, but without robust weighting.
    """

    _num_outputs = 1  # Required by GPyTorchModel

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: GaussianLikelihood | None = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ):
        """
        Initialize Standard GP.

        Args:
            train_X: Training inputs [n, d]
            train_Y: Training targets [n, 1] or [n]
            likelihood: Gaussian likelihood (created if None)
            mean_module: Mean function (defaults to ConstantMean)
            covar_module: Kernel function (defaults to RBF)
            outcome_transform: Transform applied to outcomes. If DEFAULT, uses
                Standardize to normalize outcomes (recommended). Pass None to
                disable outcome normalization.
        """
        # Validate and process inputs
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
            
        # Store original data dimensions before any transformations
        num_outputs = train_Y.shape[-1]
        
        # Handle outcome transform
        if outcome_transform == DEFAULT:
            # Use Standardize with numerical stability improvements
            outcome_transform = Standardize(m=num_outputs, batch_shape=train_X.shape[:-2])
        
        # Apply outcome transform if provided
        if outcome_transform is not None:
            # Fit and transform the training data
            train_Y_transformed = outcome_transform(train_Y)[0]
            train_Y_to_use = train_Y_transformed
        else:
            train_Y_to_use = train_Y
        
        # Initialize likelihood first
        if likelihood is None:
            from gpytorch.priors import LogNormalPrior
            from gpytorch.constraints import GreaterThan

            # CRITICAL: Use BoTorch's exact noise prior configuration (LogNormalPrior, not GammaPrior!)
            # Using GammaPrior will cause MLL computation differences!
            # Use noise prior only when standardizing (calibrated for std=1 data)
            # When not standardizing, remove prior to allow sigma to match data scale
            MIN_INFERRED_NOISE_LEVEL = 1e-4
            if outcome_transform is not None:
                # Standardized: use tuned prior (critical for performance)
                noise_prior = LogNormalPrior(-4.0, 1.0)  # BoTorch's exact default
                BOTORCH_NOISE_INITIAL_VALUE = 0.0067379469983279705  # BoTorch's specific constant
            else:
                # Not standardized: no prior, higher initial value for larger scale
                noise_prior = None
                BOTORCH_NOISE_INITIAL_VALUE = 0.1  # More reasonable for non-standardized data

            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=BOTORCH_NOISE_INITIAL_VALUE
                )
            )
            # Ensure likelihood uses double precision (BoTorch recommendation)
            likelihood = likelihood.to(dtype=torch.float64, device=train_X.device)

        # Initialize GP components FIRST (before setting any modules)
        super().__init__(train_X, train_Y_to_use.squeeze(-1), likelihood)
        GPyTorchModel.__init__(self)  # Initialize GPyTorchModel
        
        # NOW we can safely work with modules after __init__ is complete
        # Store the num_outputs now that we can assign to self
        self._num_outputs = num_outputs
        
        # Store the outcome transform
        self.outcome_transform = outcome_transform
        
        # Store the transformed targets (used during training)
        self.train_targets = train_Y_to_use.squeeze(-1)
            
        # Validate input scaling on the transformed data
        validate_input_scaling(train_X=train_X, train_Y=train_Y_to_use)

        # Store likelihood reference
        self.likelihood = likelihood

        # Store dimensions
        self._input_batch_shape = train_X.shape[:-2]
        self._aug_batch_shape = train_X.shape[:-2]

        # Set mean and covariance modules
        if mean_module is None:
            mean_module = ConstantMean()
        # Ensure mean_module uses double precision (BoTorch recommendation)
        self.mean_module = mean_module.to(dtype=torch.float64, device=train_X.device)

        if covar_module is None:
            # CRITICAL: Match BoTorch's SingleTaskGP covariance module exactly
            import math
            from gpytorch.priors import LogNormalPrior
            from gpytorch.constraints import GreaterThan
            
            # CRITICAL: Use BoTorch's dimension-scaled lengthscale prior
            # This matches get_covar_module_with_dim_scaled_prior() in BoTorch
            ard_num_dims = train_X.shape[-1]
            SQRT2 = math.sqrt(2)  # ≈ 1.414
            SQRT3 = math.sqrt(3)  # ≈ 1.732
            
            lengthscale_prior = LogNormalPrior(
                loc=SQRT2 + math.log(ard_num_dims) * 0.5,  # Dimension-scaled
                scale=SQRT3
            )
            
            # CRITICAL: BoTorch SingleTaskGP uses plain RBFKernel (for standardized data)
            # However, when not standardizing, we need ScaleKernel to learn output scale

            # When standardizing: use transform=None (match BoTorch exactly for MLL)
            # When not standardizing: use default transform (enforce constraints properly)
            if outcome_transform is not None:
                # Standardizing: match BoTorch with transform=None
                lengthscale_constraint = GreaterThan(
                    2.5e-2,
                    initial_value=lengthscale_prior.mode,
                    transform=None
                )
            else:
                # Not standardizing: higher minimum to prevent overfitting at larger scales
                lengthscale_constraint = GreaterThan(
                    0.2,  # Higher minimum for non-standardized data
                    initial_value=max(lengthscale_prior.mode, 0.3)
                )

            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint
            )

            # When not standardizing, wrap with ScaleKernel to learn output scale
            # When standardizing (std≈1), ScaleKernel is redundant (matches BoTorch)
            if outcome_transform is None:
                covar_module = ScaleKernel(base_kernel)
            else:
                covar_module = base_kernel
        # Ensure covar_module uses double precision (BoTorch recommendation)
        self.covar_module = covar_module.to(dtype=torch.float64, device=train_X.device)

        # Store training data (potentially transformed)
        self.train_inputs = (train_X,)
        # Note: train_targets may have already been set by outcome transform above

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
    
    def _compute_standard_posterior(self, x: Tensor) -> MultivariateNormal:
        """
        Compute standard posterior distribution (internal method).
        
        This contains the standard GP conditioning logic.
        
        Args:
            x: Test inputs [*, n_test, d]
            
        Returns:
            Standard posterior distribution
        """
        # Get prior distribution
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if hasattr(covar_x, "evaluate"):
            covar_x = covar_x.evaluate()

        # Check if we have training data for conditioning
        if self.train_inputs is None or len(self.train_inputs[0]) == 0:
            return MultivariateNormal(mean_x, covar_x)

        # Standard GP conditioning
        train_x = self.train_inputs[0]
        train_y = self.train_targets

        # Compute cross-covariance
        covar_x_train = self.covar_module(x, train_x)
        if hasattr(covar_x_train, "evaluate"):
            covar_x_train = covar_x_train.evaluate()

        # Compute training covariance with standard noise
        covar_train_train = self.covar_module(train_x)

        # Add noise: K + σ²·I (standard GP)
        # Evaluate to dense tensor for matrix operations
        if hasattr(covar_train_train, "evaluate"):
            covar_train_train_dense = covar_train_train.evaluate()
        else:
            covar_train_train_dense = covar_train_train
        
        # Form K + σ²·I (standard GP)
        covar_train_train_noisy = covar_train_train_dense + self.likelihood.noise * torch.eye(
            len(train_x), dtype=covar_train_train_dense.dtype, device=covar_train_train_dense.device
        )

        # Standard training targets (no robust corrections)
        mean_train = self.mean_module(train_x)
        if hasattr(mean_train, 'squeeze'):
            mean_train = mean_train.squeeze(-1) if mean_train.dim() > 1 else mean_train
        standard_targets = train_y - mean_train

        # Solve linear system for conditioning
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(covar_train_train_noisy)
            # Solve L @ alpha = standard_targets
            alpha = torch.cholesky_solve(standard_targets.unsqueeze(-1), L).squeeze(-1)

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
            covar_train_train_noisy = covar_train_train_noisy + jitter * torch.eye(
                covar_train_train_noisy.shape[-1],
                device=covar_train_train_noisy.device,
            )

            # Use standard solve
            alpha = torch.linalg.solve(covar_train_train_noisy, standard_targets)
            posterior_mean = mean_x + covar_x_train @ alpha

            covar_x_train_t = covar_x_train.transpose(-2, -1)
            v = torch.linalg.solve(covar_train_train_noisy, covar_x_train_t)
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
        Eval mode: Returns standard posterior distribution (for predictions)
        
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
            # Eval mode: return standard posterior
            return self._compute_standard_posterior(x)

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
            GPyTorchPosterior object with standard posterior distribution
            in the ORIGINAL outcome scale (untransformed)
        """
        # Store current mode and switch to eval
        was_training = self.training
        self.eval()
        
        try:
            # Get standard posterior distribution (in transformed space if applicable)
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
            New StandardGP with updated training data
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

        # Create new model with the same outcome transform
        # The transform will be re-fitted to the new combined data
        new_model = StandardGP(
            train_X=new_train_X,
            train_Y=new_train_Y,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            outcome_transform=self.outcome_transform,  # Pass the same transform type
        )

        return new_model

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

    def eval(self):
        """Put model in evaluation mode for predictions."""
        return super().eval()

    def train(self, mode: bool = True):
        """Put model in training mode for hyperparameter optimization."""
        return super().train(mode)
    
    def fit_hyperparameters(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 500,
        convergence_threshold: float = 1e-6,
        patience: int = 50,
        verbose: bool = False,
        fixed_params: Optional[dict] = None
    ) -> dict:
        """
        Fit hyperparameters using standard marginal log-likelihood with Adam optimizer.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for parameter changes  
            patience: Early stopping patience (iterations without improvement)
            verbose: Whether to print optimization progress
            fixed_params: Dictionary of parameters to fix (not implemented)
        
        Returns:
            Dictionary of fitted parameter values
        """
        from gpytorch.mlls import ExactMarginalLogLikelihood
        import torch.optim as optim
        
        # Set up marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        
        # Set model to training mode
        self.train()
        
        # Get trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if not trainable_params:
            if verbose:
                print("No trainable parameters found. Returning current values.")
            return self._extract_parameter_values()
        
        # Set up optimizer
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        prev_params = None
        
        if verbose:
            print(f"Starting StandardGP hyperparameter optimization for {max_iterations} iterations")
        
        for i in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute loss (negative MLL)
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            
            loss.backward()
            optimizer.step()
            
            # Apply parameter bounds
            self._ensure_parameter_bounds()
            
            current_loss = loss.item()
            
            # Check for improvement
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check convergence
            current_params = self._extract_parameter_values()
            if prev_params is not None:
                param_change = max(
                    abs(current_params[k] - prev_params[k]) 
                    for k in current_params.keys() 
                    if k in prev_params
                )
                if param_change < convergence_threshold:
                    if verbose:
                        print(f"Converged at iteration {i+1} (param change: {param_change:.2e})")
                    break
            
            prev_params = current_params.copy()
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {i+1}")
                break
            
            # Progress logging
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: MLL Loss = {current_loss:.6f}")
        
        # Return to eval mode
        self.eval()
        
        final_params = self._extract_parameter_values()
        if verbose:
            print("StandardGP optimization complete:")
            print(f"  Final MLL Loss: {best_loss:.6f}")
            print(f"  Noise: {final_params.get('noise', 'N/A'):.6f}")
            if 'lengthscale' in final_params:
                print(f"  Lengthscale: {final_params['lengthscale']:.6f}")
        
        return final_params
    
    def _ensure_parameter_bounds(self):
        """Ensure parameters stay within valid bounds."""
        # Lengthscale bounds
        if hasattr(self.covar_module, 'lengthscale'):
            with torch.no_grad():
                self.covar_module.lengthscale.clamp_(min=1e-4, max=100.0)
        
        # Outputscale bounds (if it exists)
        if hasattr(self.covar_module, 'outputscale'):
            with torch.no_grad():
                self.covar_module.outputscale.clamp_(min=1e-6, max=100.0)
        
        # Noise bounds
        if hasattr(self.likelihood, 'noise'):
            with torch.no_grad():
                self.likelihood.noise.clamp_(min=1e-6, max=10.0)
        
        # Mean constant bounds (if using ConstantMean)
        if hasattr(self.mean_module, 'constant'):
            with torch.no_grad():
                self.mean_module.constant.clamp_(min=-100.0, max=100.0)
    
    def _extract_parameter_values(self) -> dict:
        """Extract current parameter values from model."""
        params = {}
        
        # Kernel parameters
        if hasattr(self.covar_module, 'lengthscale'):
            lengthscale = self.covar_module.lengthscale
            if lengthscale.numel() == 1:
                params['lengthscale'] = lengthscale.item()
            else:
                # ARD case - store as list or take mean
                params['lengthscale'] = lengthscale.mean().item()
        
        if hasattr(self.covar_module, 'outputscale'):
            params['outputscale'] = self.covar_module.outputscale.item()
        
        # Likelihood parameters
        if hasattr(self.likelihood, 'noise'):
            params['noise'] = self.likelihood.noise.item()
        
        # Mean parameters
        if hasattr(self.mean_module, 'constant'):
            params['mean_constant'] = self.mean_module.constant.item()
        
        return params
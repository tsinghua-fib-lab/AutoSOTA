"""Student-t Process Model implementation using Pyro's MultivariateStudentT."""

import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import Standardize
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from pyro.distributions import MultivariateStudentT
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel


class StudentTProcessModel(ExactGP, GPyTorchModel):
    """
    Student-t Process model for robust Gaussian process regression.
    
    Implements the Student-t Process (Shah et al., 2014) which provides
    robustness to outliers through heavy-tailed distributions.
    
    Args:
        train_X: Training inputs [n, d]
        train_Y: Training targets [n, 1]
        nu: Degrees of freedom (default: 3.0). Must be > 2 for finite variance.
            Lower values = heavier tails = more robust to outliers
        outcome_transform: Optional outcome transformation (e.g., Standardize)
    """
    
    def __init__(self, train_X, train_Y, nu=3.0, outcome_transform=None):
        # Ensure proper tensor format
        train_X = train_X.double()
        train_Y = train_Y.double()
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        
        # Apply outcome transform if provided
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        
        # Initialize likelihood with proper priors and constraints
        from gpytorch.priors import LogNormalPrior
        from gpytorch.constraints import GreaterThan

        # Use noise prior only when standardizing (calibrated for std=1 data)
        # When not standardizing, remove prior to allow sigma to match data scale
        MIN_INFERRED_NOISE_LEVEL = 1e-4
        if outcome_transform is not None:
            # Standardized: use tuned prior (critical for performance)
            noise_prior = LogNormalPrior(-4.0, 1.0)
            BOTORCH_NOISE_INITIAL_VALUE = 0.0067379469983279705  # BoTorch's specific constant
        else:
            # Not standardized: no prior, higher initial value for larger scale
            noise_prior = None
            BOTORCH_NOISE_INITIAL_VALUE = 0.1  # More reasonable for non-standardized data

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=BOTORCH_NOISE_INITIAL_VALUE
            )
        )
        
        # Squeeze Y to 1D for ExactGP (it expects 1D targets)
        train_Y = train_Y.squeeze(-1)
        
        super().__init__(train_X, train_Y, likelihood)
        
        # Store the outcome transform
        self.outcome_transform = outcome_transform
        
        # Initialize mean and covariance modules with proper priors
        self.mean_module = ConstantMean()
        
        # Set up covariance module with proper priors and constraints
        import math
        from gpytorch.priors import LogNormalPrior
        from gpytorch.constraints import GreaterThan
        
        # Use dimension-scaled lengthscale prior like BoTorch
        ard_num_dims = train_X.shape[-1]
        SQRT2 = math.sqrt(2)  # ≈ 1.414
        SQRT3 = math.sqrt(3)  # ≈ 1.732
        
        lengthscale_prior = LogNormalPrior(
            loc=SQRT2 + math.log(ard_num_dims) * 0.5,  # Dimension-scaled
            scale=SQRT3
        )

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
        # When standardizing (std≈1), ScaleKernel is redundant
        if outcome_transform is None:
            self.covar_module = ScaleKernel(base_kernel)
        else:
            self.covar_module = base_kernel
        
        # Validate and register nu (degrees of freedom)
        if nu <= 2:
            raise ValueError("nu must be > 2 for finite variance")
        
        self.register_buffer("nu", torch.tensor(float(nu)))
        
        # Ensure model uses double precision like other models
        self.to(dtype=torch.float64, device=train_X.device)
        
        # Required for BoTorch compatibility
        self._num_outputs = 1
    
    @property
    def num_outputs(self):
        """Number of outputs for BoTorch compatibility."""
        return self._num_outputs
    
    def forward(self, x):
        """Forward pass returns the prior distribution."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def _calculate_beta(self):
        """
        Calculate the Mahalanobis distance beta for the training data.
        beta = (y - mu)^T (K + \sigma^2 I)^{-1} (y - mu)
        """
        if self.train_inputs is None or len(self.train_inputs[0]) == 0:
            return torch.tensor(0.0, device=self.nu.device)
        
        # Get prior distribution on the latent function f
        prior_dist_f = self.forward(self.train_inputs[0])

        # Get the distribution of y = f + noise, which is the one we need
        # for the Mahalanobis distance of the observations.
        prior_dist_y = self.likelihood(prior_dist_f)
        
        # Calculate residuals against the noisy prior mean
        diff = self.train_targets.squeeze() - prior_dist_y.mean
        
        # Compute inverse quadratic form using the noisy covariance matrix
        beta = prior_dist_y.lazy_covariance_matrix.inv_quad(diff.unsqueeze(-1)).squeeze()
        
        return torch.clamp(beta, min=0.0)  # Ensure non-negative
    
    def posterior(self, X, observation_noise=False, **kwargs):
        """
        Compute the Student-t posterior distribution.
        
        The posterior is a multivariate Student-t distribution with:
        - Updated degrees of freedom: nu' = nu + n
        - Scaled covariance: Sigma' = S * Sigma_GP
        - Where S = (nu + beta) / (nu + n)
        
        IMPORTANT: The posterior is automatically untransformed to the original
        outcome scale if an outcome_transform was used.
        
        Args:
            X: Test points [n_test, d]
            observation_noise: Whether to include observation noise
            **kwargs: Additional arguments
            
        Returns:
            GPyTorchPosterior containing MultivariateStudentT distribution
            in the ORIGINAL outcome scale (untransformed)
        """
        # Store current mode and switch to eval
        was_training = self.training
        self.eval()
        
        try:
            # Get the standard GP posterior (a MultivariateNormal) in the transformed space.
            # self(X) in eval() mode returns the posterior of the latent function f.
            # This is on the transformed scale if an outcome_transform is used.
            gp_posterior_f = self(X, **kwargs)
            
            # If observation_noise is requested, get the posterior of y = f + noise.
            # Otherwise, use the posterior of f.
            if observation_noise:
                mvn = self.likelihood(gp_posterior_f, **kwargs)
            else:
                mvn = gp_posterior_f
            
            # Calculate scaling factor S = (nu + beta) / (nu + n)
            n = self.train_targets.size(0)
            beta = self._calculate_beta()
            scaling_factor = (self.nu + beta) / (self.nu + n)
            
            # Only ensure it's positive (Student-t process theory guarantees this)
            scaling_factor = torch.clamp(scaling_factor, min=1e-6)
            
            # Update degrees of freedom
            nu_prime = self.nu + n
            
            # Scale the covariance and get Cholesky decomposition
            scaled_covar = mvn.lazy_covariance_matrix * scaling_factor
            
            # Add small jitter for numerical stability
            from linear_operator.operators import DiagLinearOperator
            jitter = DiagLinearOperator(torch.full(scaled_covar.shape[-1:], 1e-6, dtype=scaled_covar.dtype, device=scaled_covar.device))
            scaled_covar = scaled_covar + jitter
            
            # Get the lower triangular Cholesky factor
            L = scaled_covar.cholesky()
            
            # Convert to dense tensor for Pyro
            if hasattr(L, 'to_dense'):
                scale_tril = L.to_dense()
            else:
                scale_tril = L.evaluate() if hasattr(L, 'evaluate') else L
            
            # Create MultivariateStudentT distribution using Pyro
            mvt = MultivariateStudentT(
                df=nu_prime,
                loc=mvn.mean,
                scale_tril=scale_tril
            )
            
            # CRITICAL: Untransform if we have an outcome transform
            # This converts from standardized space back to original space
            if self.outcome_transform is not None:
                # For Student-t distributions, we must manually handle the untransformation
                # because BoTorch's untransform_posterior expects a GPyTorchPosterior
                # wrapping a distribution with specific attributes (like .islazy) that
                # Pyro's MultivariateStudentT does not have.
                
                # Get transformation parameters from the Standardize transform
                mean_transform = self.outcome_transform.means.squeeze()
                std_transform = self.outcome_transform.stdvs.squeeze()
                
                # Transform the Student-t parameters back to the original data scale
                loc_untransformed = mvt.loc * std_transform + mean_transform
                scale_untransformed = mvt.scale_tril * std_transform
                
                # Create a new MultivariateStudentT distribution in the original space
                final_mvt = MultivariateStudentT(df=mvt.df, loc=loc_untransformed, scale_tril=scale_untransformed)
            else:
                final_mvt = mvt
            
            return GPyTorchPosterior(distribution=final_mvt)
            
        finally:
            # Restore original training mode
            if was_training:
                self.train()

class StudentTMarginalLogLikelihood(MarginalLogLikelihood):
    """
    Marginal log likelihood for the Student-t Process.
    
    This computes the log marginal likelihood of the data under a 
    multivariate Student-t prior distribution.
    """
    
    def __init__(self, likelihood, model):
        """
        Args:
            likelihood: The likelihood module (typically GaussianLikelihood)
            model: The StudentTProcessModel
        """
        super().__init__(likelihood, model)
    
    def forward(self, function_dist, target, **kwargs):
        """
        Compute the marginal log likelihood.
        
        Args:
            function_dist: The GP prior distribution (MultivariateNormal)
            target: Training targets
            
        Returns:
            Scalar log marginal likelihood (normalized by data size)
        """
        # Get the output from the likelihood (adds noise)
        output = self.likelihood(function_dist)
        
        # Extract mean and covariance
        mean = output.mean
        covar = output.lazy_covariance_matrix
        
        # Get the Cholesky decomposition
        L = covar.cholesky()
        
        # Convert to dense tensor for Pyro
        if hasattr(L, 'to_dense'):
            L_dense = L.to_dense()
        else:
            L_dense = L.evaluate() if hasattr(L, 'evaluate') else L
        
        # Ensure target has correct shape
        if target.dim() > 1:
            target = target.squeeze(-1)
        
        # Create multivariate Student-t distribution
        prior_mvt = MultivariateStudentT(
            df=self.model.nu,
            loc=mean,
            scale_tril=L_dense
        )
        
        # Get log probability
        res = prior_mvt.log_prob(target)
        
        # Normalize by the number of data points (following GPyTorch convention)
        num_data = function_dist.event_shape.numel()
        return res / num_data
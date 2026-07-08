"""
Student-t GP implementation using GPyTorch with Variational Inference (VI).
Includes Robust Standardization (Median/MAD), Bayesian Priors, and improved optimization for robust outlier detection.
"""

import torch
import gpytorch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import StudentTLikelihood
# Import GammaPrior for defining priors
from gpytorch.priors import GammaPrior
from typing import Optional
import math

# =====================================================================
# 1. Core Variational GP Model Definition (with Priors)
# =====================================================================

class StudentTApproximateGP(ApproximateGP):
    """Core VI model definition with Priors."""
    def __init__(self, inducing_points, kernel_priors: dict = {}):
        inducing_points = inducing_points.to(dtype=torch.float64)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # --- FIX: Define Kernel with Priors ---
        # Priors suitable for standardized data (X normalized, Y robustly standardized)
        
        # Lengthscale Prior: Gamma(3, 6) gives a mean around 0.5. Encourages smoothness.
        lengthscale_prior = kernel_priors.get("lengthscale", GammaPrior(3.0, 6.0))

        ard_num_dims = inducing_points.size(-1)
        lengthscale_prior = gpytorch.priors.LogNormalPrior(
                loc=math.sqrt(2) + math.log(ard_num_dims) * 0.5,
                scale=math.sqrt(3)
            )
        self.covar_module = gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(
                2.5e-2,
                initial_value=lengthscale_prior.mode, transform=None
            )
        )
        # --------------------------------------
        
        self.to(dtype=torch.float64)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# =====================================================================
# 2. The BoTorch-Compatible Wrapper (StudentTGP)
# =====================================================================

class StudentTGP(Model):
    """
    Student-t GP using GPyTorch VI for robust diagnostics.
    """
    _num_outputs = 1

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        degrees_of_freedom: float = 3.0, # Default changed to 3.0 for heavier tails
        n_inducing: Optional[int] = None,
        training_iterations: int = 300, # Increased default for better VI convergence
        fix_nu: bool = True,
        priors: dict = {} # Allow custom priors
    ):
        super().__init__()
        self.train_X = train_X.to(dtype=torch.float64)
        self.train_Y = train_Y.to(dtype=torch.float64)
        self.training_iterations = training_iterations
        self.fix_nu = fix_nu

        # --- ROBUST STANDARDIZATION (Median/MAD) ---
        self._Y_median = self.train_Y.median()
        mad = (self.train_Y - self._Y_median).abs().median()
        # Scale factor (approx 1.4826) makes MAD consistent with Std Dev for Gaussian data
        self._Y_scale = 1.4826 * mad 

        # Fallback mechanism if MAD is zero (e.g., >50% data identical)
        if self._Y_scale < 1e-6:
            self._Y_scale = self.train_Y.std() # Fallback to non-robust std
            if self._Y_scale < 1e-6:
               self._Y_scale = torch.tensor(1.0, device=self.train_Y.device, dtype=self.train_Y.dtype)
        
        Y_standardized = (self.train_Y - self._Y_median) / self._Y_scale
        # ----------------------------------------------------

        # --- FIX: Initialize Likelihood with Noise Prior ---
        # The StudentTLikelihood has an internal 'noise' parameter that scales the distribution.
        # We must constrain this to prevent adaptation to outliers.
        
        # Noise Prior: Strong belief that noise in standardized space is small.
        # Gamma(1.1, 0.05) favors small values.
        noise_prior = priors.get("noise", GammaPrior(1.1, 0.05)) 

        self.likelihood = StudentTLikelihood(noise_prior=noise_prior)
        # ---------------------------------------------------

        # Initialize nu (must be > 2 for variance to be defined)
        if degrees_of_freedom <= 2.0:
            print(f"Warning: degrees_of_freedom must be > 2. Setting to 2.01.")
            degrees_of_freedom = 2.01
            
        self.likelihood.nu = torch.tensor(degrees_of_freedom, device=self.train_X.device, dtype=torch.float64)
        self.likelihood = self.likelihood.to(dtype=torch.float64)

        # Initialize Model (Inducing points selection)
        if n_inducing is None:
            n_inducing = min(len(self.train_X), max(min(20, len(self.train_X)), len(self.train_X) // 2))
        
        if len(self.train_X) > n_inducing:
            indices = torch.randperm(len(self.train_X))[:n_inducing]
            inducing_points = self.train_X[indices]
        else:
            inducing_points = self.train_X.clone()

        # Initialize core model with kernel priors
        self.model = StudentTApproximateGP(inducing_points, kernel_priors=priors.get("kernel", {}))
        self.model = self.model.to(dtype=torch.float64)

        self.Y_standardized = Y_standardized.squeeze(-1).to(dtype=torch.float64)

    def _fit_model(self, verbose: bool = False):
        """The VI training loop with improved optimization."""
        self.model.train()
        self.likelihood.train()

        # Freeze NU parameter (if requested)
        model_params = list(self.model.parameters())
        likelihood_params = []
        
        if self.fix_nu:
            for name, param in self.likelihood.named_parameters():
                # The parameter controlling 'nu' is 'raw_nu'.
                if 'raw_nu' in name:
                    param.requires_grad = False # Fix nu
                else:
                    # Optimize other parameters (including noise, subject to its prior)
                    likelihood_params.append(param)
        else:
            likelihood_params = list(self.likelihood.parameters())

        optimizer = torch.optim.Adam(model_params + likelihood_params, lr=0.1)
        
        # --- FIX: Learning Rate Scheduler for stability ---
        # Decaying the learning rate helps stabilize convergence
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.training_iterations // 3, gamma=0.3)
        # ----------------------------------------------------
        
        # The ELBO naturally incorporates the registered priors during optimization
        mll = VariationalELBO(self.likelihood, self.model, num_data=len(self.Y_standardized))

        # Training loop
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -mll(output, self.Y_standardized)
            
            if not torch.isfinite(loss):
                if verbose:
                    print(f"Warning: Loss is {loss.item()} at iteration {i}. Stopping optimization.")
                break
                
            loss.backward()
            optimizer.step()

            # Clip parameters to ensure they stay within constraint bounds
            with torch.no_grad():
                # Clip lengthscale raw parameters to prevent constraint violations
                if hasattr(self.model.covar_module, 'raw_lengthscale'):
                    # For RBF kernel, raw_lengthscale is the unconstrained parameter
                    # The constraint GreaterThan(2.5e-2) with transform=None means:
                    # constrained_value = raw_value, so we clip raw_value >= 2.5e-2
                    self.model.covar_module.raw_lengthscale.clamp_(min=2.5e-2)

                # Clip likelihood noise parameter if it exists
                for name, param in self.likelihood.named_parameters():
                    if 'raw_noise' in name and param.requires_grad:
                        # Noise should be positive, clip to prevent negative values
                        param.clamp_(min=1e-6)

            scheduler.step() # Update learning rate
            
            if verbose and (i+1) % 50 == 0:
                 print(f'Iter {i+1}/{self.training_iterations} - Loss: {loss.item():.3f} - LR: {scheduler.get_last_lr()[0]:.4f}')

    def _get_predictive_distribution(self, X, include_likelihood=True):
        """Helper to compute distributions, handle MC samples, and unstandardize."""
        self.model.eval()
        self.likelihood.eval()
        X = X.to(dtype=torch.float64)
        
        # Use a fixed number of MC samples for stable predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(100):
            latent_dist = self.model(X)
            
            if include_likelihood:
                # Predictive distribution (Student-t marginals via MC sampling)
                predictive_dist = self.likelihood(latent_dist)
            else:
                # Latent distribution (Gaussian)
                predictive_dist = latent_dist

            mean = predictive_dist.mean
            variance = predictive_dist.variance.clamp(min=1e-9)

            # Handle MC Samples (Average across the sample dimension dim=0)
            if include_likelihood:
                 mean = mean.mean(dim=0)
                 variance = variance.mean(dim=0)

            # Unstandardize (Using Robust Stats)
            mean_unstd = mean * self._Y_scale + self._Y_median
            variance_unstd = variance * (self._Y_scale ** 2)
            
            # Rebuild the distribution
            if mean_unstd.dim() == 1:
                covar = torch.diag(variance_unstd)
            else:
                covar = torch.diag_embed(variance_unstd)

            # Returns a MultivariateNormal parameterized by the Student-t mean and variance.
            return gpytorch.distributions.MultivariateNormal(mean_unstd, covar)

    def posterior(self, X: torch.Tensor, **kwargs) -> GPyTorchPosterior:
        latent_dist = self._get_predictive_distribution(X, include_likelihood=False)
        return GPyTorchPosterior(distribution=latent_dist)

    def predictive_posterior(self, X: torch.Tensor):
        """
        Compute predictive posterior (including likelihood).
        Used for outlier detection by DiagnosticGPWrapper.
        """
        predictive_dist = self._get_predictive_distribution(X, include_likelihood=True)
        
        # Wrapper for interface compatibility
        class PredictiveDistWrapper:
            def __init__(self, dist):
                self.dist = dist
                self.mean = dist.mean
                self.stddev = dist.stddev
        
        return PredictiveDistWrapper(predictive_dist)

    def get_degrees_of_freedom(self) -> float:
        return self.likelihood.nu.item()

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

# =====================================================================
# 3. Helper Function
# =====================================================================

def fit_student_t_gp(
    model: StudentTGP,
    verbose: bool = False
) -> None:
    model._fit_model(verbose=verbose)
"""
Residual-based A2RCGP implementations.

This module provides two residual-based A2RCGP variants:
1. ResidualA2RCGP: Standard residual fitting with WLOO-CV
2. ExperimentalResidualA2RCGP: Residual fitting with GP MLL optimization for outer model
"""

import torch
from typing import Optional

from rcgp.models.robust_gp import RobustConjugateGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal


def fixed_zero_center_fn(x: torch.Tensor) -> torch.Tensor:
    """Center function that always returns 0."""
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


class ResidualA2RCGP:
    """
    Residual-based A2RCGP implementation.

    - Inner model: fits on lagged data with own outcome transform
    - Outer model: fits on residuals with own outcome transform and zero center
    - Posterior: combines inner + outer predictions
    """

    def __init__(self, train_X, train_Y, inner_weighting_function, outer_weighting_function, **kwargs):
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        self._raw_train_X = train_X.clone()
        self._raw_train_Y = train_Y.clone()
        self._original_outer_weighting = outer_weighting_function
        self._original_inner_weighting = inner_weighting_function

        # Each model gets its own outcome transform
        from botorch.models.transforms.outcome import Standardize
        self._inner_outcome_transform = Standardize(m=train_Y.shape[-1])
        self._outer_outcome_transform = Standardize(m=train_Y.shape[-1])

        # Prepare lagged data for inner model
        N = train_X.shape[0]
        if N > 1:
            inner_train_X = train_X[:-1]
            inner_train_Y = train_Y[:-1]
        else:
            inner_train_X = train_X[:0]
            inner_train_Y = train_Y[:0]

        # Create inner RCGP
        self.inner_rcgp = RobustConjugateGP(
            train_X=inner_train_X,
            train_Y=inner_train_Y,
            weighting_function=inner_weighting_function,
            outcome_transform=self._inner_outcome_transform
        )

        self.outer_rcgp = None
        self._is_fitted = False

    def fit(self, inner_param_dict, outer_param_dict, objective_type="wloo-cv", **kwargs):
        verbose = kwargs.get('verbose', False)

        # Fit inner model
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            if verbose:
                print("Inner RCGP has no data. Skipping inner fitting.")
            inner_predictions = torch.zeros_like(self._raw_train_Y)
        else:
            if verbose:
                print("Fitting inner RCGP...")
            self.inner_rcgp.fit(inner_param_dict, objective_type=objective_type, **kwargs)

            # Freeze inner model parameters after fitting
            for param in self.inner_rcgp.parameters():
                param.requires_grad_(False)

            with torch.no_grad():
                self.inner_rcgp.eval()
                inner_posterior = self.inner_rcgp.posterior(self._raw_train_X)
                inner_predictions = inner_posterior.mean
                if inner_predictions.dim() == 2:
                    inner_predictions = inner_predictions.squeeze(-1)
                if inner_predictions.dim() == 1:
                    inner_predictions = inner_predictions.unsqueeze(-1)

        # Compute residuals
        residual_Y = self._raw_train_Y - inner_predictions
        if verbose:
            print(f"Residuals: mean={residual_Y.mean().item():.6f}, std={residual_Y.std().item():.6f}")

        # Create outer model with residuals
        outer_weighting = type(self._original_outer_weighting)(
            plateau_width=self._original_outer_weighting.plateau_width.item(),
            c=self._original_outer_weighting.c.item()
        )
        outer_weighting.set_center_function(fixed_zero_center_fn)

        self.outer_rcgp = RobustConjugateGP(
            train_X=self._raw_train_X,
            train_Y=residual_Y,
            weighting_function=outer_weighting,
            outcome_transform=self._outer_outcome_transform
        )

        if verbose:
            print("Fitting outer RCGP on residuals...")
        self.outer_rcgp.fit(outer_param_dict, objective_type=objective_type, **kwargs)

        # Freeze outer model parameters after fitting
        for param in self.outer_rcgp.parameters():
            param.requires_grad_(False)

        self._is_fitted = True

    def posterior(self, X, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing posterior")

        # Get predictions from both models
        # NOTE: We don't use torch.no_grad() here because we need gradients w.r.t. X
        # for acquisition function optimization. The model parameters are already frozen.
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            inner_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
            inner_variance = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        else:
            self.inner_rcgp.eval()
            inner_posterior = self.inner_rcgp.posterior(X)
            inner_mean = inner_posterior.mean
            inner_variance = inner_posterior.variance

        self.outer_rcgp.eval()
        outer_posterior = self.outer_rcgp.posterior(X)
        outer_mean = outer_posterior.mean
        outer_variance = outer_posterior.variance

        # Combine predictions
        combined_mean = inner_mean + outer_mean
        combined_variance = inner_variance + outer_variance

        combined_mvn = MultivariateNormal(
            mean=combined_mean.squeeze(-1),
            covariance_matrix=torch.diag_embed(combined_variance.squeeze(-1))
        )
        return GPyTorchPosterior(combined_mvn)

    def eval(self):
        """Set both models to eval mode for compatibility with plotting utilities."""
        if hasattr(self, 'inner_rcgp'):
            self.inner_rcgp.eval()
        if hasattr(self, 'outer_rcgp') and self.outer_rcgp is not None:
            self.outer_rcgp.eval()
        return self

    def get_inner_weights(self):
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            return torch.tensor([])
        return self.inner_rcgp.get_weights()

    def get_outer_weights(self):
        if not self._is_fitted:
            raise RuntimeError("Must fit before accessing weights")
        return self.outer_rcgp.get_weights()


class ExperimentalResidualA2RCGP:
    """
    ResidualA2RCGP where outer model uses GP-fitted parameters on residuals.

    - Inner model: Standard WLOO-CV fitting on lagged data
    - Outer model: GP MLL optimization on residuals with zero center
    - Posterior: combines inner + outer predictions
    """

    def __init__(self, train_X, train_Y, inner_weighting_function, outer_weighting_function, **kwargs):
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        self._raw_train_X = train_X.clone()
        self._raw_train_Y = train_Y.clone()
        self._original_outer_weighting = outer_weighting_function
        self._original_inner_weighting = inner_weighting_function

        # Each model gets its own outcome transform
        from botorch.models.transforms.outcome import Standardize
        self._inner_outcome_transform = Standardize(m=train_Y.shape[-1])
        self._outer_outcome_transform = Standardize(m=train_Y.shape[-1])

        # Prepare lagged data for inner model
        N = train_X.shape[0]
        if N > 1:
            inner_train_X = train_X[:-1]
            inner_train_Y = train_Y[:-1]
        else:
            inner_train_X = train_X[:0]
            inner_train_Y = train_Y[:0]

        # Create inner RCGP
        self.inner_rcgp = RobustConjugateGP(
            train_X=inner_train_X,
            train_Y=inner_train_Y,
            weighting_function=inner_weighting_function,
            outcome_transform=self._inner_outcome_transform
        )

        self.outer_rcgp = None
        self._is_fitted = False

    def fit(self, inner_param_dict, outer_param_dict, objective_type="wloo-cv", **kwargs):
        verbose = kwargs.get('verbose', False)

        # Fit inner model (normal WLOO-CV)
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            if verbose:
                print("Inner RCGP has no data. Skipping inner fitting.")
            inner_predictions = torch.zeros_like(self._raw_train_Y)
        else:
            if verbose:
                print("Fitting inner RCGP...")
            self.inner_rcgp.fit(inner_param_dict, objective_type=objective_type, **kwargs)

            # Freeze inner model parameters after fitting
            for param in self.inner_rcgp.parameters():
                param.requires_grad_(False)

            with torch.no_grad():
                self.inner_rcgp.eval()
                inner_posterior = self.inner_rcgp.posterior(self._raw_train_X)
                inner_predictions = inner_posterior.mean
                if inner_predictions.dim() == 2:
                    inner_predictions = inner_predictions.squeeze(-1)
                if inner_predictions.dim() == 1:
                    inner_predictions = inner_predictions.unsqueeze(-1)

        # Compute residuals
        residual_Y = self._raw_train_Y - inner_predictions
        if verbose:
            print(f"Residuals: mean={residual_Y.mean().item():.6f}, std={residual_Y.std().item():.6f}")

        # Create outer model with residuals and zero center
        outer_weighting = type(self._original_outer_weighting)(
            plateau_width=self._original_outer_weighting.plateau_width.item(),
            c=self._original_outer_weighting.c.item()
        )
        outer_weighting.set_center_function(fixed_zero_center_fn)

        self.outer_rcgp = RobustConjugateGP(
            train_X=self._raw_train_X,
            train_Y=residual_Y,
            weighting_function=outer_weighting,
            outcome_transform=self._outer_outcome_transform
        )

        # Fit outer model with GP parameters on residuals
        if verbose:
            print("Fitting outer RCGP on residuals with GP parameters...")

        # Set manual parameters for outer model
        for param_name, param_config in outer_param_dict.items():
            if param_config["method"] == "manual" and "value" in param_config:
                if param_name == "plateau_width":
                    self.outer_rcgp.weighting_function.plateau_width = torch.tensor(param_config["value"], dtype=torch.float64)
                elif param_name == "c":
                    self.outer_rcgp.weighting_function.c = torch.tensor(param_config["value"], dtype=torch.float64)

        # Step 1: Fit sigma and mean using separate standard GP with MLL
        from botorch.models import SingleTaskGP
        if verbose:
            print("  Step 1 - Fitting sigma and mean using standard GP with MLL on residuals...")

        standard_gp = SingleTaskGP(self._raw_train_X, residual_Y)
        standard_gp.train()
        standard_mll = ExactMarginalLogLikelihood(standard_gp.likelihood, standard_gp)
        fit_gpytorch_mll(standard_mll)
        standard_gp.eval()

        # Extract fitted noise (sigma) and mean from standard GP
        fitted_noise = standard_gp.likelihood.noise.clone()
        fitted_mean_constant = standard_gp.mean_module.constant.clone()

        if verbose:
            print(f"    Fitted noise (sigma²): {fitted_noise.item():.6f}")
            print(f"    Fitted mean constant: {fitted_mean_constant.item():.6f}")

        # Step 2: Copy fitted parameters to outer RCGP and freeze them
        self.outer_rcgp.likelihood.noise = fitted_noise
        self.outer_rcgp.mean_module.constant = fitted_mean_constant

        # Freeze sigma and mean parameters
        self.outer_rcgp.likelihood.noise.requires_grad_(False)
        self.outer_rcgp.mean_module.constant.requires_grad_(False)

        if verbose:
            print("  Step 2 - Fitting kernel parameters using specified fitting objective...")

        # Step 3: Fit kernel parameters using the specified fitting objective
        self.outer_rcgp.fit(outer_param_dict, objective_type=objective_type, **kwargs)

        # Freeze outer model parameters after fitting
        for param in self.outer_rcgp.parameters():
            param.requires_grad_(False)

        self._is_fitted = True

    def posterior(self, X, **kwargs):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing posterior")

        # Get predictions from both models
        # NOTE: We don't use torch.no_grad() here because we need gradients w.r.t. X
        # for acquisition function optimization. The model parameters are already frozen.
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            inner_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
            inner_variance = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        else:
            self.inner_rcgp.eval()
            inner_posterior = self.inner_rcgp.posterior(X)
            inner_mean = inner_posterior.mean
            inner_variance = inner_posterior.variance

        self.outer_rcgp.eval()
        outer_posterior = self.outer_rcgp.posterior(X)
        outer_mean = outer_posterior.mean
        outer_variance = outer_posterior.variance

        # Combine predictions
        combined_mean = inner_mean + outer_mean
        combined_variance = inner_variance + outer_variance

        combined_mvn = MultivariateNormal(
            mean=combined_mean.squeeze(-1),
            covariance_matrix=torch.diag_embed(combined_variance.squeeze(-1))
        )
        return GPyTorchPosterior(combined_mvn)

    def eval(self):
        """Set both models to eval mode for compatibility with plotting utilities."""
        if hasattr(self, 'inner_rcgp'):
            self.inner_rcgp.eval()
        if hasattr(self, 'outer_rcgp') and self.outer_rcgp is not None:
            self.outer_rcgp.eval()
        return self

    def get_inner_weights(self):
        if self.inner_rcgp.train_inputs[0].numel() == 0:
            return torch.tensor([])
        return self.inner_rcgp.get_weights()

    def get_outer_weights(self):
        if not self._is_fitted:
            raise RuntimeError("Must fit before accessing weights")
        return self.outer_rcgp.get_weights()
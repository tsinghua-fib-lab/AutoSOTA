"""
Robust Leave-One-Out Cross-Validation (LOO-CV) for RCGP Hyperparameter Fitting.

This module provides two MLL classes for robust hyperparameter optimization:
1. RobustLeaveOneOutMLL: The exact, unweighted LOO-CV objective from the RCGP paper.
2. WeightedRobustLeaveOneOutMLL: An experimental, weighted version of the LOO-CV objective.

The unweighted version is the theoretically grounded approach from the paper, while
the weighted version provides a more aggressive down-weighting of outliers, which can
be useful for comparative experiments.
"""

import torch
import warnings
from torch import Tensor
from gpytorch.mlls import MarginalLogLikelihood
from typing import Tuple

from rcgp.models.robust_gp import RobustConjugateGP


class RobustLeaveOneOutMLL(MarginalLogLikelihood):
    """
    Robust Leave-One-Out Cross-Validation (LOO-CV) Marginal Log-Likelihood.

    This class computes the exact LOO-CV objective for a Robust Conjugate GP model.
    The final loss is the unweighted average of the log probabilities of each observation 
    under its respective LOO distribution.

    For use with standard GPyTorch optimizers (like fit_gpytorch_mll), this class
    returns the positive objective, which the optimizer will then minimize the negative of.

    Objective Maximized:
      LOO-CV(θ, σ²) = (1/n) * Σᵢ log p^w(yᵢ | X, y₋ᵢ, θ, σ²)
    """

    def __init__(self, likelihood, model: RobustConjugateGP):
        """
        Initialize the Robust LOO-CV MLL.

        Args:
            likelihood: A GPyTorch GaussianLikelihood.
            model: The RobustConjugateGP model instance.
        """
        super().__init__(likelihood, model)

    def _compute_loo_predictions(self) -> Tuple[Tensor, Tensor]:
        """
        Compute the leave-one-out predictive means and variances analytically.
        """
        # --- 1. Get model components ---
        train_x = self.model.train_inputs[0]
        train_y = self.model.train_targets.squeeze(-1)
        # Ensure noise_var is treated correctly, squeezing if it's a 1-element tensor
        noise_var = self.likelihood.noise.squeeze()

        # --- 2. Compute RCGP-specific terms ---
        # J_matrix is a [N, N] diagonal matrix.
        # gradient_log_weights includes the factor of 2.
        _, J_matrix, gradient_log_weights = self.model._get_robust_components(
            train_x, train_y
        )

        # --- 3. Form the full robust covariance matrix K_robust ---
        covar_module = self.model.covar_module
        K = covar_module(train_x).evaluate()
        # This addition is correct as K and J_matrix are both [N, N].
        K_robust = K + noise_var * J_matrix
        
        # --- 4. Compute the corrected residuals z ---
        mean_module = self.model.mean_module
        mean_train = mean_module(train_x).squeeze(-1)
        z = train_y - mean_train - noise_var * gradient_log_weights
        
        # --- 5. Compute inverse components efficiently ---
        jitter = 1e-6
        N = K_robust.size(0)
        eye_mat = torch.eye(N, dtype=K_robust.dtype, device=K_robust.device)
        K_robust_reg = K_robust + jitter * eye_mat
        
        try:
            L = torch.linalg.cholesky(K_robust_reg)
            # Solve K_inv @ z
            K_inv_z = torch.cholesky_solve(z.unsqueeze(-1), L).squeeze(-1)
            
            # IMPROVEMENT: Efficiently compute diag(K_inv) without full inversion.
            # 1. Compute L_inv using triangular solve (more efficient/stable than torch.inverse(L))
            # This solves L @ X = I for X (which is L_inv).
            L_inv = torch.linalg.solve_triangular(L, eye_mat, upper=False)
            # 2. Compute the diagonal of K_inv (K_inv = L_inv^T @ L_inv) as sum of squares of columns of L_inv
            K_inv_diag = torch.sum(L_inv**2, dim=0)

        except torch.linalg.LinAlgError:
            warnings.warn("Cholesky decomposition failed in LOO-CV. Using direct torch.inverse().")
            K_inv = torch.inverse(K_robust_reg)
            K_inv_z = K_inv @ z
            K_inv_diag = torch.diagonal(K_inv)
        
        # --- 6. Compute LOO predictions using the analytical formulas ---
        # Prevent division by zero
        K_inv_diag_safe = K_inv_diag + 1e-10

        loo_means = z + mean_train - K_inv_z / K_inv_diag_safe
        
        # FIX: J_matrix is [N, N], but we need its diagonal [N] here to subtract from K_inv_diag [N].
        J_diag = torch.diagonal(J_matrix)
        latent_loo_variances = 1.0 / K_inv_diag_safe - (noise_var * J_diag)
        
        loo_variances = latent_loo_variances + noise_var
        # Ensure variances are positive for numerical stability
        loo_variances = torch.clamp(loo_variances, min=1e-8)
        
        return loo_means, loo_variances
    
    def forward(self, function_dist, target: Tensor) -> Tensor:
        """
        Computes the LOO-CV objective for hyperparameter optimization.

        GPyTorch optimizers minimize the negative of this value.

        Args:
            function_dist: The prior distribution (unused, for API compatibility).
            target: The training targets (tensor of shape [n]).

        Returns:
            The mean log predictive probability (a scalar tensor).
        """
        if not self.model.training:
            raise RuntimeError(
                "The RobustLeaveOneOutMLL should only be used in training mode."
            )
        
        loo_means, loo_variances = self._compute_loo_predictions()
        
        log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=target.dtype, device=target.device))
        squared_errors = (target.squeeze(-1) - loo_means)**2
        
        # Log probability density of the Gaussian distribution
        log_probs = -0.5 * (log_2pi + torch.log(loo_variances) + squared_errors / loo_variances)
        
        # Use the mean for optimization stability across varying dataset sizes.
        loo_cv_objective = log_probs.mean()
        
        # Return the positive objective; GPyTorch handles the negation for minimization.
        return loo_cv_objective


class WeightedRobustLeaveOneOutMLL(RobustLeaveOneOutMLL):
    """
    Weighted Robust Leave-One-Out Cross-Validation (WLOO-CV) Marginal Log-Likelihood.

    This experimental class computes a weighted version of the LOO-CV objective.
    Weights are normalized and detached for stable optimization.

    Objective Maximized (GPyTorch minimizes the negative of this):
      WLOO-CV(θ, σ²) = (1/n) * Σᵢ norm(wᵢ) * log p^w(yᵢ | X, y₋ᵢ, θ, σ²)
    """
    def forward(self, function_dist, target: Tensor) -> Tensor:
        """
        Computes the weighted LOO-CV objective.
        """
        if not self.model.training:
            raise RuntimeError(
                "The WeightedRobustLeaveOneOutMLL should only be used in training mode."
            )

        # 1. Inherit the LOO prediction calculation from the parent class.
        loo_means, loo_variances = self._compute_loo_predictions()
        
        # 2. Calculate the standard, unweighted log probabilities.
        train_y = target.squeeze(-1)
        log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=train_y.dtype, device=train_y.device))
        squared_errors = (train_y - loo_means)**2
        log_probs = -0.5 * (log_2pi + torch.log(loo_variances) + squared_errors / loo_variances)

        # 3. Get the explicit weights w_i from the model's weighting function.
        train_x = self.model.train_inputs[0]
        sigma = torch.sqrt(self.likelihood.noise.squeeze())
        weights = self.model.weighting_function.weight(train_x, train_y, sigma=sigma)
        
        # 4. Detach and Normalize weights.
        # TODO: not sure what's best, detaching the weights is possible because we can use any weights that reflects the uncertainty of the model. But is it the right call?
        # Currently attaching or detaching yield similar results, I will detach for stability.
        # IMPROVEMENT: Detaching prevents gradients from flowing through the weights.
        weights_1 = weights.detach()
        # weights_1 = weights
        # IMPROVEMENT: Normalize weights so that mean(weights) = 1. This keeps the loss scale consistent.
        normalized_weights = weights_1 / (weights_1.sum() + 1e-10)
        
        # 5. Compute the weighted average of the log probabilities.
        wloo_cv_objective = (normalized_weights * log_probs).mean()
        
        # 6. Return the positive objective; GPyTorch handles the negation for minimization.
        return wloo_cv_objective
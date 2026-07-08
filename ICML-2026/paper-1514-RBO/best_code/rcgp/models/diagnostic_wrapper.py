"""
Diagnostic GP Wrapper for Outlier-Diagnostic Bayesian Optimization (OD-BO).
Based on Martinez-Cantin et al. (2018).
"""

import torch
from typing import Dict, Any, Optional
from scipy.stats import t as student_t_dist
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior


class DiagnosticGPWrapper(Model):
    """
    Wrapper that periodically diagnoses outliers using a robust Student-t GP
    and trains a standard GP on the filtered clean data.
    
    The wrapper maintains the standard model interface (posterior, forward)
    and abstracts the filtering process from the user. The acquisition function
    is separate and calls this model's posterior() method.
    """
    _num_outputs = 1
    
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, config: Dict[str, Any]):
        """
        Initialize the diagnostic wrapper.
        
        Args:
            train_X: Training inputs [n, d]
            train_Y: Training targets [n, 1]
            config: Configuration dictionary with:
                - n_init: Minimum points before diagnosis starts (default: 10)
                - n_schedule: Frequency of diagnosis (default: 2)
                - nu: Degrees of freedom for Student-t (default: 4.0)
                - alpha: Outlier threshold (default: 0.05)
                - fitting_kwargs: kwargs for Student-t GP fitting
                - model_kwargs: kwargs for the underlying GP model
        """
        super().__init__()
        
        # Store all data
        self.X_all = train_X
        self.Y_all = train_Y
        self.config = config
        
        # Initialize state
        self.inlier_mask = torch.ones(train_Y.size(0), dtype=torch.bool, device=train_X.device)
        self.iteration_count = train_Y.size(0)
        
        # The underlying model (trained on clean data)
        self.model = None
        
        # Fit model
        self._fit_model()
    
    @property
    def num_outputs(self) -> int:
        """Number of model outputs."""
        return self._num_outputs
    
    def _check_schedule(self) -> bool:
        """Check if diagnostics should run based on n_init and n_schedule."""
        t = self.iteration_count
        n_init = self.config.get('n_init', 10)
        n_schedule = self.config.get('n_schedule', 2)
        
        if t < n_init:
            return False
        
        # Run diagnosis at n_init and then every n_schedule iterations
        return (t - n_init) % n_schedule == 0
    
    def _run_diagnostics(self):
        """Fit the robust model and identify outliers."""
        from bo_framework.models.factory import create_student_t_gp_model
        
        # Create and fit Student-t GP for diagnosis
        nu = self.config.get('nu', 4.0)
        fitting_kwargs = self.config.get('fitting_kwargs', {})
        
        diagnostic_model = create_student_t_gp_model(
            self.X_all,
            self.Y_all,
            degrees_of_freedom=nu,
            **fitting_kwargs
        )
        
        # Get predictive distribution for outlier detection
        predictive_dist = diagnostic_model.predictive_posterior(self.X_all)
        mean = predictive_dist.mean
        stddev = predictive_dist.stddev
        
        # Calculate critical t-value for confidence interval
        alpha = self.config.get('alpha', 0.05)
        # Use the actual degrees of freedom from the fitted model
        df = diagnostic_model.get_degrees_of_freedom()
        t_crit_np = student_t_dist.ppf(1 - alpha / 2, df)
        # Convert to PyTorch tensor with same device/dtype as mean
        t_crit = torch.tensor(t_crit_np, dtype=mean.dtype, device=mean.device)
        
        # Compute confidence bounds
        lower = mean - t_crit * stddev
        upper = mean + t_crit * stddev
        
        # Identify inliers (ensure proper dimensions)
        Y_flat = self.Y_all.squeeze(-1)
        
        # Handle VI models that may return multiple samples
        if mean.dim() > 1:
            # If shape is [samples, points], take mean across samples
            if mean.shape[1] == len(Y_flat):
                mean = mean.mean(dim=0)
            else:
                mean = mean.squeeze()

        if lower.dim() > 1:
            # If shape is [samples, points], take mean across samples
            if lower.shape[1] == len(Y_flat):
                lower = lower.mean(dim=0)
            else:
                lower = lower.squeeze()

        if upper.dim() > 1:
            # If shape is [samples, points], take mean across samples
            if upper.shape[1] == len(Y_flat):
                upper = upper.mean(dim=0)
            else:
                upper = upper.squeeze()
        
        # Create mask
        new_mask = (Y_flat >= lower) & (Y_flat <= upper)
        
        # Safeguard: keep at least 50% of data (Martinez-Cantin constraint)
        if new_mask.sum() < len(self.Y_all) / 2:
            print(f"[DiagnosticWrapper] Warning: >50% outliers detected ({new_mask.sum()}/{len(self.Y_all)} inliers). "
                  f"Keeping previous mask.")
        else:
            self.inlier_mask = new_mask
            if self.config.get('verbose', False):
                print(f"[DiagnosticWrapper] Diagnosed {(~new_mask).sum()} outliers, "
                      f"keeping {new_mask.sum()}/{len(self.Y_all)} inliers.")
    
    def _fit_model(self):
        """Manage scheduling and fitting of the underlying model."""
        from bo_framework.models.factory import create_gp_model
        
        # Run diagnostics if scheduled
        if self._check_schedule():
            self._run_diagnostics()
        
        # Get clean data
        X_inliers = self.X_all[self.inlier_mask]
        Y_inliers = self.Y_all[self.inlier_mask]
        
        # Train standard GP model on clean data
        model_kwargs = self.config.get('model_kwargs', {})
        
        # Ensure fit_hyperparameters is set if not already in model_kwargs
        if 'fit_hyperparameters' not in model_kwargs:
            model_kwargs['fit_hyperparameters'] = True
            
        self.model = create_gp_model(
            X_inliers, Y_inliers,
            **model_kwargs
        )
    
    def forward(self, X: torch.Tensor) -> torch.distributions.MultivariateNormal:
        """
        Forward pass for the model.
        
        Args:
            X: Input tensor [batch_shape, n, d]
            
        Returns:
            MultivariateNormal distribution
        """
        return self.model.forward(X)
    
    def posterior(self, X: torch.Tensor, **kwargs) -> GPyTorchPosterior:
        """
        Return posterior from the underlying model (trained on clean data).
        
        Args:
            X: Points at which to evaluate posterior [m, d]
            **kwargs: Additional arguments for posterior computation
            
        Returns:
            GPyTorchPosterior: Posterior distribution from the clean model
        """
        return self.model.posterior(X, **kwargs)
    
    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        """
        Add new observations and return updated model.
        
        Args:
            X: New input points [m, d]
            Y: New target values [m, 1]
            **kwargs: Additional arguments
            
        Returns:
            DiagnosticGPWrapper: New instance with updated data
        """
        # Concatenate new observations
        new_X = torch.cat([self.X_all, X])
        new_Y = torch.cat([self.Y_all, Y])
        
        # Return new instance (BoTorch convention)
        return DiagnosticGPWrapper(new_X, new_Y, self.config)
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the current state.
        
        Returns:
            Dictionary with diagnostic info
        """
        return {
            'total_points': len(self.Y_all),
            'num_inliers': self.inlier_mask.sum().item(),
            'num_outliers': (~self.inlier_mask).sum().item(),
            'outlier_indices': torch.where(~self.inlier_mask)[0].tolist(),
            'underlying_model': self.model
        }
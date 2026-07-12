from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from torchtyping import TensorType
import torch.nn as nn

import sys
sys.path.extend(['./', '../', '../../'])

class StdScaler(nn.Module):
    """PyTorch StdScaler for normalization."""
    def __init__(self, dim=-1, keepdim=True, minimum_scale=1e-10):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
    
    def forward(self, x, observed_indicator=None):
        """
        Standardize input tensor.
        
        Args:
            x: Tensor of shape (..., time, features) or (..., time)
            observed_indicator: Binary tensor indicating observed values
            
        Returns:
            Tuple of (scaled_x, loc, scale)
        """
        if observed_indicator is None:
            observed_indicator = torch.ones_like(x)
        
        num_observed = observed_indicator.sum(dim=self.dim, keepdim=self.keepdim)
        loc = (x * observed_indicator).sum(dim=self.dim, keepdim=self.keepdim) / num_observed.clamp(min=1)
        
        centered = (x - loc) * observed_indicator
        variance = (centered ** 2).sum(dim=self.dim, keepdim=self.keepdim) / num_observed.clamp(min=1)
        scale = torch.sqrt(variance).clamp(min=self.minimum_scale)
        
        return x, loc, scale


PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values", "mean"]


class RegimeFlowBase(pl.LightningModule):
    """
    Base class for time series flow matching models.
    
    This class implements the core flow matching algorithm for time series generation,
    including forward path computation, loss calculation, and sampling via ODE integration.
    """
    
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        optimizer_params: dict,
        prior_params: dict,
        matching: str = "random",
        normalization: str | None = None,
        num_steps: int = 16,
        sigm: float = 0.001,
        solver: str = "euler",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        # Normalization
        if normalization == "zscore": self.scaler = StdScaler(dim=1, keepdim=True, minimum_scale=1)
        else: raise NotImplementedError(f"{normalization} scaler is not implemented.")
        
        # Lengths
        self.context_length = context_length
        self.prior_context_length = (
            context_length
            if "context_freqs" not in prior_params.keys()
            else prior_params["context_freqs"] * prediction_length
        )
        self.prediction_length = prediction_length
        
        # Optimizer
        self.optimizer_params = optimizer_params
        
        # Prior
        self.prior = 'BLR'
        self.q0 = None  # Will be initialized in setup()
        
        # Flow matching
        self.matching = matching
        self.num_steps = num_steps
        self.solver = solver
        self.num_samples = 1
        
        # Noise schedule
        self.sigmin = sigm
        self.sigmax = 1
        
        # Quantile range for guidance
        self.quantile_range = (0.1, 0.9)

    def _extract_features(self, data):
        """Extract features from data. Must be implemented by subclasses."""
        raise NotImplementedError()

    def configure_optimizers(self):
        """Configure optimizer with ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mse",
                "interval": "epoch",
                "frequency": 10,
            }
        }

    def forward_path(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float],
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
    ]:
        """
        Compute the forward path for flow matching.
        
        Uses linear interpolation with noise schedule:
            x_t = t*x1 + (1-t)*x0 + sigma(t)*epsilon
            velocity = x1 - x0 + (sigma_min - sigma_max)*epsilon
        
        Args:
            x1: Clean target data (B, L, C)
            x0: Noisy source data (B, L, C)
            t: Time steps (B, 1) or (B,) in [0, 1]
            
        Returns:
            x_t: Interpolated data at time t
            velocity: True flow velocity (target for model)
        """
        eps = torch.randn_like(x0)
        sig_t = (1 - t) * self.sigmax + t * self.sigmin
        
        x_t = t * x1 + (1 - t) * x0 + sig_t * eps
        velocity = x1 - x0 + (self.sigmin - self.sigmax) * eps
        
        return x_t, velocity

    def p_losses(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float, "batch", 1],
        cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
    ) -> TensorType[float]:
        """
        Compute flow matching loss.
        
        Args:
            x1: Target (clean) data
            x0: Source (noisy) data
            t: Time steps in [0, 1]
            cond_emb: Condition embedding from ConditionEncoder (B, cond_dim)
            
        Returns:
            MSE loss between predicted and true flow velocity
        """
        # Reshape t to match x1 dimensions
        num_dims_to_add = x1.dim() - t.dim()
        if num_dims_to_add == 1:
            t = t.unsqueeze(-1)
        elif num_dims_to_add == 2:
            t = t.unsqueeze(-1).unsqueeze(-1)

        # Compute forward path
        x_t, true_velocity = self.forward_path(x1, x0, t)
        
        # Predict velocity using backbone
        predicted_velocity = self.backbone(t, x_t, cond_emb)

        # Compute MSE loss
        loss = F.mse_loss(true_velocity, predicted_velocity)
        return loss

    @torch.no_grad()
    def get_vf(
        self,
        cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
        observation: TensorType[float, "batch", "length", "num_series"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
        guidance_scale: float = 0,
        null_cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
    ):
        """
        Get vector field for ODE sampling.
        
        Args:
            cond_emb: Condition embedding (B, cond_dim)
            observation: Observed values for guidance
            observation_mask: Binary mask for observations (1=observed, 0=missing)
            guidance_scale: Guidance scale for conditional generation (0=no guidance)
            
        Returns:
            vf: Vector field module for NeuralODE integration
        """
        # Capture instance variables for closures
        num_samples = self.num_samples
        sigmin = self.sigmin
        sigmax = self.sigmax
        quantile_range = self.quantile_range
        
        def quantile_loss(y_prediction, y_target):
            """
            Compute quantile loss for guidance.
            
            Args:
                y_prediction: Predicted values (B*N, L, C)
                y_target: Target values (B*N, L, C)
                
            Returns:
                Quantile loss per element (B*N, L, C)
            """
            assert y_target.shape == y_prediction.shape
            device = y_prediction.device
            batch_size_x_num_samples, _, _ = y_target.shape
            batch_size = batch_size_x_num_samples // num_samples
            
            # Create quantile levels
            q_min, q_max = quantile_range
            q = torch.linspace(q_min, q_max, num_samples, device=device).repeat(batch_size)
            q = q[:, None, None]
            
            # Compute quantile loss
            e = y_target - y_prediction
            loss = torch.max(q * e, (q - 1) * e)
            return loss

        def score_func(t, x, model, cond_emb_inner):
            """
            Compute score function for guidance.
            
            Args:
                t: Time (scalar or tensor)
                x: Current state (B, L, C)
                model: Backbone model
                cond_emb_inner: Condition embedding
                
            Returns:
                velocity: Predicted velocity
                score: Gradient of quantile loss w.r.t. x
            """
            # Create detached copy to avoid modifying input
            x_copy = x.detach().clone().requires_grad_(True)
            
            with torch.enable_grad():
                # Predict velocity
                velocity = model(t, x_copy, cond_emb_inner)
                
                # Predict final state
                pred = x_copy + (1 - t) * velocity
                
                # Compute loss on observed values
                E = quantile_loss(pred, observation)[observation_mask == 1].sum()
                
                # Compute gradient
                grad = torch.autograd.grad(E, x_copy, create_graph=False)[0]
            
            return velocity.detach(), -grad.detach()

        class VectorField(torch.nn.Module):
            """Vector field for ODE integration."""
            
            def __init__(self, backbone, guidance_scale, sigmin, sigmax, cond_emb, null_cond_emb=None):
                super().__init__()
                self.backbone = backbone
                self.guidance_scale = guidance_scale
                self.sigmin = sigmin
                self.sigmax = sigmax
                self.cond_emb = cond_emb
                self.null_cond_emb = null_cond_emb

            def forward(self, t, x, args=None):
                """
                Compute vector field at time t.
                
                Args:
                    t: Time (scalar or tensor)
                    x: Current state (B, L, C)
                    args: Unused, for NeuralODE compatibility
                    
                Returns:
                    Velocity field at time t
                """
                if self.guidance_scale > 0 and self.null_cond_emb is not None:
                    # CFG: Classifier-Free Guidance
                    v_cond = self.backbone(t, x, self.cond_emb)
                    v_uncond = self.backbone(t, x, self.null_cond_emb)
                    velocity = v_uncond + self.guidance_scale * (v_cond - v_uncond)
                elif self.guidance_scale > 0 and observation is not None:
                    # Reconstruction guidance (original behavior)
                    velocity, score = score_func(t, x, self.backbone, self.cond_emb)
                    sig_t = (1 - t) * self.sigmax + t * self.sigmin
                    dsig_t = self.sigmin - self.sigmax
                    velocity = velocity - dsig_t * sig_t * self.guidance_scale * score
                else:
                    # Standard sampling
                    velocity = self.backbone(t, x, self.cond_emb)
                return velocity

        return VectorField(self.backbone, guidance_scale, sigmin, sigmax, cond_emb, null_cond_emb)

    @torch.no_grad()
    def sample(
        self,
        noise: TensorType[float, "batch", "length", "num_series"],
        cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
        observation: TensorType[float, "batch", "length", "num_series"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
        guidance_scale: float = 0,
        null_cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
    ) -> TensorType[float, "batch", "length", "num_series"]:
        """
        Sample from the model using ODE integration.
        
        Args:
            noise: Initial noise (x0)
            cond_emb: Condition embedding (B, cond_dim)
            observation: Observed values for guidance
            observation_mask: Mask for observations
            guidance_scale: Guidance scale (0=no guidance)
            
        Returns:
            Generated samples (x1)
        """
        if self.num_steps == 0:
            return noise.to(self.device)
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Create time span [0, 1]
        t_span = torch.linspace(0, 1, self.num_steps + 1, device=device)
        
        # Create ODE solver
        node = NeuralODE(
            self.get_vf(cond_emb, observation, observation_mask, guidance_scale, null_cond_emb), 
            solver=self.solver
        )
        
        # Integrate
        return node.trajectory(noise.to(device), t_span)[-1]

    def sample_n(
        self,
        num_samples: int,
        cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
    ) -> TensorType[float, "batch", "length", 1]:
        """
        Sample multiple trajectories from the prior.
        
        Args:
            num_samples: Number of samples to generate
            cond_emb: Condition embedding
            
        Returns:
            Generated samples
            
        Raises:
            RuntimeError: If q0 prior is not initialized
        """
        if self.q0 is None:
            raise RuntimeError(
                "q0 prior is not initialized. "
                "Make sure to call setup() before sampling."
            )
        
        device = next(self.parameters()).device
        
        # Sample from prior
        noise = self.q0(num_samples).to(device).unsqueeze(-1)
        noise = noise + self.sigmax * torch.randn_like(noise)
        
        # Sample through flow
        return self.sample(noise, cond_emb).cpu() + 1

    def fast_denoise(
        self, 
        xt: torch.Tensor, 
        t: float, 
        cond_emb: torch.Tensor | None = None, 
        steps: int = 4
    ):
        """
        Fast denoising from time t to 1.
        
        Args:
            xt: Current state at time t
            t: Current time in [0, 1]
            cond_emb: Condition embedding
            steps: Number of integration steps
            
        Returns:
            Denoised state at time 1
        """
        device = next(self.parameters()).device
        t_span = torch.linspace(t, 1, steps + 1, device=device)[:-1]
        
        node = NeuralODE(
            self.get_vf(cond_emb),
            solver=self.solver,
            sensitivity="adjoint",
        )
        
        return node.trajectory(xt.to(device), t_span)[-1]

    def fast_noise(
        self, 
        xt: torch.Tensor, 
        t: float, 
        cond_emb: torch.Tensor | None = None, 
        steps: int = 4
    ):
        """
        Fast noising from time 1 to t.
        
        Args:
            xt: Current state at time 1
            t: Target time in [0, 1]
            cond_emb: Condition embedding
            steps: Number of integration steps
            
        Returns:
            Noised state at time t
        """
        device = next(self.parameters()).device
        t_span = torch.linspace(1, t, steps + 1, device=device)[:-1]
        
        node = NeuralODE(
            self.get_vf(cond_emb),
            solver=self.solver,
            sensitivity="adjoint",
        )
        
        return node.trajectory(xt.to(device), t_span)[-1]

    def forward(self, *args):
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError()

    def training_step(self, *args):
        """Training step. Must be implemented by subclasses."""
        raise NotImplementedError()

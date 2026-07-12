"""
Sampling and ODE Solving for Flow Matching in GAT-FM.

This module provides:
1. ODE solvers for flow matching (all using torchdiffeq: Euler, RK4, midpoint, heun, dopri5, etc.)
2. Sampling utilities for generating protein predictions
3. Integration unified under torchdiffeq for all solvers

Flow Matching formulation:
- Forward process: x_t = (1-t) * x_0 + t * x_1  (linear interpolation)
- Velocity: v = x_1 - x_0
- ODE: dx/dt = v(x_t, t)
- Sampling: integrate from t=0 (noise) to t=1 (data)
"""

from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("torchdiffeq not installed. Please run: pip install torchdiffeq")


class FlowMatchingSampler:
    """
    Sampler for Flow Matching models.

    Uses torchdiffeq for ODE integration from t=0 (noise) to t=1 (data)
    with multiple solver options: 'euler', 'rk4', 'midpoint', 'heun', 'dopri5', etc.
    """

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        solver: str = 'euler',  # now passed directly to torchdiffeq
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Args:
            model: Velocity prediction model v(x_t, t, conditions)
            num_steps: Number of output intermediate points for trajectory (default 100)
            solver: torchdiffeq ODE solver method (e.g. 'euler', 'rk4', 'midpoint', 'heun', 'dopri5')
            atol: Absolute tolerance (for adaptive solvers)
            rtol: Relative tolerance (for adaptive solvers)
        """
        self.model = model
        self.num_steps = num_steps
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        device: torch.device = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate samples by integrating the flow ODE with torchdiffeq.

        Args:
            shape: Output shape (B, P)
            edge_index: Graph edges (2, E)
            cond_rna: RNA embeddings (B, 512)
            cond_dataset: Dataset IDs (B,)
            mask: Protein mask (B, P), optional
            device: Device to generate on
            return_trajectory: Whether to return intermediate states

        Returns:
            x_1: Generated protein expression (B, P)
            trajectory: (optional) Tensor of intermediate states (B, num_steps, P)
        """
        if device is None:
            device = next(self.model.parameters()).device

        # Initial state: standard normal noise
        x0 = torch.randn(shape, device=device)
        t_span = torch.linspace(0, 1, self.num_steps, device=device)

        def ode_fn(t_scalar, x):
            # x: (B, P), t_scalar: scalar tensor (essentially float)
            t_batch = torch.full((x.shape[0],), t_scalar.item(), device=device)
            return self.model(
                x_t=x,
                t=t_batch,
                edge_index=edge_index,
                cond_rna=cond_rna,
                cond_dataset=cond_dataset,
                mask=mask,
            )

        # Integrate ODE system, output shape: (num_steps, B, P)
        trajectory = odeint(
            ode_fn,
            x0,
            t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        # trajectory: (num_steps, B, P)
        if return_trajectory:
            # Return (B, num_steps, P)
            return trajectory[-1], trajectory.permute(1, 0, 2)
        return trajectory[-1]


def sample_time_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps uniformly in [0, 1].

    Args:
        batch_size: Number of timesteps to sample
        device: Device

    Returns:
        (B,) tensor of timesteps
    """
    return torch.rand(batch_size, device=device)


def sample_time_logit_normal(
    batch_size: int,
    device: torch.device,
    loc: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Sample timesteps from logit-normal distribution.

    Concentrates samples around t=0.5, which can improve training
    for flow matching.

    Args:
        batch_size: Number of timesteps
        device: Device
        loc: Location parameter
        scale: Scale parameter

    Returns:
        (B,) tensor of timesteps in [0, 1]
    """
    normal_samples = torch.randn(batch_size, device=device) * scale + loc
    return torch.sigmoid(normal_samples)


def get_flow_interpolation(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute linear interpolation and target velocity for flow matching.

    x_t = (1 - t) * x_0 + t * x_1
    v = x_1 - x_0

    Args:
        x0: Source samples (noise), shape (B, D)
        x1: Target samples (data), shape (B, D)
        t: Timesteps, shape (B,) or (B, 1)

    Returns:
        x_t: Interpolated samples (B, D)
        v: Target velocity (B, D)
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B, 1)

    x_t = (1 - t) * x0 + t * x1
    v = x1 - x0

    return x_t, v


def sample_conditional_ot(
    x0: torch.Tensor,
    x1: torch.Tensor,
    sigma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample from Conditional Optimal Transport flow matching.

    This implements the OT-CFM approach where we:
    1. Sample t ~ Uniform(0, 1)
    2. Compute x_t = (1-t) * x_0 + t * x_1 + sigma * sqrt(t(1-t)) * noise
    3. Target: v = x_1 - x_0

    Args:
        x0: Source samples (noise or prior)
        x1: Target samples (data)
        sigma: Noise scale (0 = deterministic OT)

    Returns:
        t: Sampled timesteps (B,)
        x_t: Noisy interpolated samples (B, D)
        v: Target velocity (B, D)
    """
    B = x0.shape[0]
    device = x0.device

    t = torch.rand(B, device=device)
    t_expanded = t.unsqueeze(-1)

    # Linear interpolation
    x_t = (1 - t_expanded) * x0 + t_expanded * x1

    # Add noise if sigma > 0
    if sigma > 0:
        noise_scale = sigma * torch.sqrt(t_expanded * (1 - t_expanded))
        x_t = x_t + noise_scale * torch.randn_like(x_t)

    # Target velocity
    v = x1 - x0

    return t, x_t, v


class ConditionalFlowMatcher:
    """
    Conditional Flow Matching utility class.

    Handles sampling of (t, x_t, target_v) for training.
    Compatible with torch-cfm interface.
    """

    def __init__(self, sigma: float = 0.0):
        """
        Args:
            sigma: Noise scale for stochastic OT (0 = deterministic)
        """
        self.sigma = sigma

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample time, location, and target velocity.

        Args:
            x0: Source samples (typically noise)
            x1: Target samples (data)

        Returns:
            t: Timesteps (B,)
            x_t: Interpolated samples (B, D)
            target_v: Target velocity (B, D)
        """
        return sample_conditional_ot(x0, x1, self.sigma)

    def compute_loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred_v: Predicted velocity (B, D)
            target_v: Target velocity (B, D)
            mask: Observation mask (B, D), 1 = observed

        Returns:
            Scalar loss
        """
        sq_error = (pred_v - target_v) ** 2

        if mask is not None:
            loss = (sq_error * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = sq_error.mean()

        return loss


# Alias for compatibility with torch-cfm
ExactOptimalTransportConditionalFlowMatcher = ConditionalFlowMatcher


class GuidedFlowMatchingSampler:
    """
    Guided Sampler for Flow Matching with ground truth correction.

    Uses torchdiffeq for all ODE integration with extra logic for observation correction.

    This sampler corrects observed (non-masked) proteins to their ground truth
    values at each integration step to avoid cumulative errors.

    Sampling strategy:
    1. Start from noise for masked proteins, ground truth for observed proteins
    2. In ODE function, correct observed proteins to interpolated ground truth at time t
    3. Only masked proteins evolve freely according to the learned velocity field
    """

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        solver: str = 'euler',
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Args:
            model: Velocity prediction model v(x_t, t, conditions)
            num_steps: Number of intermediate output points for trajectory (default 100)
            solver: torchdiffeq ODE solver method
            atol: Absolute tolerance (for adaptive solvers)
            rtol: Relative tolerance (for adaptive solvers)
        """
        self.model = model
        self.num_steps = num_steps
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def sample(
        self,
        x1_true: torch.Tensor,  # (B, P)
        sample_mask: torch.Tensor,  # (B, P)
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        full_mask: Optional[torch.Tensor] = None,
        device: torch.device = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate samples with ground truth guidance for observed proteins via torchdiffeq.

        Args:
            x1_true: Ground truth protein expression (B, P) - normalized
            sample_mask: Mask for proteins to predict (B, P), 1 = need to predict (masked during sampling)
            edge_index: Graph edges (2, E)
            cond_rna: RNA embeddings (B, 512)
            cond_dataset: Dataset IDs (B,)
            full_mask: Full protein observation mask (B, P), 1 = observed in dataset
            device: Device to generate on
            return_trajectory: Whether to return intermediate states

        Returns:
            x_pred: Predicted protein expression (B, P)
            trajectory: (optional) List of intermediate states (B, num_steps, P)
        """
        if device is None:
            device = next(self.model.parameters()).device

        shape = x1_true.shape

        x0 = torch.randn(shape, device=device)

        # Observed mask: 1 means we use ground truth for correction
        observed_mask = 1.0 - sample_mask
        if full_mask is not None:
            observed_mask = observed_mask * full_mask

        t_span = torch.linspace(0, 1, self.num_steps, device=device)

        def ode_fn(t_scalar, x):
            # x: (B, P), t_scalar: scalar tensor (float)
            t_value = t_scalar.item()
            t_batch = torch.full((x.shape[0],), t_value, device=device)
            # Interpolated ground truth at time t
            x_t_true = (1 - t_value) * x0 + t_value * x1_true
            # Apply correction: use ground truth for observed proteins, generate only masked proteins
            x_corrected = x * sample_mask + x_t_true * observed_mask
            # Pass full_mask as mask to velocity predictor so it knows observed/known proteins
            v = self.model(
                x_t=x_corrected,
                t=t_batch,
                edge_index=edge_index,
                cond_rna=cond_rna,
                cond_dataset=cond_dataset,
                mask=full_mask,
            )
            return v

        # torchdiffeq expects x0; output is (num_steps, B, P)
        trajectory = odeint(
            ode_fn,
            x0,
            t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        # After ODE, at final time, make sure observed proteins set to ground truth
        x_final = trajectory[-1]
        x1_final = x_final * sample_mask + x1_true * observed_mask

        if return_trajectory:
            # For each step in trajectory, apply observed protein correction
            t_all = t_span.cpu().numpy()
            corrected_traj = []
            for step_idx, t_step in enumerate(t_all):
                x_step = trajectory[step_idx]
                x_t_true_step = (1 - t_step) * x0 + t_step * x1_true
                x_corrected_step = x_step * sample_mask + x_t_true_step * observed_mask
                corrected_traj.append(x_corrected_step)
            corrected_traj = torch.stack(corrected_traj, dim=0)  # (num_steps, B, P)
            return x1_final, corrected_traj.permute(1, 0, 2)  # (B, num_steps, P)
        else:
            return x1_final


def create_sample_mask(
    protein_mask: torch.Tensor,
    sample_mask_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Create a sample mask by randomly masking a portion of observed proteins.

    For proteins that are observed (protein_mask=1), randomly select
    sample_mask_ratio of them to be masked (need to predict).

    Args:
        protein_mask: Original protein observation mask (B, P), 1 = observed
        sample_mask_ratio: Fraction of observed proteins to mask for prediction

    Returns:
        sample_mask: Mask for proteins to predict (B, P), 1 = need to predict
    """
    device = protein_mask.device

    random_mask = torch.rand_like(protein_mask.float())
    sample_mask = (protein_mask > 0) & (random_mask < sample_mask_ratio)
    return sample_mask.float()

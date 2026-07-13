"""Conditional flow matching for simulation-based inference.

This module implements flow matching with automatic rescaling, clean separation
of concerns, and a simplified training API.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Beta, Distribution, Normal, Uniform

from utils.networks import ResMLP, create_cf


# =============================================================================
# Configuration for Serialization
# =============================================================================

@dataclass
class FlowMatchingConfig:
    """Configuration for FlowMatching model.

    Captures all parameters needed to reconstruct a FlowMatching model.
    """
    probability_path: str
    prior: str
    base_dist: str
    dim: Tuple[int, ...]
    cond_dim: Tuple[int, ...]
    num_steps: int = 50
    probability_path_params: Dict[str, Any] = field(default_factory=dict)
    prior_params: Dict[str, Any] = field(default_factory=dict)
    base_dist_params: Dict[str, Any] = field(default_factory=dict)
    drift: Dict[str, Any] = field(default_factory=dict)
    rescale_mode: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "probability_path": self.probability_path,
            "prior": self.prior,
            "base_dist": self.base_dist,
            "dim": list(self.dim),
            "cond_dim": list(self.cond_dim),
            "num_steps": self.num_steps,
            "probability_path_params": self.probability_path_params,
            "prior_params": self.prior_params,
            "base_dist_params": self.base_dist_params,
            "drift": self.drift,
            "rescale_mode": self.rescale_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowMatchingConfig":
        """Create config from dictionary."""
        return cls(
            probability_path=data["probability_path"],
            prior=data["prior"],
            base_dist=data["base_dist"],
            dim=tuple(data["dim"]),
            cond_dim=tuple(data["cond_dim"]),
            num_steps=data.get("num_steps", 50),
            probability_path_params=data.get("probability_path_params", {}),
            prior_params=data.get("prior_params", {}),
            base_dist_params=data.get("base_dist_params", {}),
            drift=data.get("drift", {}),
            rescale_mode=data.get("rescale_mode", "none"),
        )

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FlowMatchingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ==============================================================================
# Internal Components (Probability Paths, Base Distributions)
# ==============================================================================


class _ProbabilityPath(ABC):
    """Base class for probability paths defining interpolation between source and target."""

    @abstractmethod
    def interpolate(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute interpolant x_t = α(t)x₀ + β(t)x₁ + γ(t)ε.

        Args:
            t: Time steps, shape (batch,)
            x0: Source samples, shape (batch, ...)
            x1: Target samples, shape (batch, ...)

        Returns:
            Interpolated samples x_t, shape (batch, ...)
        """
        pass

    def target_velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute target velocity field v = x₁ - x₀.

        Args:
            x0: Source samples
            x1: Target samples

        Returns:
            Target velocity
        """
        return x1 - x0


class _OptimalTransportPath(_ProbabilityPath):
    """Optimal transport path with stochastic component.

    Defines: x_t = (1-t)x₀ + tx₁ + √(2t(1-t))ε, where ε ~ N(0, I)
    """

    def interpolate(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        rsize = (-1,) + (1,) * (x0.dim() - 1)
        alpha = (1.0 - t).view(rsize)
        beta = t.view(rsize)
        gamma = torch.sqrt(2 * t * (1.0 - t)).view(rsize)
        return alpha * x0 + beta * x1 + gamma * torch.randn_like(x0)


class _OT2Path(_ProbabilityPath):
    """Optimal transport path with minimal noise.

    Defines: x_t = (1-(1-σ_min)t)x₀ + tx₁
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def interpolate(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        rsize = (-1,) + (1,) * (x0.dim() - 1)
        alpha = (1.0 - (1 - self.sigma_min) * t).view(rsize)
        beta = t.view(rsize)
        return alpha * x0 + beta * x1


class _BaseDistribution(ABC):
    """Base class for source distributions p(x₀|y)."""

    @abstractmethod
    def sample(self, cond: Tensor, nsamples: Optional[int] = None) -> Tensor:
        """Sample from base distribution.

        Args:
            cond: Conditioning variables, shape (batch, ...)
            nsamples: Number of samples per condition (if None, one per condition)

        Returns:
            Samples from base distribution
        """
        pass

    def log_prob(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """Compute log probability of samples under the base distribution.

        Args:
            x: Samples, shape (batch, *dim)
            cond: Optional conditioning variables

        Returns:
            Log probability, shape (batch,)
        """
        raise NotImplementedError


class _GaussianBase(_BaseDistribution):
    """Standard Gaussian base distribution N(0, I)."""

    def __init__(self, dim: torch.Size):
        self.dim = dim
        self.dist = Normal(torch.zeros(dim), torch.ones(dim))

    def sample(self, cond: Tensor, nsamples: Optional[int] = None) -> Tensor:
        if nsamples:
            size = torch.Size((nsamples, cond.shape[0]))
        else:
            size = torch.Size((cond.shape[0],))
        return self.dist.sample(size).to(cond.device)

    def log_prob(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """Compute log N(0, I) for each sample.

        Args:
            x: Samples, shape (batch, *dim)
            cond: Ignored (unconditional base).

        Returns:
            Log probability, shape (batch,)
        """
        d = x.shape[-1]
        return -0.5 * (x.pow(2).sum(dim=-1) + d * math.log(2 * math.pi))


class _DataEpsBase(_BaseDistribution):
    """Base distribution as data plus Gaussian noise: x₀ = y + ε."""

    def __init__(self, dim: torch.Size, eps: float = 0.01):
        self.dim = dim
        self.eps = eps
        self.dist = Normal(torch.zeros(dim), eps * torch.ones(dim))

    def sample(self, cond: Tensor, nsamples: Optional[int] = None) -> Tensor:
        if nsamples:
            size = torch.Size((nsamples, cond.shape[0]))
            noise = self.dist.sample(size).to(cond.device)
            return cond[None, :] + noise
        noise = self.dist.sample(torch.Size((cond.shape[0],))).to(cond.device)
        return cond + noise

    def log_prob(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """Compute log N(cond, eps^2 I) for each sample.

        Args:
            x: Samples, shape (batch, *dim)
            cond: Conditioning (mean), shape (batch, *dim). Required for DataEps.

        Returns:
            Log probability, shape (batch,)
        """
        if cond is None:
            raise ValueError("DataEpsBase requires conditioning (cond) for log_prob")
        d = x.shape[-1]
        diff = x - cond
        return -0.5 * (diff.pow(2).sum(dim=-1) / (self.eps ** 2) + d * math.log(2 * math.pi) + d * 2 * math.log(self.eps))


# ==============================================================================
# Main FlowMatching Class
# ==============================================================================


class FlowMatching(nn.Module):
    """Conditional flow matching for simulation-based inference.

    Learns velocity field v_θ(x_t, t, y) to transport samples from source
    distribution p(x₀|y) to target distribution p(x₁|y).

    Key features:
    - Automatic data rescaling (transparent to user)
    - Simple training API via compute_loss()
    - Flexible probability paths (OT, OT2)
    - Compatible with ResMLP and CFNet architectures

    Example:
        >>> flow = FlowMatching(
        ...     probability_path="ot2",
        ...     prior="uniform",
        ...     base_dist="gaussian",
        ...     dim=theta.shape[1:],
        ...     cond_dim=y.shape[1:],
        ...     drift={"architecture": "resmlp", "hidden_dim": [64, 64]}
        ... )
        >>> flow.set_scales(theta, y, "z_score")
        >>>
        >>> # Training
        >>> loss = flow.compute_loss(theta_batch, y_batch)
        >>>
        >>> # Sampling
        >>> samples = flow.sample(x0, y, device)
    """

    def __init__(
        self,
        probability_path: str,
        prior: str,
        base_dist: str,
        dim: torch.Size,
        cond_dim: torch.Size,
        **kwargs,
    ) -> None:
        """Initialize FlowMatching model (always conditional).

        Args:
            probability_path: Probability path type ("ot" or "ot2")
            prior: Time prior distribution ("uniform" or "power")
            base_dist: Base distribution ("gaussian" or "data_eps")
            dim: Dimension of target space (theta or x)
            cond_dim: Dimension of conditioning space (y)
            **kwargs: Additional parameters:
                - num_steps: Integration steps (default: 50)
                - probability_path_params: Path-specific parameters
                - prior_params: Time prior parameters
                - base_dist_params: Base distribution parameters
                - drift: Drift model configuration
                - rescale_mode: Rescaling mode (default: "none")
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_steps = kwargs.get("num_steps", 50)
        rescale_mode = kwargs.get("rescale_mode", "none")

        # Store configuration for serialization
        self._config = FlowMatchingConfig(
            probability_path=probability_path,
            prior=prior,
            base_dist=base_dist,
            dim=tuple(dim),
            cond_dim=tuple(cond_dim),
            num_steps=self.num_steps,
            probability_path_params=kwargs.get("probability_path_params", {}),
            prior_params=kwargs.get("prior_params", {}),
            base_dist_params=kwargs.get("base_dist_params", {}),
            drift=kwargs.get("drift", {}),
            rescale_mode=rescale_mode,
        )

        # Validate configuration
        self._validate_config(probability_path, prior, base_dist, **kwargs)

        # Build components
        self.path = self._create_path(
            probability_path, **kwargs.get("probability_path_params", {})
        )
        self.time_prior = self._create_time_prior(prior, **kwargs.get("prior_params", {}))
        self.base_dist = self._create_base_dist(
            base_dist, dim, **kwargs.get("base_dist_params", {})
        )
        self.drift = self._build_drift(dim, cond_dim, **kwargs.get("drift", {}))

        # Initialize rescalers based on mode
        from utils.rescaling import create_rescaler
        self.target_rescaler = create_rescaler(rescale_mode)
        self.cond_rescaler = create_rescaler(rescale_mode)

    def forward(self, xt: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        """Compute velocity field v_θ(x_t, t, y).

        Args:
            xt: Interpolated samples, shape (batch, ...)
            cond: Conditioning variables, shape (batch, ...)
            t: Time steps, shape (batch,)

        Returns:
            Predicted velocity, shape (batch, ...)
        """
        t = t.view(-1, 1)
        return self.drift((xt, t, cond))

    def compute_loss(
        self, target: Tensor, cond: Tensor, t: Optional[Tensor] = None,
        velocity_reg: float = 0.0,
    ) -> Tensor:
        """Compute flow matching loss with automatic rescaling.

        Implements: E_{t,x₀,x₁}[||v_θ(x_t, t, y) - (x₁ - x₀)||²] + λ·||v_θ||²

        Args:
            target: Target samples (x₁), shape (batch, ...)
            cond: Conditioning variables (y), shape (batch, ...)
            t: Time steps. If None, sampled from prior
            velocity_reg: Weight for velocity norm regularization (0 = off)

        Returns:
            Scalar loss (mean squared error + optional regularization)
        """
        # Automatic rescaling
        target_scaled, cond_scaled = self._auto_rescale(target, cond)

        # Sample time if not provided
        if t is None:
            t = self.time_prior.sample(torch.Size((target.shape[0],))).to(target.device)

        # Sample source
        source = self.base_dist.sample(cond_scaled)

        # Interpolate
        xt = self.path.interpolate(t, source, target_scaled)

        # Compute velocities
        v_pred = self.forward(xt, cond_scaled, t)
        v_target = self.path.target_velocity(source, target_scaled)

        # MSE loss + optional velocity regularization
        loss = (v_pred - v_target).pow(2).mean()
        if velocity_reg > 0:
            loss = loss + velocity_reg * v_pred.pow(2).mean()
        return loss

    def compute_loss_with_source(
        self, source: Tensor, target: Tensor, cond: Tensor, t: Optional[Tensor] = None,
        velocity_reg: float = 0.0,
    ) -> Tensor:
        """Compute flow matching loss with a custom source distribution.

        Use this when the source samples come from a different distribution
        (e.g., samples from another model) rather than the standard base_dist.

        Args:
            source: Source samples (x₀), shape (batch, ...) - in data space
            target: Target samples (x₁), shape (batch, ...)
            cond: Conditioning variables (y), shape (batch, ...)
            t: Time steps. If None, sampled from prior
            velocity_reg: Weight for velocity norm regularization (0 = off)

        Returns:
            Scalar loss (mean squared error + optional regularization)
        """
        # Rescale all inputs so the velocity field operates in rescaled space.
        # Source must be rescaled with target_rescaler to match inference, where
        # DualFlowPosteriorEstimator.sample() rescales source before the ODE.
        target_scaled, cond_scaled = self._auto_rescale(target, cond)
        source_scaled = self.target_rescaler.transform(source)

        # Sample time if not provided
        if t is None:
            t = self.time_prior.sample(torch.Size((target.shape[0],))).to(target.device)

        # Interpolate using rescaled source
        xt = self.path.interpolate(t, source_scaled, target_scaled)

        # Compute velocities
        v_pred = self.forward(xt, cond_scaled, t)
        v_target = self.path.target_velocity(source_scaled, target_scaled)

        # MSE loss + optional velocity regularization
        loss = (v_pred - v_target).pow(2).mean()
        if velocity_reg > 0:
            loss = loss + velocity_reg * v_pred.pow(2).mean()
        return loss

    def compute_loss_weighted(
        self,
        target: Tensor,
        cond: Tensor,
        t: Optional[Tensor] = None,
        epsilon: float = 0.01,
        boundary_weight: float = 1.0,
    ) -> Tensor:
        """Compute flow matching loss with time-dependent weighting.

        Upweights samples near t=0 and t=1 boundaries where the velocity
        field is often more difficult to learn accurately. The weighting
        scheme uses: w(t) = 1 / (t * (1-t) + epsilon)

        This emphasizes the boundary regions where:
        - Near t=0: The model must correctly map from the base distribution
        - Near t=1: The model must accurately reach the target distribution

        Args:
            target: Target samples (x₁), shape (batch, ...)
            cond: Conditioning variables (y), shape (batch, ...)
            t: Time steps. If None, sampled from prior
            epsilon: Small constant to prevent division by zero (default: 0.01)
            boundary_weight: Scale factor for boundary weighting (default: 1.0)

        Returns:
            Scalar weighted loss
        """
        # Automatic rescaling
        target_scaled, cond_scaled = self._auto_rescale(target, cond)

        # Sample time if not provided
        if t is None:
            t = self.time_prior.sample(torch.Size((target.shape[0],))).to(target.device)

        # Sample source
        source = self.base_dist.sample(cond_scaled)

        # Interpolate
        xt = self.path.interpolate(t, source, target_scaled)

        # Compute velocities
        v_pred = self.forward(xt, cond_scaled, t)
        v_target = self.path.target_velocity(source, target_scaled)

        # Compute per-sample squared errors
        sq_errors = (v_pred - v_target).pow(2).sum(dim=-1)  # (batch,)

        # Compute boundary-emphasizing weights: higher near t=0 and t=1
        weights = boundary_weight / (t.view(-1) * (1 - t.view(-1)) + epsilon)

        # Normalize weights to have mean 1 (preserves loss scale)
        weights = weights / weights.mean()

        # Weighted mean
        return (weights * sq_errors).mean()

    def sample(
        self,
        x0: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: Optional[int] = None,
        only_last: bool = True,
        disable_tqdm: bool = False,
        return_nfe: bool = False,
    ) -> Tensor | Tuple[Tensor, int]:
        """Sample from x₀ to x₁ via ODE integration (Euler method).

        Args:
            x0: Initial samples, shape (batch, ...)
            cond: Conditioning variables, shape (batch, ...)
            device: Device for computation
            num_steps: Integration steps (default: self.num_steps)
            only_last: If True, return only final samples
            disable_tqdm: If True, disable progress bar (currently unused)
            return_nfe: If True, return (samples, num_function_evals)

        Returns:
            If only_last=True: final samples, shape (batch, ...)
            If only_last=False: trajectory, shape (batch, num_steps+1, ...)
            If return_nfe=True: (samples, nfe) tuple
        """
        with torch.no_grad():
            # Only rescale conditioning, NOT x0 (base samples are already in latent space)
            # During training: source from base_dist has std~1, target_scaled has std~1
            # During sampling: x0 from sample_base has std~1, so no rescaling needed
            cond_scaled = self.cond_rescaler.transform(cond)
            x0_scaled = x0  # Base samples are already in correct space

            steps = num_steps or self.num_steps

            samples = self._integrate_ode(
                x0_scaled, cond_scaled, device, steps, only_last
            )
            nfe = steps  # 1 function evaluation per step

            # Inverse rescale output
            if only_last:
                samples = self.target_rescaler.inverse_transform(samples)
            else:
                # Trajectory: rescale each timestep
                samples = torch.stack(
                    [self.target_rescaler.inverse_transform(s) for s in samples], dim=1
                )

            if return_nfe:
                return samples, nfe
            return samples

    def sample_base(self, cond: Tensor, nsamples: Optional[int] = None) -> Tensor:
        """Sample from the base distribution p(x₀|cond).

        Args:
            cond: Conditioning variables, shape (batch, ...)
            nsamples: Number of samples per condition (if None, one per condition)

        Returns:
            Samples from base distribution
        """
        cond_scaled = self.cond_rescaler.transform(cond)
        return self.base_dist.sample(cond_scaled, nsamples)

    def reverse_ode(
        self,
        target: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: Optional[int] = None,
        exact_trace: bool = True,
        n_hutchinson_probes: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run backward ODE from t=1 to t=0 with divergence tracking.

        Solves the reverse ODE while accumulating the log-determinant of the
        Jacobian via the instantaneous change of variables formula.

        Args:
            target: Data points (batch, *dim) -- values in data space
            cond: Conditioning (batch, *cond_dim) -- observations y
            device: torch device
            num_steps: Number of ODE steps (default: self.num_steps)
            exact_trace: If True, compute exact divergence via autograd.
                         If False, use Hutchinson's stochastic trace estimator.
            n_hutchinson_probes: Number of random probes for Hutchinson estimator.

        Returns:
            (x0, cond_scaled, log_det):
                x0: State at t=0 in rescaled space (batch, dim)
                cond_scaled: Rescaled conditioning (batch, cond_dim)
                log_det: Accumulated log-determinant (batch,)
        """
        steps = num_steps or self.num_steps
        dt = 1.0 / steps

        # Rescale inputs
        target_scaled, cond_scaled = self._auto_rescale(target, cond)
        target_scaled = target_scaled.to(device)
        cond_scaled = cond_scaled.to(device)

        # Start at t=1 with x = target_scaled
        xt = target_scaled.clone()
        log_det = torch.zeros(xt.shape[0], device=device)

        # Backward Euler from t=1 to t=0
        for step in range(steps, 0, -1):
            t_val = step * dt
            t = torch.full((xt.shape[0],), t_val, device=device)

            if exact_trace:
                div = self._compute_exact_divergence(xt, cond_scaled, t)
            else:
                div = self._compute_hutchinson_divergence(
                    xt, cond_scaled, t, n_hutchinson_probes
                )

            # Compute velocity (no grad needed for the Euler step itself)
            with torch.no_grad():
                v = self.forward(xt, cond_scaled, t)

            # Backward Euler step: x_{t-dt} = x_t - v * dt
            xt = xt - v * dt
            log_det = log_det - div * dt

        return xt, cond_scaled, log_det

    def log_prob(
        self,
        target: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: Optional[int] = None,
        exact_trace: bool = True,
        n_hutchinson_probes: int = 1,
    ) -> Tensor:
        """Compute log p(target|cond) via instantaneous change of variables.

        Solves the reverse ODE from t=1 to t=0 while tracking the divergence
        of the velocity field to accumulate the log-determinant correction.

        log p(x_1|y) = log p_0(x_0|y) - int_0^1 div_x(v(x_t, t, y)) dt

        Args:
            target: Data points (batch, *dim) -- theta values in data space
            cond: Conditioning (batch, *cond_dim) -- observations y
            device: torch device
            num_steps: Number of ODE steps (default: self.num_steps)
            exact_trace: If True, compute exact divergence via autograd.
                         If False, use Hutchinson's stochastic trace estimator.
            n_hutchinson_probes: Number of random probes for Hutchinson estimator.

        Returns:
            log_prob: (batch,) tensor of log-probabilities
        """
        xt, cond_scaled, log_det = self.reverse_ode(
            target, cond, device, num_steps, exact_trace, n_hutchinson_probes
        )

        # Compute base distribution log probability at t=0
        if isinstance(self.base_dist, _DataEpsBase):
            base_log_prob = self.base_dist.log_prob(xt, cond_scaled)
        else:
            base_log_prob = self.base_dist.log_prob(xt)

        return base_log_prob + log_det

    def _compute_exact_divergence(
        self, x: Tensor, cond_scaled: Tensor, t: Tensor
    ) -> Tensor:
        """Compute exact divergence div_x(v(x, t, cond)) via autograd.

        Computes sum_i dv_i/dx_i by looping over output dimensions.

        Args:
            x: Current state, shape (batch, dim)
            cond_scaled: Rescaled conditioning, shape (batch, cond_dim)
            t: Time, shape (batch,)

        Returns:
            Divergence, shape (batch,)
        """
        x = x.detach().requires_grad_(True)
        v = self.forward(x, cond_scaled, t)

        div = torch.zeros(x.shape[0], device=x.device)
        for i in range(x.shape[-1]):
            grad_i = torch.autograd.grad(
                v[..., i].sum(), x, create_graph=False, retain_graph=(i < x.shape[-1] - 1)
            )[0]
            div = div + grad_i[..., i]

        return div.detach()

    def _compute_hutchinson_divergence(
        self, x: Tensor, cond_scaled: Tensor, t: Tensor, n_probes: int = 1
    ) -> Tensor:
        """Compute stochastic divergence estimate via Hutchinson's trace estimator.

        E[eps^T (dv/dx) eps] = tr(dv/dx) for eps ~ N(0, I) or Rademacher.

        Args:
            x: Current state, shape (batch, dim)
            cond_scaled: Rescaled conditioning, shape (batch, cond_dim)
            t: Time, shape (batch,)
            n_probes: Number of random probes to average over.

        Returns:
            Divergence estimate, shape (batch,)
        """
        div_sum = torch.zeros(x.shape[0], device=x.device)

        for _ in range(n_probes):
            x_in = x.detach().requires_grad_(True)
            v = self.forward(x_in, cond_scaled, t)

            # Rademacher random vectors
            eps = torch.randint(0, 2, x_in.shape, device=x.device).float() * 2 - 1

            # eps^T @ (dv/dx) @ eps = eps^T @ d(v^T eps)/dx
            v_eps = (v * eps).sum()
            grad_v_eps = torch.autograd.grad(v_eps, x_in, create_graph=False)[0]
            div_sum = div_sum + (grad_v_eps * eps).sum(dim=-1).detach()

        return div_sum / n_probes

    def rescale(self, target: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Rescale target and conditioning variables.

        Args:
            target: Target samples to rescale
            cond: Conditioning variables to rescale

        Returns:
            Tuple of (target_scaled, cond_scaled)
        """
        return self._auto_rescale(target, cond)

    def set_rescalers(self, target_rescaler, cond_rescaler):
        """Set rescalers for target and conditioning variables.

        Args:
            target_rescaler: DataRescaler for flow target (theta or x)
            cond_rescaler: DataRescaler for conditioning variables (y)
        """
        from utils.rescaling import DataRescaler

        if not isinstance(target_rescaler, DataRescaler):
            raise TypeError(f"target_rescaler must be DataRescaler, got {type(target_rescaler)}")
        if not isinstance(cond_rescaler, DataRescaler):
            raise TypeError(f"cond_rescaler must be DataRescaler, got {type(cond_rescaler)}")

        self.target_rescaler = target_rescaler
        self.cond_rescaler = cond_rescaler

    def set_scales(self, data: Tensor, cond: Tensor, rescale_name: str):
        """Create and fit rescalers for target and conditioning variables.

        Args:
            data: Training data (flow target: theta or x)
            cond: Conditioning variables (y)
            rescale_name: Type of rescaling ('none', 'z_score', 'whiten')
        """
        from utils.rescaling import create_rescaler

        target_rescaler = create_rescaler(rescale_name)
        cond_rescaler = create_rescaler(rescale_name)

        target_rescaler.fit(data)
        cond_rescaler.fit(cond)

        self.set_rescalers(target_rescaler, cond_rescaler)

        # Update config
        self._config.rescale_mode = rescale_name

    # =========================================================================
    # Serialization Methods (Clean API)
    # =========================================================================

    def get_config(self) -> FlowMatchingConfig:
        """Get model configuration for serialization."""
        return self._config

    @classmethod
    def from_config(cls, config: FlowMatchingConfig) -> "FlowMatching":
        """Create FlowMatching from configuration.

        Args:
            config: FlowMatchingConfig instance

        Returns:
            New FlowMatching with architecture matching config
        """
        return cls(
            probability_path=config.probability_path,
            prior=config.prior,
            base_dist=config.base_dist,
            dim=torch.Size(config.dim),
            cond_dim=torch.Size(config.cond_dim),
            num_steps=config.num_steps,
            probability_path_params=config.probability_path_params,
            prior_params=config.prior_params,
            base_dist_params=config.base_dist_params,
            drift=config.drift,
            rescale_mode=config.rescale_mode,
        )

    def save(self, path: Path) -> None:
        """Save model to directory.

        Creates:
            path/config.json - Model configuration
            path/weights.pth - Model weights

        Args:
            path: Directory to save model to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._config.save(path / "config.json")
        torch.save(self.state_dict(), path / "weights.pth")

    @classmethod
    def load(cls, path: Path, device: torch.device = torch.device("cpu")) -> "FlowMatching":
        """Load model from directory.

        Args:
            path: Directory containing config.json and weights.pth
            device: Device to load model to

        Returns:
            Loaded FlowMatching
        """
        path = Path(path)

        config = FlowMatchingConfig.load(path / "config.json")
        model = cls.from_config(config)

        state_dict = torch.load(path / "weights.pth", map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)

        return model

    def state_dict(self, *args, **kwargs):
        """Get model state dict.

        Note: For new code, prefer using save()/load() methods.
        """
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict.

        Note: For new code, prefer using save()/load() methods which handle
        both config and weights properly.
        """
        super().load_state_dict(state_dict, strict=strict)

    # ==========================================================================
    # Internal Helper Methods
    # ==========================================================================

    def _auto_rescale(self, target: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rescaling transformations automatically.

        Args:
            target: Target data
            cond: Conditioning data

        Returns:
            (rescaled_target, rescaled_cond)
        """
        target = self.target_rescaler.transform(target)
        cond = self.cond_rescaler.transform(cond)
        return target, cond

    def _integrate_ode(
        self,
        x0: Tensor,
        cond: Tensor,
        device: torch.device,
        num_steps: int,
        only_last: bool,
    ) -> Tensor:
        """Integrate ODE using Euler method.

        Solves: dx/dt = v_θ(x, t, cond) from t=0 to t=1

        Args:
            x0: Initial state (already rescaled)
            cond: Conditioning (already rescaled)
            device: Device for computation
            num_steps: Number of Euler steps
            only_last: Return only final state vs full trajectory

        Returns:
            If only_last: x(t=1), shape (batch, ...)
            Else: List of [x(t=0), ..., x(t=1)]
        """
        dt = 1.0 / num_steps
        xt = x0.to(device)
        cond = cond.to(device)

        if not only_last:
            trajectory = [xt.cpu()]

        for step in range(num_steps):
            t = torch.full((xt.shape[0],), step * dt, device=device)
            v = self.forward(xt, cond, t)
            xt = xt + v * dt

            if not only_last:
                trajectory.append(xt.cpu())

        if only_last:
            return xt
        return trajectory

    @staticmethod
    def _validate_config(
        probability_path: str, prior: str, base_dist: str, **kwargs
    ) -> None:
        """Validate configuration parameters.

        Args:
            probability_path: Probability path type
            prior: Time prior type
            base_dist: Base distribution type
            **kwargs: Additional parameters

        Raises:
            ValueError: If any parameter is invalid
        """
        if probability_path not in ["ot", "ot2"]:
            raise ValueError(
                f"Invalid probability_path: {probability_path}. Must be 'ot' or 'ot2'."
            )

        if prior not in ["uniform", "power"]:
            raise ValueError(f"Invalid prior: {prior}. Must be 'uniform' or 'power'.")

        if base_dist not in ["gaussian", "data_eps"]:
            raise ValueError(
                f"Invalid base_dist: {base_dist}. Must be 'gaussian' or 'data_eps'."
            )

        drift_config = kwargs.get("drift", {})
        architecture = drift_config.get("architecture")
        if architecture and architecture not in ["resmlp", "cfnet"]:
            raise ValueError(
                f"Invalid drift architecture: {architecture}. Must be 'resmlp' or 'cfnet'."
            )

    @staticmethod
    def _create_path(probability_path: str, **params) -> _ProbabilityPath:
        """Create probability path instance.

        Args:
            probability_path: Path type ("ot" or "ot2")
            **params: Path-specific parameters

        Returns:
            Probability path instance
        """
        if probability_path == "ot":
            return _OptimalTransportPath()
        elif probability_path == "ot2":
            sigma_min = params.get("sigma_min", 1e-4)
            return _OT2Path(sigma_min=sigma_min)
        else:
            raise ValueError(f"Unknown probability path: {probability_path}")

    @staticmethod
    def _create_time_prior(prior: str, **params) -> Distribution:
        """Create time prior distribution.

        Args:
            prior: Prior type ("uniform" or "power")
            **params: Prior-specific parameters

        Returns:
            PyTorch distribution for time sampling
        """
        if prior == "uniform":
            return Uniform(0.0, 1.0)
        elif prior == "power":
            rate = params.get("rate", 1.5)
            return Beta(rate, 1.0)
        else:
            raise ValueError(f"Unknown time prior: {prior}")

    @staticmethod
    def _create_base_dist(base_dist: str, dim: torch.Size, **params) -> _BaseDistribution:
        """Create base distribution instance.

        Args:
            base_dist: Distribution type ("gaussian" or "data_eps")
            dim: Dimension of target space
            **params: Distribution-specific parameters

        Returns:
            Base distribution instance
        """
        if base_dist == "gaussian":
            return _GaussianBase(dim)
        elif base_dist == "data_eps":
            eps = params.get("eps", 0.01)
            return _DataEpsBase(dim, eps=eps)
        else:
            raise ValueError(f"Unknown base distribution: {base_dist}")

    @staticmethod
    def _build_drift(dim: torch.Size, cond_dim: torch.Size, **kwargs) -> nn.Module:
        """Build drift model (velocity field network).

        Args:
            dim: Target dimension
            cond_dim: Conditioning dimension
            **kwargs: Drift model configuration

        Returns:
            Drift model as nn.Module
        """
        architecture = kwargs.get("architecture", "resmlp")

        if architecture == "resmlp":
            hidden_features = kwargs.get("hidden_dim", (64, 64))
            mlp_kwargs = kwargs.get("mlp_params", {"activation": nn.ELU})
            if len(dim) != 1 or len(cond_dim) != 1:
                raise ValueError("ResMLP requires 1-dimensional target and condition.")
            return ResMLP(dim[0] + cond_dim[0] + 1, dim[0], hidden_features, **mlp_kwargs)

        elif architecture == "cfnet":
            return create_cf(
                kwargs["posterior_kwargs"],
                kwargs.get("embedding_kwargs", {}),
                kwargs.get("theta_embedding_kwargs", {}),
            )

        else:
            raise ValueError(f"Invalid drift model architecture: {architecture}")

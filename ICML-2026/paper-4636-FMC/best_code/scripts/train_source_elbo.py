#!/usr/bin/env python3
"""Train source flow T_ψ(ε,y)→x using ELBO loss with normalizing flows.

This script implements a two-stage training pipeline:
1. Train NPE p(θ|x) on simulation data
2. Train source flow T_ψ: (ε,y) → x using custom ELBO loss with reparameterization

The ELBO loss is: E_{θ,y,ε~N(0,I)}[log p_NPE(θ|x=T_ψ(ε,y)) + log p(ε)]

The source flow T_ψ is a conditional normalizing flow that maps noise ε ~ N(0,I)
to observations x conditioned on y. We use the reparameterization trick:
- Sample ε ~ N(0,I)
- Generate x = T_ψ(ε, y) via the flow
- Evaluate log p(θ|x) + log p(ε)

Author: Generated with Claude Code
Date: 2026-01-05
"""

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from typing import Callable
from zuko.distributions import DiagNormal
from zuko.flows import Flow, NSF, UnconditionalDistribution

# Lampe imports for NPE
from lampe.inference import NPE, NPELoss
from lampe.utils import GDStep

# Local imports
from posteriors import Estimator, get_build_fn
from baselines.trainers import train_npe, train_flow_matching
from flow_matching.torch_flow import FlowMatching
from mlxp.logger import Logger
from simulator import generate_calibration_dataset, get_simulator
from simulator.base import Simulator
from simulator.pure_gaussian import PureGaussian
from utils.metrics import MMD, classifier_two_samples_test
from utils.misc import train_val_split as utils_train_val_split
from utils.rescaling import ZScoreRescaler, NoOpRescaler


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SimulatorConfig:
    """Configuration for simulator."""

    name: str = "pure_gaussian"
    theta_dim: int = 2
    obs_dim: int = 3
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "prior_var_scale": 2.0,
            "likelihood_var_scale": 1.0,
            "noisy_var_scale": 0.1,
            "seed": 0,
        }
    )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Dimensions (inferred from data)
    theta_dim: int = 2
    obs_dim: int = 3

    # Stage 1: NPE
    npe_epochs: int = 2000
    npe_lr: float = 1e-4
    npe_batch_size: int = 512
    npe_transforms: int = 5
    npe_hidden_features: List[int] = field(default_factory=lambda: [64, 64])

    # Stage 2: Source Flow
    source_epochs: int = 100
    source_lr: float = 1e-4
    source_batch_size: int = 256
    source_transforms: int = 5
    source_hidden_features: List[int] = field(default_factory=lambda: [64, 64])

    # Training
    max_patience: int = 20
    train_size: float = 0.8
    gradient_clip: float = 5.0

    # Learning rate scheduling
    warmup_epochs: int = 10
    min_lr_ratio: float = 0.01  # min_lr = source_lr * min_lr_ratio

    # Evaluation
    num_eval_samples: int = 1000
    num_viz_obs: int = 4

    # Paths
    model_path: Path = Path("models/elbo_source")
    log_path: Path = Path("logs/elbo_source")

    # Device and seed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ============================================================================
# Config Conversion Utilities
# ============================================================================


def config_to_npe_dict(config: TrainingConfig, task_name: str = "generic") -> Dict:
    """Convert TrainingConfig to task dict for train_npe baseline function.

    Args:
        config: Training configuration
        task_name: Name of the task (for Estimator initialization)

    Returns:
        Task dictionary with NPE training and params nested structure
    """
    return {
        "name": task_name,  # Required by Estimator constructor
        "npe": {
            "training": {
                "epochs": config.npe_epochs,
                "lr": config.npe_lr,
                "batch_size": config.npe_batch_size,
                "train_size": config.train_size,
                "max_patience": config.max_patience,
                "rescale": "none",  # No rescaling
            },
            "params": {
                "transforms": config.npe_transforms,
                "hidden_features": config.npe_hidden_features,
            },
        },
    }


def config_to_fm_dict(conditional: bool = True, hidden_features: List[int] = None) -> Dict:
    """Create FlowMatching config dict for train_flow_matching baseline function.

    Args:
        conditional: Whether the flow is conditional (should be True for source learning)
        hidden_features: Hidden layer dimensions for the network

    Returns:
        FlowMatching configuration dictionary
    """
    if hidden_features is None:
        hidden_features = [128, 128]

    return {
        "conditional": conditional,
        "probability_path": "ot",  # Optimal transport
        "prior": "uniform",
        "base_dist": "gaussian",
        "params": {
            "drift": {
                "architecture": "resmlp",
                "hidden_dims": hidden_features + [hidden_features[-1]],
                "time_embedding": "sinusoidal",
                "time_embedding_dim": 64,
            }
        },
    }


# ============================================================================
# Source Flow (Conditional Normalizing Flow)
# ============================================================================


class SourceFlow(nn.Module):
    """Conditional normalizing flow for learning p(x|y).

    This learns the transformation T_ψ: (ε, y) → x where ε ~ N(0,I).
    Using a normalizing flow allows direct computation of log p(x|y) via
    the change of variables formula.

    Attributes:
        flow: NSF (Neural Spline Flow) conditional normalizing flow
        x_dim: Dimension of x (observation space)
        y_dim: Dimension of y (noisy observation space)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        transforms: int = 5,
        hidden_features: List[int] = None,
    ):
        """Initialize SourceFlow.

        Args:
            x_dim: Dimension of x (clean observations)
            y_dim: Dimension of y (noisy observations)
            transforms: Number of coupling transforms
            hidden_features: Hidden layer sizes for conditioner networks
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        if hidden_features is None:
            hidden_features = [64, 64]

        # Create NSF (Neural Spline Flow)
        # This is a conditional flow: samples x given context y
        self.flow = NSF(
            features=x_dim,
            context=y_dim,
            transforms=transforms,
            hidden_features=hidden_features,
        )

    def log_prob(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute log p(x|y).

        Args:
            x: Observations (batch_size, x_dim)
            y: Conditions (batch_size, y_dim)

        Returns:
            log_prob: (batch_size,)
        """
        # Get conditional flow given y
        conditional_flow = self.flow(y)

        # Compute log probability
        return conditional_flow.log_prob(x)

    def sample(self, y: Tensor, num_samples: int = 1, eps: Optional[Tensor] = None) -> Tensor:
        """Sample x given y, optionally using provided noise eps.

        Args:
            y: Conditions (batch_size, y_dim)
            num_samples: Number of samples per condition (ignored if eps provided)
            eps: Optional noise (batch_size, x_dim) or (num_samples, batch_size, x_dim)

        Returns:
            samples: (num_samples, batch_size, x_dim) or (batch_size, x_dim) if eps provided
        """
        conditional_flow = self.flow(y)

        if eps is not None:
            # Use provided noise: transform eps -> x
            return conditional_flow.transform(eps)
        else:
            # Sample noise internally
            return conditional_flow.sample((num_samples,))


# ============================================================================
# Unified ELBO Loss
# ============================================================================


class UnifiedELBOLoss(nn.Module):
    """Unified ELBO loss supporting both learned and true posteriors.

    Computes: -E_{θ,y,ε~N(0,I)}[log p(θ|x=T_ψ(ε,y)) + log p(ε)]

    The source flow T_ψ maps noise ε ~ N(0,I) to observations x conditioned on y.
    We use the reparameterization trick: sample ε, generate x = T_ψ(ε,y),
    then evaluate the ELBO.

    Args:
        source_flow: SourceFlow model to train
        posterior_fn: Either NPE estimator or simulator.posterior_theta_given_x
        use_true_posterior: If True, posterior_fn is analytical posterior
    """

    def __init__(
        self, source_flow: SourceFlow, posterior_fn: Callable, use_true_posterior: bool = False
    ):
        """Initialize unified ELBO loss.

        Args:
            source_flow: SourceFlow to train
            posterior_fn: Either NPE estimator or simulator.posterior_theta_given_x
            use_true_posterior: Whether posterior_fn is analytical (True) or learned NPE (False)
        """
        super().__init__()
        self.source_flow = source_flow
        self.posterior_fn = posterior_fn
        self.use_true_posterior = use_true_posterior

        # If using learned posterior (NPE), freeze it
        if not use_true_posterior:
            self.posterior_fn.eval()
            for param in self.posterior_fn.parameters():
                param.requires_grad = False

    def __call__(
        self,
        theta: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute ELBO loss using reparameterization.

        Args:
            theta: Parameters (batch_size, theta_dim)
            y: Noisy observations (batch_size, y_dim)

        Returns:
            loss: Negative ELBO (scalar)
            metrics: Dict with loss components
        """
        batch_size = theta.shape[0]
        device = theta.device
        x_dim = self.source_flow.x_dim

        # 1. Sample ε ~ N(0,I) with shape matching x
        eps = torch.randn(batch_size, x_dim, device=device)

        # 2. Generate x = T_ψ(ε, y) using source flow
        x_generated = self.source_flow.sample(y, eps=eps)

        # 3. Compute log p(θ|x_generated)
        if self.use_true_posterior:
            # Use analytical posterior: returns distribution
            posterior_dist = self.posterior_fn(x_generated.cpu())
            log_p_theta_given_x = posterior_dist.log_prob(theta.cpu()).to(device)
        else:
            # Use learned NPE posterior
            log_p_theta_given_x = self.posterior_fn.log_prob(theta, x_generated)

        # 4. Compute log p(ε) for Gaussian prior
        log_p_eps = -0.5 * eps.pow(2).sum(-1) - 0.5 * x_dim * np.log(2 * np.pi)

        # 5. ELBO = E[log p(θ|x) + log p(ε)]
        elbo_per_sample = log_p_theta_given_x
        elbo = elbo_per_sample.mean()

        # Loss is negative ELBO
        loss = -elbo

        # Metrics
        metrics = {
            "elbo": elbo.item(),
            "log_p_theta_x": log_p_theta_given_x.mean().item(),
            "log_p_eps": log_p_eps.mean().item(),
        }

        return loss, metrics


# ============================================================================
# Data Utilities
# ============================================================================


def train_val_split(
    *tensors: Tensor,
    train_size: float = 0.8,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Split data into train/val loaders.

    Args:
        *tensors: Data tensors to split
        train_size: Fraction for training
        batch_size: Batch size

    Returns:
        train_loader, val_loader
    """
    n_total = tensors[0].shape[0]
    n_train = int(n_total * train_size)

    # Split
    train_tensors = [t[:n_train] for t in tensors]
    val_tensors = [t[n_train:] for t in tensors]

    # Create datasets
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# ============================================================================
# Unified Training Function
# ============================================================================


def train_source_elbo(
    theta: Tensor,
    y: Tensor,
    posterior_fn: Callable,
    use_true_posterior: bool,
    config: TrainingConfig,
    device: torch.device,
    logger: Optional[Logger] = None,
    load: bool = False,
    model_path: Optional[Path] = None,
    logname: str = "",
) -> SourceFlow:
    """Train source flow with ELBO loss (unified function for both NPE and true posterior).

    Args:
        theta: Parameters (N, theta_dim)
        y: Noisy observations (N, obs_dim)
        posterior_fn: Either NPE estimator or simulator.posterior_theta_given_x
        use_true_posterior: If True, posterior_fn is analytical posterior
        config: Training configuration
        device: Device to train on
        logger: Optional logger
        load: If True, try to load model from disk before training
        model_path: Path to save/load model
        logname: Name for saving/loading model file

    Returns:
        Trained SourceFlow model
    """
    method_name = "True Posterior" if use_true_posterior else "NPE-based"
    print("\n" + "=" * 80)
    print(f"TRAINING Source Flow with {method_name} ELBO")
    print("=" * 80)

    # Initialize source flow (no rescaling)
    source_flow = SourceFlow(
        x_dim=config.obs_dim,
        y_dim=config.obs_dim,
        transforms=config.source_transforms,
        hidden_features=config.source_hidden_features,
    ).to(device)

    # Try to load existing model if requested
    if load and model_path is not None and logname:
        model_file = model_path / f"{logname}.pth"
        if model_file.exists():
            print(f"✓ Loading existing model from {model_file}")
            try:
                source_flow.load_state_dict(
                    torch.load(model_file, weights_only=True, map_location=device)
                )
                print(f"✓ Successfully loaded {method_name} source flow")
                return source_flow
            except Exception as e:
                print(f"⚠ Failed to load model: {e}")
                print("  Training from scratch...")
        else:
            print(f"  Model file not found at {model_file}, training from scratch...")

    # Create unified ELBO loss
    elbo_loss = UnifiedELBOLoss(
        source_flow=source_flow, posterior_fn=posterior_fn, use_true_posterior=use_true_posterior
    )

    # Training setup
    optimizer = torch.optim.Adam(source_flow.parameters(), lr=config.source_lr)

    # Learning rate scheduler: linear warmup + cosine decay
    warmup_epochs = config.warmup_epochs
    total_epochs = config.source_epochs
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(epoch: int) -> float:
        """Compute LR multiplier: linear warmup then cosine decay."""
        if epoch < warmup_epochs:
            # Linear warmup from 0 to 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay from 1 to min_lr_ratio
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader, val_loader = train_val_split(
        theta,
        y,
        train_size=config.train_size,
        batch_size=config.source_batch_size,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    pbar = trange(config.source_epochs, desc=f"Training Source Flow ({method_name})")
    for epoch in pbar:
        # Training phase
        source_flow.train()
        train_losses = []
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss, metrics = elbo_loss(theta_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(source_flow.parameters(), config.gradient_clip)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        source_flow.eval()
        val_losses = []
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                loss, metrics = elbo_loss(theta_batch, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_elbo = -val_loss

        # Step the scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Update progress bar
        pbar.set_postfix(
            {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "elbo": f"{val_elbo:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = source_flow.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    if best_state_dict is not None:
        source_flow.load_state_dict(best_state_dict)

    print(f"✓ Source flow ({method_name}) training complete. Best val loss: {best_val_loss:.4f}")
    return source_flow


# ============================================================================
# Unified Evaluation Functions
# ============================================================================


def generate_samples_for_evaluation(
    source_model,  # SourceFlow or FlowMatching
    posterior_fn,  # Estimator NPE or simulator.posterior_theta_given_x
    y_test: Tensor,
    use_true_posterior: bool,
    num_samples: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Generate theta samples via π(θ|y) = ∫ p(θ|x) q(x|y) dx.

    Args:
        source_model: Trained source flow or flow matching model
        posterior_fn: Estimator with .sample() or simulator.posterior_theta_given_x
        y_test: Test noisy observations (N_test, obs_dim)
        use_true_posterior: Whether posterior_fn is analytical
        num_samples: Number of samples to generate per observation
        device: Device

    Returns:
        theta_samples: (num_samples, N_test, theta_dim)
    """
    print(f"Sampling {num_samples} θ via π(θ|y) = ∫ p(θ|x) q(x|y) dx...")

    theta_samples_list = []
    x_sample_list = []
    source_model.eval()

    for i in range(num_samples):
        # Sample x ~ q(x|y)
        if isinstance(source_model, SourceFlow):
            # SourceFlow.sample(y, num_samples) returns (batch_size, num_samples, x_dim)
            # We want 1 sample per y, so squeeze
            x_sampled = source_model.sample(y_test.to(device), num_samples=1).squeeze(1)
        else:  # FlowMatching
            # FlowMatching requires manual source sampling
            source = source_model.sample_source(y_test.to(device)).to(device)
            x_sampled = source_model.sample(
                source, y_test.to(device), device, only_last=True, disable_tqdm=True
            )

        # Sample θ ~ p(θ|x)
        if use_true_posterior:
            # Analytical posterior: returns distribution
            posterior_dist = posterior_fn(x_sampled.cpu())
            theta_sampled = posterior_dist.sample()
        else:
            # Learned NPE: use Estimator.sample() method
            # Estimator.sample(x, nsamples, device) returns (nsamples, batch_size, theta_dim)
            theta_sampled = posterior_fn.sample(x_sampled, nsamples=1, device=device).squeeze(0)

        # Ensure shape is (N_test, theta_dim) by removing any extra dimensions
        while len(theta_sampled.shape) > 2:
            theta_sampled = theta_sampled.squeeze(0)

        theta_samples_list.append(theta_sampled.cpu())
        x_sample_list.append(x_sampled.squeeze(0).cpu())

    # Stack to (num_samples, N_test, theta_dim)
    return torch.stack(theta_samples_list, dim=0), torch.stack(x_sample_list, dim=0)


def evaluate_source_quality(
    theta_samples: Tensor,  # Pre-generated samples (num_samples, N_test, theta_dim)
    theta_test: Tensor,  # True parameters (N_test, theta_dim)
    config: TrainingConfig,
) -> Dict[str, float]:
    """Evaluate source quality from pre-generated theta samples.

    Args:
        theta_samples: Samples from π(θ|y) via source flow (num_samples, N_test, theta_dim)
        theta_test: True test parameters (N_test, theta_dim)
        config: Configuration

    Returns:
        Dictionary of metrics
    """
    # Average samples across the num_samples dimension: (num_samples, N_test, theta_dim) -> (N_test, theta_dim)
    theta_samples_mean = theta_samples.mean(dim=0)

    # Ensure theta_samples_mean has correct shape (N_test, theta_dim)
    while len(theta_samples_mean.shape) > 2:
        theta_samples_mean = theta_samples_mean.squeeze(0)

    metrics = {}

    # Compute Wasserstein distance per dimension
    for dim in range(theta_test.shape[1]):
        # Ensure arrays are 1D
        test_dim = theta_test[:, dim].cpu().numpy().flatten()
        samples_dim = theta_samples_mean[:, dim].cpu().numpy().flatten()

        wd = wasserstein_distance(test_dim, samples_dim)
        metrics[f"wasserstein_theta_dim_{dim}"] = float(wd)

    # Compute MMD
    device = torch.device(config.device if hasattr(config, "device") else "cpu")
    mmd = MMD(theta_test.to(device), theta_samples_mean.to(device), kernel="rbf")
    metrics["mmd_theta"] = float(mmd.item())

    return metrics


def train_source(
    source_flow: SourceFlow,
    simulator,
    npe: Estimator,
    theta: Tensor,
    x: Tensor,
    y: Tensor,
    config: TrainingConfig,
    use_true_npe=True,
) -> SourceFlow:
    """Train source flow T_ψ using TRUE POSTERIOR ELBO loss.

    Args:
        source_flow: SourceFlow model to train
        simulator: Simulator with analytical posterior
        theta: Parameters (n_samples, theta_dim)
        x: Clean observations (n_samples, x_dim)
        y: Noisy observations (n_samples, y_dim)
        config: Training configuration

    Returns:
        Trained SourceFlow
    """
    print("\n" + "=" * 80)
    print("TRAINING Source Flow with TRUE POSTERIOR ELBO")
    print("=" * 80)

    device = torch.device(config.device)

    # Setup rescaling for source flow
    source_flow.set_scales(x, y, config.rescale)
    source_flow.to(device)

    # Create ELBO loss with true posterior
    if use_true_npe:
        elbo_loss = TruePosteriorELBOLoss(simulator, source_flow)
    else:
        elbo_loss = ELBOLoss(npe, source_flow)

    # Optimizer
    optimizer = torch.optim.Adam(source_flow.parameters(), lr=config.source_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.source_epochs)

    # Data loaders
    train_loader, val_loader = train_val_split(
        theta,
        x,
        y,
        train_size=config.train_size,
        batch_size=config.source_batch_size,
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 0
    best_state = None
    train_losses = []
    val_losses = []
    train_elbos = []
    val_elbos = []

    pbar = trange(config.source_epochs, desc="Training Source Flow (True P)")
    for epoch in pbar:
        # Training
        source_flow.train()
        epoch_train_losses = []
        epoch_train_metrics = defaultdict(list)

        for theta_batch, x_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Rescale x and y for source flow (theta stays unscaled for simulator)
            x_batch_scaled, y_batch_scaled = source_flow.rescale(x_batch, y_batch)

            # Compute ELBO loss
            loss, metrics = elbo_loss(theta_batch, x_batch_scaled, y_batch_scaled)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(source_flow.parameters(), config.gradient_clip)
            optimizer.step()

            epoch_train_losses.append(loss.item())
            for k, v in metrics.items():
                epoch_train_metrics[k].append(v)

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)
        train_elbos.append(np.mean(epoch_train_metrics["elbo"]))
        scheduler.step()

        # Validation
        source_flow.eval()
        epoch_val_losses = []
        epoch_val_metrics = defaultdict(list)

        with torch.no_grad():
            for theta_batch, x_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                x_batch_scaled, y_batch_scaled = source_flow.rescale(x_batch, y_batch)
                loss, metrics = elbo_loss(theta_batch, x_batch_scaled, y_batch_scaled)

                epoch_val_losses.append(loss.item())
                for k, v in metrics.items():
                    epoch_val_metrics[k].append(v)

        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)
        val_elbos.append(np.mean(epoch_val_metrics["elbo"]))

        # Update progress bar
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_elbo": f"{val_elbos[-1]:.4f}",
            }
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = deepcopy(source_flow.state_dict())
        else:
            patience += 1

        if patience >= config.max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        source_flow.load_state_dict(best_state, strict=False)

    print(f"\n✓ Source flow (TRUE P) training complete. Best val loss: {best_val_loss:.4f}")

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Source Flow Training Loss (True Posterior)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ELBO curves
    ax = axes[1]
    ax.plot(train_elbos, label="Train ELBO")
    ax.plot(val_elbos, label="Val ELBO")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO (higher is better)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.log_path / f"source_training_curves_{'true' if use_true_npe else 'npe'}.pdf")
    plt.close()

    return source_flow


# ============================================================================
# Stage 2c: Train Source Flow with Flow Matching (Standard Loss)
# ============================================================================


def train_source_with_flow_matching(
    x: Tensor,
    y: Tensor,
    config: TrainingConfig,
) -> FlowMatching:
    """Train source flow p(x|y) using standard FLOW MATCHING loss.

    This trains q_ψ(x|y) to match p(x|y) using only the (x,y) data pairs
    with standard flow matching loss. No posterior p(θ|x) is used during training.

    This is comparable to the fm_post_transform_only_ut approach in the main codebase.

    Args:
        x: Clean observations (n_samples, x_dim)
        y: Noisy observations (n_samples, y_dim)
        config: Training configuration

    Returns:
        Trained FlowMatching model
    """
    print("\n" + "=" * 80)
    print("STAGE 2c: Training Source Flow with FLOW MATCHING")
    print("=" * 80)
    print("This variant uses standard flow matching loss on (x,y) pairs.")
    print("No posterior is used during training - only the data distribution p(x|y).")

    device = torch.device(config.device)

    # Create FlowMatching model for p(x|y)
    flow_config = {
        "conditional": True,
        "probability_path": "ot2",  # Optimal transport
        "prior": "uniform",
        "base_dist": "gaussian",
        "params": {
            "drift": {
                "architecture": "resmlp",
                "hidden_dims": config.source_hidden_features + [config.source_hidden_features[-1]],
                "time_embedding": "sinusoidal",
                "time_embedding_dim": 64,
            }
        },
    }

    flow_matching_x = FlowMatching(
        flow_config["conditional"],
        flow_config["probability_path"],
        flow_config["prior"],
        flow_config["base_dist"],
        x.shape[1:],
        cond_dim=y.shape[1:],
        **flow_config["params"],
    )

    flow_matching_x.to(device)

    # Set rescaling
    flow_matching_x.set_scales(x, y, config.rescale)

    # Optimizer
    optimizer = torch.optim.Adam(flow_matching_x.parameters(), lr=config.source_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.source_epochs)

    # Prepare data
    train_loader, val_loader = train_val_split(
        x,
        y,
        train_size=config.train_size,
        batch_size=config.source_batch_size,
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 0
    best_state = None

    train_losses = []
    val_losses = []

    print(f"Training for up to {config.source_epochs} epochs...")
    print(f"Batch size: {config.source_batch_size}, LR: {config.source_lr}")
    print(f"Rescaling: {config.rescale}")

    for epoch in trange(config.source_epochs, desc="Training FM Source"):
        flow_matching_x.train()

        # Training
        epoch_loss = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Rescale
            x_batch_scaled, y_batch_scaled = flow_matching_x.rescale(x_batch, y_batch)

            # Sample time
            time = flow_matching_x.sample_time(x_batch.shape[0]).to(device)

            # Sample source (noise)
            source = flow_matching_x.sample_source(y_batch_scaled).to(device)

            # Interpolant
            z_t = flow_matching_x.interpolant(time, source, x_batch_scaled)

            # Target velocity
            v = flow_matching_x.target(source, x_batch_scaled)

            # Predicted velocity
            pred = flow_matching_x(z_t, y_batch_scaled, time)

            # Flow matching loss
            loss = (pred - v).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_matching_x.parameters(), config.gradient_clip)
            optimizer.step()

            epoch_loss.append(loss.item())

        train_loss = np.mean(epoch_loss)
        train_losses.append(train_loss)
        scheduler.step()

        # Validation
        flow_matching_x.eval()
        epoch_val_loss = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                x_batch_scaled, y_batch_scaled = flow_matching_x.rescale(x_batch, y_batch)
                time = flow_matching_x.sample_time(x_batch.shape[0]).to(device)
                source = flow_matching_x.sample_source(y_batch_scaled).to(device)
                z_t = flow_matching_x.interpolant(time, source, x_batch_scaled)
                v = flow_matching_x.target(source, x_batch_scaled)
                pred = flow_matching_x(z_t, y_batch_scaled, time)
                loss = (pred - v).pow(2).mean()

                epoch_val_loss.append(loss.item())

        val_loss = np.mean(epoch_val_loss)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = deepcopy(flow_matching_x.state_dict())
        else:
            patience += 1

        if patience >= config.max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        flow_matching_x.load_state_dict(best_state)

    print(f"\n✓ Flow matching source training complete. Best val loss: {best_val_loss:.4f}")

    # Save training curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow Matching Loss")
    ax.set_title("Source Flow Training (Flow Matching)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.log_path / "source_training_curves_fm.pdf")
    plt.close()

    return flow_matching_x


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_source_quality_old(
    source_flow: SourceFlow,
    npe: Estimator,
    simulator,
    theta_test: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    config: TrainingConfig,
) -> Dict[str, float]:
    """[OLD VERSION - NOT USED] Evaluate quality of learned source flow.

    Args:
        source_flow: Trained source flow
        npe: Trained NPE
        simulator: Simulator with true posteriors
        theta_test: Test parameters
        x_test: Test clean observations
        y_test: Test noisy observations
        config: Configuration

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    device = torch.device(config.device)
    source_flow.eval()
    npe.eval()

    metrics = {}

    # Generate samples from learned source
    with torch.no_grad():
        y_test_device = y_test.to(device)
        _, y_rescaled = source_flow.rescale(torch.zeros_like(x_test[:1]).to(device), y_test_device)

        x_generated = source_flow.sample(
            y_rescaled, num_samples=config.num_eval_samples
        )  # (num_samples, n_test, x_dim)

        # Inverse rescale
        x_generated_unscaled = []
        for i in range(config.num_eval_samples):
            x_gen_i, _ = source_flow.inverse_rescale(x_generated[i], y_test_device)
            x_generated_unscaled.append(x_gen_i.cpu())
        x_generated = torch.stack(x_generated_unscaled, dim=0)

    # 1. Wasserstein distance per dimension
    from scipy.stats import wasserstein_distance

    for dim in range(min(3, x_test.shape[1])):
        w_dist = wasserstein_distance(
            x_test[:, dim].cpu().numpy(), x_generated[:, :, dim].mean(dim=0).cpu().numpy()
        )
        metrics[f"wasserstein_dim_{dim}"] = float(w_dist)

    # 2. MMD
    mmd = MMD(x_test.to(device), x_generated.mean(dim=0).to(device), kernel="rbf")
    metrics["mmd"] = float(mmd.item())

    # 3. Test ELBO
    elbo_loss = ELBOLoss(npe, source_flow)
    test_losses = []
    test_elbos = []

    with torch.no_grad():
        # Process in batches
        batch_size = 100
        for i in range(0, len(theta_test), batch_size):
            theta_batch = theta_test[i : i + batch_size].to(device)
            x_batch = x_test[i : i + batch_size].to(device)
            y_batch = y_test[i : i + batch_size].to(device)

            theta_batch, x_batch = npe.rescale(theta_batch, x_batch)
            x_batch, y_batch = source_flow.rescale(x_batch, y_batch)

            loss, batch_metrics = elbo_loss(theta_batch, x_batch, y_batch)
            test_losses.append(loss.item())
            test_elbos.append(batch_metrics["elbo"])

    metrics["test_loss"] = float(np.mean(test_losses))
    metrics["test_elbo"] = float(np.mean(test_elbos))

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return metrics


def evaluate_source_quality_true_old(
    source_flow: SourceFlow,
    simulator,
    theta_test: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    config: TrainingConfig,
) -> Dict[str, float]:
    """[OLD VERSION - NOT USED] Evaluate ELBO source quality by sampling π(θ|y) = ∫ p(θ|x) q(x|y) dx.

    This evaluates the induced marginal distribution on θ (the source).

    Args:
        source_flow: Trained source flow
        simulator: Simulator with analytical posterior
        theta_test: Test parameters (ground truth)
        x_test: Test clean observations
        y_test: Test noisy observations
        config: Configuration

    Returns:
        Dictionary of evaluation metrics on θ
    """
    print("\n" + "=" * 80)
    print("EVALUATION (TRUE POSTERIOR ELBO)")
    print("=" * 80)

    device = torch.device(config.device)
    source_flow.eval()

    metrics = {}

    # Sample from induced marginal π(θ|y) = ∫ p(θ|x) q(x|y) dx
    with torch.no_grad():
        y_test_device = y_test.to(device)
        _, y_rescaled = source_flow.rescale(torch.zeros_like(x_test[:1]).to(device), y_test_device)

        theta_samples_list = []

        print(f"Sampling {config.num_eval_samples} θ via π(θ|y) = ∫ p(θ|x) q(x|y) dx...")
        for i in range(config.num_eval_samples):
            # Step 1: Sample x ~ q(x|y) from source flow
            x_sampled = source_flow.sample(y_rescaled, num_samples=1)

            # Inverse rescale x
            x_sampled_unscaled, _ = source_flow.inverse_rescale(x_sampled.squeeze(0), y_test_device)

            # Step 2: Sample θ ~ p_true(θ|x) using analytical posterior
            posterior_dist = simulator.posterior_theta_given_x(x_sampled_unscaled.cpu())
            theta_sampled = posterior_dist.sample()

            theta_samples_list.append(theta_sampled.cpu())

        theta_samples = torch.stack(theta_samples_list, dim=0)  # (num_samples, n_test, theta_dim)

    # Evaluate θ samples against true θ
    from scipy.stats import wasserstein_distance

    for dim in range(theta_test.shape[1]):
        w_dist = wasserstein_distance(
            theta_test[:, dim].cpu().numpy(), theta_samples[:, :, dim].mean(dim=0).cpu().numpy()
        )
        metrics[f"wasserstein_theta_dim_{dim}"] = float(w_dist)

    # MMD between sampled θ and true θ
    mmd = MMD(theta_test.to(device), theta_samples.mean(dim=0).to(device), kernel="rbf")
    metrics["mmd_theta"] = float(mmd.item())

    # ELBO with TRUE posterior (training objective)
    elbo_loss = TruePosteriorELBOLoss(simulator, source_flow)
    test_losses = []
    test_elbos = []

    with torch.no_grad():
        batch_size = 100
        for i in range(0, len(theta_test), batch_size):
            theta_batch = theta_test[i : i + batch_size].to(device)
            x_batch = x_test[i : i + batch_size].to(device)
            y_batch = y_test[i : i + batch_size].to(device)

            x_batch, y_batch = source_flow.rescale(x_batch, y_batch)

            loss, batch_metrics = elbo_loss(theta_batch, x_batch, y_batch)
            test_losses.append(loss.item())
            test_elbos.append(batch_metrics["elbo"])

    metrics["test_loss"] = float(np.mean(test_losses))
    metrics["test_elbo"] = float(np.mean(test_elbos))

    print("\nΘ Metrics (Source Quality):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return metrics


def evaluate_source_quality_fm_old(
    flow_matching_x: FlowMatching,
    simulator,
    theta_test: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    config: TrainingConfig,
) -> Dict[str, float]:
    """[OLD VERSION - NOT USED] Evaluate FM source quality by sampling π(θ|y) = ∫ p(θ|x) q(x|y) dx.

    This evaluates the induced marginal distribution on θ (the source).

    Args:
        flow_matching_x: Trained FlowMatching model for p(x|y)
        simulator: Simulator with analytical posterior
        theta_test: Test parameters (ground truth)
        x_test: Test clean observations
        y_test: Test noisy observations
        config: Configuration

    Returns:
        Dictionary of evaluation metrics on θ
    """
    print("\n" + "=" * 80)
    print("EVALUATION (FLOW MATCHING)")
    print("=" * 80)

    device = torch.device(config.device)
    flow_matching_x.eval()

    metrics = {}

    # Sample from induced marginal π(θ|y) = ∫ p(θ|x) q(x|y) dx
    with torch.no_grad():
        y_test_device = y_test.to(device)
        _, y_rescaled = flow_matching_x.rescale(
            torch.zeros_like(x_test[:1]).to(device), y_test_device
        )

        theta_samples_list = []

        print(f"Sampling {config.num_eval_samples} θ via π(θ|y) = ∫ p(θ|x) q(x|y) dx...")
        for i in range(config.num_eval_samples):
            # Step 1: Sample x ~ q(x|y) from flow matching
            source = flow_matching_x.sample_source(y_rescaled).to(device)
            x_sampled = flow_matching_x.sample(
                source, y_rescaled, device, only_last=True, disable_tqdm=True
            )
            # Inverse rescale x
            x_sampled, _ = flow_matching_x.scale(x_sampled, y_test_device)

            # Step 2: Sample θ ~ p_true(θ|x) using analytical posterior
            posterior_dist = simulator.posterior_theta_given_x(x_sampled.cpu())
            theta_sampled = posterior_dist.sample()

            theta_samples_list.append(theta_sampled.cpu())

        theta_samples = torch.stack(theta_samples_list, dim=0)  # (num_samples, n_test, theta_dim)

    # Evaluate θ samples against true θ
    from scipy.stats import wasserstein_distance

    for dim in range(theta_test.shape[1]):
        w_dist = wasserstein_distance(
            theta_test[:, dim].cpu().numpy(), theta_samples[:, :, dim].mean(dim=0).cpu().numpy()
        )
        metrics[f"wasserstein_theta_dim_{dim}"] = float(w_dist)

    # MMD between sampled θ and true θ
    mmd = MMD(theta_test.to(device), theta_samples.mean(dim=0).to(device), kernel="rbf")
    metrics["mmd_theta"] = float(mmd.item())

    print("\nΘ Metrics (Source Quality):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return metrics


# ============================================================================
# Visualization
# ============================================================================


def visualize_results(
    theta_sampled: Tensor,
    x_sampled: Tensor,
    theta_post: Tensor,
    x_post: Tensor,
    theta_test: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    config: TrainingConfig,
    name="fm",
):
    """Create visualization of results.

    Args:
        source_flow: Trained source flow
        npe: Trained NPE
        simulator: Simulator with true posteriors
        theta_test: Test parameters
        x_test: Test observations
        y_test: Test noisy observations
        config: Configuration
    """
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)

    device = torch.device(config.device)

    # Select observations to visualize
    n_viz = min(config.num_viz_obs, len(y_test))
    indices = np.random.choice(len(y_test), n_viz, replace=False)

    fig, axes = plt.subplots(n_viz, 4, figsize=(15, 4 * n_viz))
    if n_viz == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        y_obs = y_test[idx : idx + 1]
        theta_true = theta_test[idx]
        x_true = x_test[idx]
        x_gen = x_sampled[:, idx, :]
        theta_learned = theta_sampled[:, idx, :]
        theta_true_post = theta_post[:, idx, :]
        x_true_post = x_post[:, idx, :]

        # Plot 1: Posterior comparison
        ax = axes[i, 0]
        if theta_true_post is not None:
            ax.scatter(
                theta_true_post[:, 0],
                theta_true_post[:, 1],
                alpha=0.3,
                s=10,
                label="True",
                c="blue",
            )
        ax.scatter(
            theta_learned[:, 0], theta_learned[:, 1], alpha=0.3, s=10, label="Learned", c="red"
        )
        ax.scatter(theta_true[0], theta_true[1], marker="*", s=200, c="black", label="Ground Truth")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.legend()
        ax.set_title(f"Posterior (obs {idx})")
        ax.grid(alpha=0.3)

        # Plot 2: Generated x vs true x
        ax = axes[i, 1]
        if x_test.shape[1] >= 2:
            ax.scatter(
                x_true_post[:, 0].cpu(),
                x_true_post[:, 1].cpu(),
                alpha=0.2,
                s=5,
                label="True x",
                c="blue",
            )
            ax.scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.2, s=5, label="Generated x", c="red")
            ax.scatter(
                x_true[0], x_true[1], marker="*", s=200, c="black", label="True x (this obs)"
            )
            ax.set_xlabel(r"$x_0$")
            ax.set_ylabel(r"$x_1$")
            ax.legend()
            ax.set_title(f"Generated vs True x")
            ax.grid(alpha=0.3)

        # Plot 3: Marginal distributions
        ax = axes[i, 2]
        if theta_true_post is not None:
            ax.hist(
                theta_true_post[:, 0], bins=30, alpha=0.5, label="True", density=True, color="blue"
            )
        ax.hist(theta_learned[:, 0], bins=30, alpha=0.5, label="Learned", density=True, color="red")
        ax.axvline(theta_true[0], color="black", linestyle="--", label="True θ₀")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(r"Marginal $\theta_0$")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.log_path / f"visualization_{name}.pdf", dpi=150)
    print(f"\n✓ Saved visualization to {config.log_path / f'visualization_{name}.pdf'}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    """Main execution function (refactored to use baseline trainers and unified functions)."""
    parser = argparse.ArgumentParser(description="Train source flow with ELBO loss")
    parser.add_argument("--task", type=str, default="pure_gaussian", help="Task name")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--theta_dim", type=int, default=2, help="Dimension of parameters")
    parser.add_argument("--obs_dim", type=int, default=3, help="Dimension of observations")
    parser.add_argument(
        "--no_load", action="store_true", help="Force retraining even if models exist"
    )
    args = parser.parse_args()

    # Determine whether to load existing models
    load_models = not args.no_load

    # Configuration
    sim_config = SimulatorConfig(
        name=args.task,
        theta_dim=args.theta_dim,
        obs_dim=args.obs_dim,
    )

    train_config = TrainingConfig(seed=args.seed)

    # Set seeds
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Create directories
    train_config.model_path.mkdir(parents=True, exist_ok=True)
    train_config.log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ELBO-BASED SOURCE LEARNING")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Samples: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"Device: {train_config.device}")

    # Load simulator and generate data
    print("\n" + "=" * 80)
    print("DATA GENERATION")
    print("=" * 80)

    simulator: PureGaussian = get_simulator(sim_config.name, **sim_config.params)
    print(f"✓ Loaded {sim_config.name} simulator")

    theta, x, y = generate_calibration_dataset(simulator, args.num_samples, generation="transitive")
    print(f"✓ Generated {args.num_samples} samples")
    print(f"  theta: {theta.shape}")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")

    # Infer dimensions from actual data
    train_config.theta_dim = theta.shape[1]
    train_config.obs_dim = x.shape[1]
    sim_config.theta_dim = theta.shape[1]
    sim_config.obs_dim = x.shape[1]

    print(
        f"✓ Inferred dimensions: theta_dim={train_config.theta_dim}, obs_dim={train_config.obs_dim}"
    )

    # Split into train/test
    n_test = int(0.2 * args.num_samples)
    theta_test, x_test, y_test = theta[:n_test], x[:n_test], y[:n_test]
    theta_train, x_train, y_train = theta[n_test:], x[n_test:], y[n_test:]

    print(f"✓ Split: {len(theta_train)} train, {len(theta_test)} test")

    device = torch.device(train_config.device)

    # ========================================================================
    # STAGE 1: Train NPE using baseline trainer
    # ========================================================================

    print("\n" + "=" * 80)
    print("STAGE 1: Training NPE p(θ|x)")
    print("=" * 80)

    task_dict = config_to_npe_dict(train_config, task_name=args.task)
    npe = train_npe(
        task=task_dict,
        theta=theta_train,
        x=x_train,
        device=device,
        model_path=train_config.model_path,
        logname="npe",
        load=load_models,  # Try to load existing model
        save=True,
    )

    print(f"✓ Saved NPE to {train_config.model_path / 'npe.pth'}")

    # ========================================================================
    # STAGE 2a: Train Source Flow with ELBO (NPE-based)
    # ========================================================================

    print("\n" + "=" * 80)
    print("STAGE 2a: Training Source Flow with ELBO (NPE-based)")
    print("=" * 80)

    source_flow_npe = train_source_elbo(
        theta=theta_train,
        y=y_train,
        posterior_fn=npe,
        use_true_posterior=False,
        config=train_config,
        device=device,
        load=load_models,  # Try to load existing model
        model_path=train_config.model_path,
        logname="source_flow_npe",
    )

    # Save source flow (NPE-based)
    source_path_npe = train_config.model_path / "source_flow_npe.pth"
    torch.save(source_flow_npe.state_dict(), source_path_npe)
    print(f"✓ Saved NPE-based source flow to {source_path_npe}")

    # ========================================================================
    # STAGE 2b: Train Source Flow with ELBO (True Posterior) - if available
    # ========================================================================

    if hasattr(simulator, "posterior_theta_given_x"):
        print("\n" + "=" * 80)
        print("STAGE 2b: Training Source Flow with ELBO (True Posterior)")
        print("=" * 80)
        print("This variant uses the analytical posterior instead of the learned NPE")
        print("to isolate source flow training from NPE approximation error.")

        source_flow_true = train_source_elbo(
            theta=theta_train,
            y=y_train,
            posterior_fn=simulator.posterior_theta_given_x,
            use_true_posterior=True,
            config=train_config,
            device=device,
            load=load_models,  # Try to load existing model
            model_path=train_config.model_path,
            logname="source_flow_true",
        )

        # Save source flow (true posterior)
        source_path_true = train_config.model_path / "source_flow_true.pth"
        torch.save(source_flow_true.state_dict(), source_path_true)
        print(f"✓ Saved true-posterior source flow to {source_path_true}")

        # ====================================================================
        # STAGE 2c: Train Source Flow with Flow Matching
        # ====================================================================

        print("\n" + "=" * 80)
        print("STAGE 2c: Training Source Flow with FLOW MATCHING")
        print("=" * 80)
        print("This variant uses standard flow matching loss on (x,y) pairs.")
        print("No posterior is used during training - only p(x|y) from data.")

        fm_config = config_to_fm_dict(conditional=True)
        flow_matching_x = train_flow_matching(
            target=x_train,
            cond=y_train,
            config=fm_config,
            device=device,
            logger=None,
            logname="flow_matching_x",
            load=load_models,  # Try to load existing model
            epochs=train_config.source_epochs,
            lr=train_config.source_lr,
            batch_size=train_config.source_batch_size,
            train_size=train_config.train_size,
            max_patience=train_config.max_patience,
            rescale="none",  # No rescaling
            model_path=train_config.model_path,
            save=True,
        )

        print(
            f"✓ Saved flow-matching source flow to {train_config.model_path / 'source_flow_fm.pth'}"
        )

        # ====================================================================
        # EVALUATION: Generate samples for all three methods
        # ====================================================================

        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        # Generate samples from true posterior
        post_theta_given_y = simulator.posterior_theta_given_y(y_test)
        post_x_given_y = simulator.denoise_dist(y_test)
        samples_true_posterior_theta = post_theta_given_y.sample((train_config.num_eval_samples,))
        samples_true_posterior_x = post_x_given_y.sample((train_config.num_eval_samples,))
        print("shape:", samples_true_posterior_theta.shape, samples_true_posterior_x.shape)

        # Generate samples for each method
        print("\nGenerating samples for ELBO-NPE...")
        samples_npe_theta, samples_npe_x = generate_samples_for_evaluation(
            source_flow_npe, npe, y_test, False, train_config.num_eval_samples, device
        )
        print("shape:", samples_npe_theta.shape, samples_npe_x.shape)

        print("\nGenerating samples for ELBO-True...")
        samples_true_theta, samples_true_x = generate_samples_for_evaluation(
            source_flow_true,
            simulator.posterior_theta_given_x,
            y_test,
            True,
            train_config.num_eval_samples,
            device,
        )

        print("\nGenerating samples for Flow Matching...")
        samples_fm_theta, samples_fm_x = generate_samples_for_evaluation(
            flow_matching_x,
            simulator.posterior_theta_given_x,
            y_test,
            True,
            train_config.num_eval_samples,
            device,
        )

        # Evaluate all methods with unified function
        print("\nEvaluating all methods...")
        metrics_npe = evaluate_source_quality(samples_npe_theta, theta_test, train_config)
        metrics_true = evaluate_source_quality(samples_true_theta, theta_test, train_config)
        metrics_fm = evaluate_source_quality(samples_fm_theta, theta_test, train_config)

        # Visualize results
        visualize_results(
            samples_npe_theta,
            samples_npe_x,
            samples_true_posterior_theta,
            samples_true_posterior_x,
            theta_test,
            x_test,
            y_test,
            train_config,
            name="npe",
        )

        visualize_results(
            samples_true_theta,
            samples_true_x,
            samples_true_posterior_theta,
            samples_true_posterior_x,
            theta_test,
            x_test,
            y_test,
            train_config,
            name="true",
        )

        visualize_results(
            samples_fm_theta,
            samples_fm_x,
            samples_true_posterior_theta,
            samples_true_posterior_x,
            theta_test,
            x_test,
            y_test,
            train_config,
            name="fm",
        )

        # Save metrics
        metrics_npe_path = train_config.log_path / "metrics_npe.json"
        with open(metrics_npe_path, "w") as f:
            json.dump(metrics_npe, f, indent=2)
        print(f"✓ Saved NPE-based metrics to {metrics_npe_path}")

        metrics_true_path = train_config.log_path / "metrics_true.json"
        with open(metrics_true_path, "w") as f:
            json.dump(metrics_true, f, indent=2)
        print(f"✓ Saved true-posterior metrics to {metrics_true_path}")

        metrics_fm_path = train_config.log_path / "metrics_fm.json"
        with open(metrics_fm_path, "w") as f:
            json.dump(metrics_fm, f, indent=2)
        print(f"✓ Saved flow-matching metrics to {metrics_fm_path}")

        # ====================================================================
        # COMPARISON
        # ====================================================================

        print("\n" + "=" * 80)
        print("COMPARISON: NPE-ELBO vs TRUE-ELBO vs FLOW-MATCHING")
        print("=" * 80)

        comparison = {
            "ELBO_NPE": metrics_npe,
            "ELBO_True": metrics_true,
            "FlowMatching": metrics_fm,
        }

        print("\nMetrics Comparison:")
        print(f"{'Metric':<30} {'ELBO_NPE':>15} {'ELBO_True':>15} {'FlowMatching':>15}")
        print("-" * 95)

        # Find all unique keys across all metrics
        all_keys = set()
        for m in [metrics_npe, metrics_true, metrics_fm]:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            npe_val = metrics_npe.get(key, float("nan"))
            true_val = metrics_true.get(key, float("nan"))
            fm_val = metrics_fm.get(key, float("nan"))
            print(f"{key:<30} {npe_val:>15.6f} {true_val:>15.6f} {fm_val:>15.6f}")

        # Save comparison
        comparison_path = train_config.log_path / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Saved comparison to {comparison_path}")

    else:
        print("\n" + "=" * 80)
        print("SKIPPING TRUE POSTERIOR AND FLOW MATCHING TRAINING")

    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)
    print(f"Models saved to: {train_config.model_path}")
    print(f"Results saved to: {train_config.log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ELBO vs Flow Matching comparison for Wind Tunnel task.

This script compares two methods for learning the source distribution p(x|y):
1. ELBO with FMPE-approximated posterior p(theta|x)
2. Flow Matching (standard conditional flow)

Key difference from simple_elbo_vs_fm.py: Instead of using an analytical
Gaussian posterior p(theta|x), we train an FMPE model on simulation data
to approximate this posterior.

Architecture:
- theta: 1D (hatch position, range [0, 45])
- x: 50D (simulated pressure time series)
- y: 50D (real observed pressure time series)

Data flow:
1. Generate simulation data (theta, x) from the simulator
2. Train FMPE to approximate p(theta|x)
3. Load calibration data (theta, y) from real observations
4. Train source flow p(x|y) using either:
   - ELBO: Uses FMPE to evaluate log p(theta|x) in the ELBO objective
   - FM: Standard flow matching on (x, y) pairs
5. Evaluate both methods on test data
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from flow_matching.torch_flow import FlowMatching
from simulator.wind_tunnel import WindTunnel
from posteriors import Estimator
from utils.networks import get_embedding_network


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> torch.Tensor:
    """Classifier-based 2-sample test returning accuracy."""
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


# =============================================================================
# Data Standardization
# =============================================================================


class Standardizer:
    """Standardize data to zero mean and unit variance."""

    def __init__(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True).clamp(min=1e-6)

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return (data - mean) / std

    def unstandardize(self, data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean


# =============================================================================
# Epsilon Encoder for ELBO
# =============================================================================


class EpsilonEncoder(nn.Module):
    """Parametrized distribution q_psi(eps|theta,y) for sampling epsilon during training."""

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        epsilon_dim: int,
        hidden_dims: list = [128, 128],
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.epsilon_dim = epsilon_dim

        # Build MLP: (theta, y) -> (mu_eps, L_eps)
        layers = []
        input_dim = theta_dim + y_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output layer for mean
        self.mean_layer = nn.Linear(input_dim, epsilon_dim)

        # Output layer for Cholesky factor (lower triangular matrix)
        self.n_tril_params = epsilon_dim * (epsilon_dim + 1) // 2
        self.chol_layer = nn.Linear(input_dim, self.n_tril_params)

    def forward(self, theta: torch.Tensor, y: torch.Tensor):
        """Compute distribution q_psi(eps|theta,y) with full covariance."""
        inputs = torch.cat([theta, y], dim=-1)
        h = self.encoder(inputs)

        mean = self.mean_layer(h)
        chol_params = self.chol_layer(h)

        batch_size = theta.shape[0]
        L = torch.zeros(
            batch_size,
            self.epsilon_dim,
            self.epsilon_dim,
            device=theta.device,
            dtype=theta.dtype,
        )

        tril_indices = torch.tril_indices(
            self.epsilon_dim, self.epsilon_dim, device=theta.device
        )
        L[:, tril_indices[0], tril_indices[1]] = chol_params

        # Ensure diagonal elements are positive
        diag_indices = torch.arange(self.epsilon_dim, device=theta.device)
        L[:, diag_indices, diag_indices] = (
            torch.nn.functional.softplus(L[:, diag_indices, diag_indices]) + 1e-4
        )

        return torch.distributions.MultivariateNormal(loc=mean, scale_tril=L)

    def sample(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self.forward(theta, y)
        return dist.rsample()

    def log_prob(
        self, epsilon: torch.Tensor, theta: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        dist = self.forward(theta, y)
        return dist.log_prob(epsilon)


# =============================================================================
# FMPE Wrapper for log_prob
# =============================================================================


class FMPEPosterior:
    """Wrapper around FMPE that provides log_prob via negative distance.

    Since FlowMatching doesn't have direct log_prob, we use a proxy based on
    the negative squared distance to the posterior mean. This is equivalent
    to assuming the posterior is approximately Gaussian centered at the
    FMPE sample mean.
    """

    def __init__(
        self,
        fmpe: FlowMatching,
        n_samples: int = 50,
        scale: float = 1.0,
    ):
        self.fmpe = fmpe
        self.n_samples = n_samples
        self.scale = scale  # Controls the "sharpness" of the proxy

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Approximate log p(theta|x) using negative squared distance to posterior mean.

        This is a simple proxy: log p(theta|x) ≈ -scale * ||theta - E[theta|x]||^2

        This encourages the source flow to generate x values such that the FMPE
        posterior mean is close to the true theta.
        """
        device = x.device
        batch_size = x.shape[0]

        # Sample from FMPE to estimate posterior mean
        with torch.no_grad():
            source = self.fmpe.sample_base(x, self.n_samples)
            cond = x.unsqueeze(0).expand(self.n_samples, -1, -1)

            samples = self.fmpe.sample(
                source.reshape(-1, *source.shape[2:]),
                cond.reshape(-1, *cond.shape[2:]),
                device,
                only_last=True,
            )
            samples = samples.reshape(self.n_samples, batch_size, -1)

            # Posterior mean and std
            posterior_mean = samples.mean(dim=0)  # (batch, theta_dim)
            posterior_std = samples.std(dim=0).clamp(min=0.1)  # (batch, theta_dim)

        # Compute log probability under Gaussian approximation
        # log p(theta|x) ≈ -0.5 * sum((theta - mu)^2 / sigma^2) - log(sigma) - const
        diff = theta - posterior_mean
        log_prob = -0.5 * self.scale * (diff**2 / posterior_std**2).sum(dim=-1)
        log_prob = log_prob - torch.log(posterior_std).sum(dim=-1)
        log_prob = log_prob - 0.5 * theta.shape[-1] * np.log(2 * np.pi)

        return log_prob

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from approximate posterior."""
        device = x.device
        batch_size = x.shape[0]

        with torch.no_grad():
            source = self.fmpe.sample_base(x, n_samples)
            cond = x.unsqueeze(0).expand(n_samples, -1, -1)

            samples = self.fmpe.sample(
                source.reshape(-1, *source.shape[2:]),
                cond.reshape(-1, *cond.shape[2:]),
                device,
                only_last=True,
            )
            return samples.reshape(n_samples, batch_size, -1)


# =============================================================================
# Training Functions
# =============================================================================


def train_fmpe_for_posterior(
    simulator: WindTunnel,
    n_sim: int = 10000,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    patience: int = 20,
) -> FlowMatching:
    """Train FMPE to approximate p(theta|x) on simulation data."""
    print("\n" + "=" * 60)
    print("Training FMPE for p(theta|x) approximation")
    print("=" * 60)

    # Generate simulation data
    print(f"Generating {n_sim} simulation samples...")
    sim_fn = simulator.get_simulator(misspecified=True)
    theta_sim = simulator.prior_dist.sample((n_sim,))
    x_sim = sim_fn(theta_sim)

    print(f"  theta shape: {theta_sim.shape}")
    print(f"  x shape: {x_sim.shape}")

    # Create FMPE model
    fmpe_config = {
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": simulator.theta_dim,
                "context_dim": 10,  # embedding dim
                "theta_with_glu": True,
                "hidden_dims": [64, 128, 128, 64],
                "dropout": 0.0,
                "batch_norm": False,
                "activation": "gelu",
            },
            "embedding_kwargs": {
                "name": "conv1d_light",
                "output_dim": 10,
                "image_size": simulator.obs_dim,
            },
        }
    }

    fmpe = FlowMatching(
        probability_path="ot2",
        prior="uniform",
        base_dist="gaussian",
        dim=torch.Size([simulator.theta_dim]),
        cond_dim=torch.Size([simulator.obs_dim]),
        probability_path_params={"sigma_min": 1e-4},
        **fmpe_config,
    ).to(device)

    # Set scales
    fmpe.set_scales(theta_sim, x_sim, "none")

    # Split into train/val
    n_train = int(len(theta_sim) * 0.8)
    theta_train, theta_val = theta_sim[:n_train], theta_sim[n_train:]
    x_train, x_val = x_sim[:n_train], x_sim[n_train:]

    train_dataset = TensorDataset(theta_train, x_train)
    val_dataset = TensorDataset(theta_val, x_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(fmpe.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = trange(epochs, desc="FMPE Training")
    for epoch in pbar:
        # Training
        fmpe.train()
        train_losses = []
        for theta_batch, x_batch in train_loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            loss = fmpe.compute_loss(theta_batch, x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fmpe.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        fmpe.eval()
        val_losses = []
        with torch.no_grad():
            for theta_batch, x_batch in val_loader:
                theta_batch = theta_batch.to(device)
                x_batch = x_batch.to(device)
                loss = fmpe.compute_loss(theta_batch, x_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        pbar.set_postfix({"train": f"{avg_train:.4f}", "val": f"{avg_val:.4f}"})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = fmpe.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        fmpe.load_state_dict(best_state)

    print(f"FMPE training complete. Best val loss: {best_val_loss:.4f}")
    return fmpe


def train_source_elbo(
    fmpe_posterior: FMPEPosterior,
    theta: torch.Tensor,
    y: torch.Tensor,
    x_dim: int,
    y_dim: int,
    theta_dim: int,
    theta_standardizer: Standardizer,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 64,
    device: str = "cuda",
    patience: int = 20,
) -> Tuple[FlowMatching, EpsilonEncoder, list]:
    """Train source flow p(x|y) using ELBO with FMPE-approximated posterior."""
    print("\n" + "=" * 60)
    print("Training Source Flow with ELBO (FMPE posterior)")
    print("=" * 60)

    # Create epsilon encoder
    epsilon_encoder = EpsilonEncoder(
        theta_dim=theta_dim,
        y_dim=y_dim,
        epsilon_dim=x_dim,
        hidden_dims=[128, 128],
    ).to(device)

    # Create source flow for p(x|y)
    source_flow_config = {
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": x_dim,
                "context_dim": 10,
                "theta_with_glu": True,
                "hidden_dims": [64, 128, 128, 64],
                "dropout": 0.0,
                "batch_norm": True,
                "activation": "gelu",
            },
            "embedding_kwargs": {
                "name": "conv1d_light",
                "output_dim": 10,
                "image_size": y_dim,
            },
        }
    }

    source_flow = FlowMatching(
        probability_path="ot2",
        prior="uniform",
        base_dist="gaussian",
        dim=torch.Size([x_dim]),
        cond_dim=torch.Size([y_dim]),
        probability_path_params={"sigma_min": 1e-4},
        **source_flow_config,
    ).to(device)

    source_flow.set_scales(torch.zeros(1, x_dim), y, "none")

    optimizer = torch.optim.Adam(
        list(source_flow.parameters()) + list(epsilon_encoder.parameters()), lr=lr
    )

    # Split into train/val
    n_train = int(len(theta) * 0.8)
    theta_train, theta_val = theta[:n_train], theta[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_dataset = TensorDataset(theta_train, y_train)
    val_dataset = TensorDataset(theta_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    best_val_loss = float("inf")
    best_source_flow_state = None
    best_encoder_state = None
    patience_counter = 0

    pbar = trange(epochs, desc="ELBO Training")
    for epoch in pbar:
        # Training
        source_flow.train()
        epsilon_encoder.train()
        epoch_losses = []

        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # 1. Get epsilon distribution from encoder
            dist = epsilon_encoder(theta_batch, y_batch)

            # 2. Sample eps ~ q_psi(eps|theta,y)
            eps = dist.rsample()

            # 3. Generate x = source_flow(eps, y)
            x0 = source_flow.sample_base(y_batch)
            x_generated = source_flow.sample(x0, y_batch, device, only_last=True)

            # 4. Evaluate log p(theta|x) using FMPE
            theta_unstd = theta_standardizer.unstandardize(theta_batch)
            log_p_theta_given_x = fmpe_posterior.log_prob(theta_unstd, x_generated)

            # 5. Compute log p(eps) under standard Gaussian prior
            log_p_eps = -0.5 * eps.pow(2).sum(-1) - 0.5 * x_dim * np.log(2 * np.pi)

            # 6. Compute log q_psi(eps|theta,y)
            log_q_eps = dist.log_prob(eps)

            # 7. ELBO with numerical stability
            # Clamp log probs to avoid extreme values
            log_p_theta_given_x = log_p_theta_given_x.clamp(min=-100, max=100)
            log_q_eps = log_q_eps.clamp(min=-100, max=100)
            log_p_eps = log_p_eps.clamp(min=-100, max=100)

            reconstruction_term = log_p_theta_given_x.mean()
            kl_div = (log_q_eps - log_p_eps).mean()

            # Clamp KL to prevent explosion
            kl_div = kl_div.clamp(min=-50, max=50)

            elbo = reconstruction_term - kl_div
            loss = -elbo

            # Skip if loss is nan
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(source_flow.parameters()) + list(epsilon_encoder.parameters()),
                5.0,
            )
            optimizer.step()

            epoch_losses.append(loss.item())

        # Validation
        source_flow.eval()
        epsilon_encoder.eval()
        val_losses_epoch = []

        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)

                dist = epsilon_encoder(theta_batch, y_batch)
                eps = dist.rsample()
                x0 = source_flow.sample_base(y_batch)
                x_generated = source_flow.sample(x0, y_batch, device, only_last=True)

                theta_unstd = theta_standardizer.unstandardize(theta_batch)
                log_p_theta_given_x = fmpe_posterior.log_prob(theta_unstd, x_generated)
                log_p_eps = -0.5 * eps.pow(2).sum(-1) - 0.5 * x_dim * np.log(2 * np.pi)
                log_q_eps = dist.log_prob(eps)

                # Clamp for stability
                log_p_theta_given_x = log_p_theta_given_x.clamp(min=-100, max=100)
                log_q_eps = log_q_eps.clamp(min=-100, max=100)
                log_p_eps = log_p_eps.clamp(min=-100, max=100)

                kl_div = (log_q_eps - log_p_eps).mean().clamp(min=-50, max=50)
                elbo = log_p_theta_given_x.mean() - kl_div

                if not (torch.isnan(elbo) or torch.isinf(elbo)):
                    val_losses_epoch.append(-elbo.item())

        avg_loss = np.mean(epoch_losses)
        avg_val = np.mean(val_losses_epoch)
        losses.append(avg_loss)

        pbar.set_postfix({"train": f"{avg_loss:.4f}", "val": f"{avg_val:.4f}"})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_source_flow_state = source_flow.state_dict()
            best_encoder_state = epsilon_encoder.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_source_flow_state is not None:
        source_flow.load_state_dict(best_source_flow_state)
        epsilon_encoder.load_state_dict(best_encoder_state)

    print(f"ELBO training complete. Best val loss: {best_val_loss:.4f}")
    return source_flow, epsilon_encoder, losses


def train_source_fm(
    x: torch.Tensor,
    y: torch.Tensor,
    x_dim: int,
    y_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
    patience: int = 20,
) -> Tuple[FlowMatching, list]:
    """Train source flow p(x|y) using standard Flow Matching."""
    print("\n" + "=" * 60)
    print("Training Source Flow with FLOW MATCHING")
    print("=" * 60)

    source_flow_config = {
        "drift": {
            "architecture": "cfnet",
            "posterior_kwargs": {
                "input_dim": x_dim,
                "context_dim": 10,
                "theta_with_glu": True,
                "hidden_dims": [64, 128, 128, 64],
                "dropout": 0.0,
                "batch_norm": True,
                "activation": "gelu",
            },
            "embedding_kwargs": {
                "name": "conv1d_light",
                "output_dim": 10,
                "image_size": y_dim,
            },
        }
    }

    source_flow = FlowMatching(
        probability_path="ot2",
        prior="uniform",
        base_dist="gaussian",
        dim=torch.Size([x_dim]),
        cond_dim=torch.Size([y_dim]),
        probability_path_params={"sigma_min": 1e-4},
        **source_flow_config,
    ).to(device)

    source_flow.set_scales(x, y, "none")

    optimizer = torch.optim.Adam(source_flow.parameters(), lr=lr)

    # Split into train/val
    n_train = int(len(x) * 0.8)
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    pbar = trange(epochs, desc="FM Training")
    for epoch in pbar:
        # Training
        source_flow.train()
        epoch_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = source_flow.compute_loss(x_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(source_flow.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Validation
        source_flow.eval()
        val_losses_epoch = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = source_flow.compute_loss(x_batch, y_batch)
                val_losses_epoch.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_val = np.mean(val_losses_epoch)
        losses.append(avg_loss)

        pbar.set_postfix({"train": f"{avg_loss:.4f}", "val": f"{avg_val:.4f}"})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = source_flow.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        source_flow.load_state_dict(best_state)

    print(f"FM training complete. Best val loss: {best_val_loss:.4f}")
    return source_flow, losses


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_source_flow(
    source_flow: FlowMatching,
    fmpe: FlowMatching,
    theta_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cuda",
    n_samples: int = 100,
) -> dict:
    """Evaluate source flow by sampling x ~ p(x|y) and then theta ~ p(theta|x)."""
    source_flow.eval()
    fmpe.eval()

    with torch.no_grad():
        y_test = y_test.to(device)

        # Sample x from source flow
        x0 = source_flow.sample_base(y_test, n_samples)
        cond = y_test.unsqueeze(0).expand(n_samples, -1, -1)

        x_samples = source_flow.sample(
            x0.reshape(-1, *x0.shape[2:]),
            cond.reshape(-1, *cond.shape[2:]),
            device,
            only_last=True,
        )
        x_samples = x_samples.reshape(n_samples, -1, x_samples.shape[-1])

        # Sample theta from FMPE given x
        theta_samples = []
        for i in range(n_samples):
            x_i = x_samples[i]
            theta0 = fmpe.sample_base(x_i)
            theta_i = fmpe.sample(theta0, x_i, device, only_last=True)
            theta_samples.append(theta_i)

        theta_samples = torch.stack(theta_samples, dim=0)  # (n_samples, n_test, theta_dim)
        theta_pred = theta_samples.mean(dim=0)  # (n_test, theta_dim)

    # Compute C2ST
    theta_test_cpu = theta_test.cpu()
    theta_pred_cpu = theta_pred.cpu()

    c2st_score = c2st(theta_test_cpu, theta_pred_cpu, n_folds=3)

    # Compute MSE
    mse = ((theta_test_cpu - theta_pred_cpu) ** 2).mean().item()

    return {
        "c2st": c2st_score.item(),
        "mse": mse,
        "theta_pred": theta_pred_cpu,
        "theta_samples": theta_samples.cpu(),
    }


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(
    simulator: WindTunnel,
    sample_sizes: list,
    n_seeds: int = 5,
    n_test: int = 50,
    epochs_fmpe: int = 200,
    epochs_source: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    """Run the full experiment comparing ELBO vs FM across sample sizes."""
    results = {
        "fm": {n: {"c2st": [], "mse": []} for n in sample_sizes},
        "elbo": {n: {"c2st": [], "mse": []} for n in sample_sizes},
    }

    # Train FMPE once (shared across all experiments)
    fmpe = train_fmpe_for_posterior(
        simulator,
        n_sim=10000,
        epochs=epochs_fmpe,
        lr=lr,
        batch_size=256,
        device=device,
    )
    fmpe_posterior = FMPEPosterior(fmpe, n_samples=20, scale=1.0)

    # Get total available observations
    total_obs = simulator.get_total_observations()
    print(f"\nTotal observations available: {total_obs}")

    # Reserve test set
    test_indices = tuple(range(total_obs - n_test, total_obs))
    theta_test, y_test = simulator.obs_from_files(n_test, indices=test_indices)
    print(f"Test set: {n_test} observations")

    for n_cal in sample_sizes:
        print(f"\n{'='*60}")
        print(f"SAMPLE SIZE: {n_cal}")
        print("=" * 60)

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load calibration data
            available_indices = list(range(total_obs - n_test))
            cal_indices = tuple(
                np.random.choice(available_indices, size=n_cal, replace=False)
            )
            theta_cal, y_cal = simulator.obs_from_files(n_cal, indices=cal_indices)

            # Generate simulation data for training FM (x, y pairs)
            # For FM, we need (x, y) pairs where x is simulation, y is observation
            # We simulate x given theta from calibration data
            sim_fn = simulator.get_simulator(misspecified=True)
            x_cal = sim_fn(theta_cal)

            # Standardize theta
            theta_standardizer = Standardizer(theta_cal)
            theta_cal_std = theta_standardizer.standardize(theta_cal)

            # Train FM
            fm_flow, _ = train_source_fm(
                x_cal,
                y_cal,
                x_dim=simulator.obs_dim,
                y_dim=simulator.obs_dim,
                epochs=epochs_source,
                lr=lr,
                batch_size=batch_size,
                device=device,
            )

            # Evaluate FM
            fm_results = evaluate_source_flow(
                fm_flow, fmpe, theta_test, y_test, device
            )
            results["fm"][n_cal]["c2st"].append(fm_results["c2st"])
            results["fm"][n_cal]["mse"].append(fm_results["mse"])
            print(f"  FM - C2ST: {fm_results['c2st']:.4f}, MSE: {fm_results['mse']:.4f}")

            # Train ELBO
            elbo_flow, _, _ = train_source_elbo(
                fmpe_posterior,
                theta_cal_std,
                y_cal,
                x_dim=simulator.obs_dim,
                y_dim=simulator.obs_dim,
                theta_dim=simulator.theta_dim,
                theta_standardizer=theta_standardizer,
                epochs=epochs_source,
                lr=lr * 0.1,  # Lower LR for ELBO
                batch_size=batch_size,
                device=device,
            )

            # Evaluate ELBO
            elbo_results = evaluate_source_flow(
                elbo_flow, fmpe, theta_test, y_test, device
            )
            results["elbo"][n_cal]["c2st"].append(elbo_results["c2st"])
            results["elbo"][n_cal]["mse"].append(elbo_results["mse"])
            print(
                f"  ELBO - C2ST: {elbo_results['c2st']:.4f}, MSE: {elbo_results['mse']:.4f}"
            )

    return results


def plot_results(results: dict, sample_sizes: list, output_dir: Path):
    """Plot comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # C2ST plot
    ax = axes[0]
    for method, color, marker in [("fm", "blue", "o"), ("elbo", "red", "s")]:
        means = [np.mean(results[method][n]["c2st"]) for n in sample_sizes]
        stds = [np.std(results[method][n]["c2st"]) for n in sample_sizes]
        ax.errorbar(
            sample_sizes,
            means,
            yerr=stds,
            label=method.upper(),
            color=color,
            marker=marker,
            capsize=5,
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Random (0.5)")
    ax.set_xlabel("Calibration Sample Size")
    ax.set_ylabel("C2ST (lower is better)")
    ax.set_title("C2ST vs Sample Size")
    ax.legend()
    ax.set_xscale("log")

    # MSE plot
    ax = axes[1]
    for method, color, marker in [("fm", "blue", "o"), ("elbo", "red", "s")]:
        means = [np.mean(results[method][n]["mse"]) for n in sample_sizes]
        stds = [np.std(results[method][n]["mse"]) for n in sample_sizes]
        ax.errorbar(
            sample_sizes,
            means,
            yerr=stds,
            label=method.upper(),
            color=color,
            marker=marker,
            capsize=5,
        )
    ax.set_xlabel("Calibration Sample Size")
    ax.set_ylabel("MSE (lower is better)")
    ax.set_title("MSE vs Sample Size")
    ax.legend()
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "wind_tunnel_elbo_vs_fm.pdf")
    plt.savefig(output_dir / "wind_tunnel_elbo_vs_fm.png", dpi=150)
    print(f"\nPlots saved to {output_dir}")


def print_summary(results: dict, sample_sizes: list):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Sample Size':<15} {'FM C2ST':<15} {'ELBO C2ST':<15} {'FM MSE':<15} {'ELBO MSE':<15}")
    print("-" * 70)

    for n in sample_sizes:
        fm_c2st = f"{np.mean(results['fm'][n]['c2st']):.3f}±{np.std(results['fm'][n]['c2st']):.3f}"
        elbo_c2st = f"{np.mean(results['elbo'][n]['c2st']):.3f}±{np.std(results['elbo'][n]['c2st']):.3f}"
        fm_mse = f"{np.mean(results['fm'][n]['mse']):.3f}±{np.std(results['fm'][n]['mse']):.3f}"
        elbo_mse = f"{np.mean(results['elbo'][n]['mse']):.3f}±{np.std(results['elbo'][n]['mse']):.3f}"
        print(f"{n:<15} {fm_c2st:<15} {elbo_c2st:<15} {fm_mse:<15} {elbo_mse:<15}")


def main():
    # Configuration
    sample_sizes = [10, 50, 100]
    n_seeds = 3
    n_test = 50
    epochs_fmpe = 200
    epochs_source = 100
    lr = 1e-3
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path("logs/wind_tunnel_elbo_vs_fm")

    print("=" * 60)
    print("WIND TUNNEL: ELBO vs FM COMPARISON")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Seeds: {n_seeds}")
    print(f"Test samples: {n_test}")

    # Initialize simulator
    simulator = WindTunnel(
        theta_dim=1,
        obs_dim=50,
    )

    # Run experiment
    results = run_experiment(
        simulator,
        sample_sizes=sample_sizes,
        n_seeds=n_seeds,
        n_test=n_test,
        epochs_fmpe=epochs_fmpe,
        epochs_source=epochs_source,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    # Plot and summarize
    plot_results(results, sample_sizes, output_dir)
    print_summary(results, sample_sizes)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

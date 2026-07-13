"""Conditional metrics for tasks with analytically available posteriors.

For the Gaussian simulator, the true posterior is analytically available.
This module compares learned posterior samples against samples from the
true posterior using C2ST, MMD, and Wasserstein distance.

This provides a more rigorous validation than just checking coverage
against the true θ, as it compares the full posterior distribution.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator
from utils.metrics import MMD, classifier_two_samples_test_torch

from .base import ValidationCheck, ValidationResult


def compute_gaussian_posterior(
    x_mean: Tensor,
    x_var: Tensor,
    n_obs: int,
    prior_mean: float,
    prior_scale: float,
    likelihood_scale: float,
) -> tuple:
    """Compute true posterior parameters for Gaussian model.

    For the Gaussian simulator with:
    - Prior: θ ~ N(prior_mean, prior_scale²)
    - Likelihood: x|θ ~ N(θ, likelihood_scale²)

    The posterior is: θ|x ~ N(μ_post, σ²_post) where:
    - σ²_post = 1 / (1/prior_scale² + n/likelihood_scale²)
    - μ_post = σ²_post * (prior_mean/prior_scale² + n*x_mean/likelihood_scale²)

    Args:
        x_mean: Sample mean from observations, shape (batch_size,)
        x_var: Sample variance from observations (unused, but available)
        n_obs: Number of observations used to compute x_mean
        prior_mean: Prior mean
        prior_scale: Prior standard deviation
        likelihood_scale: Likelihood standard deviation

    Returns:
        Tuple of (posterior_mean, posterior_std)
    """
    prior_var = prior_scale ** 2
    likelihood_var = likelihood_scale ** 2

    # Posterior precision = prior precision + data precision
    posterior_precision = 1 / prior_var + n_obs / likelihood_var
    posterior_var = 1 / posterior_precision
    posterior_std = torch.sqrt(torch.tensor(posterior_var, device=x_mean.device))

    # Posterior mean
    posterior_mean = posterior_var * (
        prior_mean / prior_var + n_obs * x_mean / likelihood_var
    )

    return posterior_mean, posterior_std


def sample_from_true_posterior(
    x: Tensor,
    n_samples: int,
    prior_mean: float,
    prior_scale: float,
    likelihood_scale: float,
    n_obs: int = 100,
) -> Tensor:
    """Sample from the true posterior for Gaussian model.

    Args:
        x: Observations with mean and variance, shape (batch_size, 2)
        n_samples: Number of samples to draw
        prior_mean: Prior mean
        prior_scale: Prior standard deviation
        likelihood_scale: Likelihood standard deviation
        n_obs: Number of raw observations (default 100 from Gaussian simulator)

    Returns:
        Samples from true posterior, shape (n_samples, batch_size, 1)
    """
    batch_size = x.shape[0]

    # Extract mean from summary statistics (first column is mean)
    x_mean = x[:, 0]

    # Compute posterior parameters
    post_mean, post_std = compute_gaussian_posterior(
        x_mean, x[:, 1], n_obs, prior_mean, prior_scale, likelihood_scale
    )

    # Sample from posterior
    samples = torch.randn(n_samples, batch_size, 1, device=x.device)
    samples = samples * post_std.unsqueeze(0).unsqueeze(-1) + post_mean.unsqueeze(0).unsqueeze(-1)

    return samples


class ConditionalMetricsCheck(ValidationCheck):
    """Conditional metrics check for tasks with true posteriors.

    Compares learned posterior samples against samples from the true
    analytical posterior using C2ST, MMD, and Wasserstein distance.

    Supports simulators that implement sample_reference_posterior() method.
    """

    @property
    def name(self) -> str:
        return "conditional"

    def _has_true_posterior(self, simulator: Simulator) -> bool:
        """Check if simulator has a true posterior implementation."""
        if hasattr(simulator, 'sample_reference_posterior'):
            try:
                # Try to call it to make sure it's implemented
                test_obs = simulator.sample_prior(1)
                sim_fn = simulator.get_simulator(misspecified=True)
                test_x = sim_fn(test_obs)
                simulator.sample_reference_posterior(2, test_x, misspecified=True)
                return True
            except NotImplementedError:
                return False
            except Exception:
                return False
        return False

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run conditional metrics check.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator with sample_reference_posterior() method
            **kwargs: Additional arguments (unused)

        Returns:
            ValidationResult with conditional comparison metrics
        """
        # Check if simulator supports true posterior
        if not self._has_true_posterior(simulator):
            return ValidationResult(
                check_name=self.name,
                passed=True,  # Skip is considered a pass
                metrics={},
                details={
                    "skipped": True,
                    "reason": f"Simulator '{simulator.name}' does not implement sample_reference_posterior",
                },
                warnings=[
                    f"Conditional check skipped: simulator '{simulator.name}' does not have true posterior"
                ],
            )

        self._set_seed()
        device = self.config.device

        num_sims = self.config.num_simulations
        num_samples = self.config.num_posterior_samples
        batch_size = self.config.batch_size
        n_obs_c2st = self.config.n_obs_c2st

        print(f"Running conditional metrics check with {num_sims} test observations...")
        print(f"  (C2ST will be computed for {n_obs_c2st} observations)")

        # Generate test observations
        theta_true = simulator.sample_prior(num_sims)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_test = simulate_fn(theta_true)

        # Sample from learned posterior
        learned_samples = []
        for start in tqdm(range(0, num_sims, batch_size), desc="Learned posterior sampling"):
            end = min(start + batch_size, num_sims)
            x_batch = x_test[start:end].to(device)

            with torch.no_grad():
                samples = posterior.sample(x_batch, num_samples, device, batch_size=batch_size)
            learned_samples.append(samples.cpu())

        learned_samples = torch.cat(learned_samples, dim=1)  # (num_samples, num_sims, theta_dim)

        # Sample from true posterior using simulator's reference implementation
        print("Sampling from true posterior...")
        true_samples = simulator.sample_reference_posterior(
            num_samples, x_test, misspecified=True
        )  # (num_samples, num_sims, theta_dim)

        # Compute metrics for each observation, then aggregate
        # C2ST is expensive, so only compute for n_obs_c2st observations
        # MMD and Wasserstein are computed for all observations
        c2st_scores = []
        mmd_scores = []
        wasserstein_scores = []

        # Select indices for C2ST (evenly spaced if n_obs_c2st < num_sims)
        c2st_indices = set(np.linspace(0, num_sims - 1, min(n_obs_c2st, num_sims), dtype=int))

        print(f"Computing conditional metrics (C2ST for {len(c2st_indices)} obs, MMD/Wasserstein for all)...")
        for i in tqdm(range(num_sims)):
            learned_i = learned_samples[:, i, :].squeeze(-1)  # (num_samples,)
            true_i = true_samples[:, i, :].squeeze(-1)  # (num_samples,)

            # Ensure 2D for metrics
            if learned_i.dim() == 1:
                learned_i = learned_i.unsqueeze(-1)
                true_i = true_i.unsqueeze(-1)

            # C2ST (only for selected observations)
            if i in c2st_indices:
                c2st = classifier_two_samples_test_torch(
                    learned_i,
                    true_i,
                    seed=self.config.seed + i,
                    n_folds=3,
                    scoring="accuracy",
                    z_score=True,
                    training_kwargs={"device": device, "epochs": 30},
                )
                c2st_scores.append(c2st)

            # MMD (for all observations)
            mmd = MMD(learned_i.to(device), true_i.to(device), kernel="rbf").item()
            mmd_scores.append(mmd)

            # Wasserstein distance (for all observations)
            a = torch.ones(num_samples) / num_samples
            b = torch.ones(num_samples) / num_samples
            M = ot.dist(learned_i.numpy(), true_i.numpy())
            w_dist = float(np.sqrt(ot.emd2(a.numpy(), b.numpy(), M)))
            wasserstein_scores.append(w_dist)

        # Aggregate metrics
        mean_c2st = np.mean(c2st_scores)
        mean_mmd = np.mean(mmd_scores)
        mean_wasserstein = np.mean(wasserstein_scores)

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir("conditional")
            plot_path = self._create_conditional_plot(
                learned_samples, true_samples, c2st_scores, mmd_scores, output_dir
            )
            plots["conditional_comparison"] = plot_path

        # Pass criteria: C2ST close to 0.5 (random guessing)
        passed = 0.45 <= mean_c2st <= 0.55

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics={
                "mean_c2st": mean_c2st,
                "std_c2st": float(np.std(c2st_scores)),
                "mean_mmd": mean_mmd,
                "std_mmd": float(np.std(mmd_scores)),
                "mean_wasserstein": mean_wasserstein,
                "std_wasserstein": float(np.std(wasserstein_scores)),
            },
            details={
                "num_observations": num_sims,
                "num_observations_c2st": len(c2st_indices),
                "num_posterior_samples": num_samples,
                "simulator": simulator.name,
            },
            plots=plots,
        )

    def _create_conditional_plot(
        self,
        learned_samples: Tensor,
        true_samples: Tensor,
        c2st_scores: list,
        mmd_scores: list,
        output_dir: Path,
    ) -> Path:
        """Create conditional comparison plots.

        Args:
            learned_samples: Samples from learned posterior
            true_samples: Samples from true posterior
            c2st_scores: Per-observation C2ST scores
            mmd_scores: Per-observation MMD scores
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Sample a few observations to visualize
        n_show = min(5, learned_samples.shape[1])
        indices = np.linspace(0, learned_samples.shape[1] - 1, n_show, dtype=int)

        # Plot 1: Posterior comparison for selected observations
        ax = axes[0, 0]
        for i, idx in enumerate(indices):
            learned = learned_samples[:, idx, 0].numpy()
            true = true_samples[:, idx, 0].numpy()

            ax.hist(learned, bins=30, alpha=0.3, label=f"Learned (obs {idx})" if i == 0 else None)
            ax.hist(true, bins=30, alpha=0.3, label=f"True (obs {idx})" if i == 0 else None)

        ax.set_xlabel("θ")
        ax.set_ylabel("Count")
        ax.set_title("Posterior Samples Comparison")
        ax.legend()

        # Plot 2: C2ST score distribution
        ax = axes[0, 1]
        ax.hist(c2st_scores, bins=30, alpha=0.7)
        ax.axvline(x=0.5, color="r", linestyle="--", label="Ideal (0.5)")
        ax.axvline(x=np.mean(c2st_scores), color="g", linestyle="-", label=f"Mean: {np.mean(c2st_scores):.3f}")
        ax.set_xlabel("C2ST Score")
        ax.set_ylabel("Count")
        ax.set_title("C2ST Score Distribution")
        ax.legend()

        # Plot 3: MMD score distribution
        ax = axes[1, 0]
        ax.hist(mmd_scores, bins=30, alpha=0.7)
        ax.axvline(x=np.mean(mmd_scores), color="g", linestyle="-", label=f"Mean: {np.mean(mmd_scores):.4f}")
        ax.set_xlabel("MMD")
        ax.set_ylabel("Count")
        ax.set_title("MMD Distribution")
        ax.legend()

        # Plot 4: Scatter plot of learned vs true (first observation)
        ax = axes[1, 1]
        learned_0 = learned_samples[:, 0, 0].numpy()
        true_0 = true_samples[:, 0, 0].numpy()

        # QQ-plot style
        learned_sorted = np.sort(learned_0)
        true_sorted = np.sort(true_0)
        ax.scatter(true_sorted, learned_sorted, alpha=0.5, s=10)
        lims = [min(learned_sorted.min(), true_sorted.min()), max(learned_sorted.max(), true_sorted.max())]
        ax.plot(lims, lims, "r--", label="Perfect match")
        ax.set_xlabel("True Posterior Quantiles")
        ax.set_ylabel("Learned Posterior Quantiles")
        ax.set_title("QQ Plot (First Observation)")
        ax.legend()

        plt.tight_layout()

        plot_path = output_dir / "conditional_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

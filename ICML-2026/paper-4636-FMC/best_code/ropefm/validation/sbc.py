"""Simulation-Based Calibration (SBC) validation check.

SBC is a fundamental diagnostic for simulation-based inference. For a
calibrated posterior, the rank of the true parameter within posterior
samples should be uniformly distributed.

References:
    Talts et al. (2018) "Validating Bayesian Inference Algorithms with
    Simulation-Based Calibration"
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator

from .base import ValidationCheck, ValidationConfig, ValidationResult, compute_ranks


class SBCCheck(ValidationCheck):
    """Simulation-Based Calibration check.

    For L simulations:
        1. Sample θ ~ prior
        2. Sample x ~ p(x|θ)
        3. Sample from posterior p(θ|x)
        4. Compute rank of true θ in posterior samples

    Ranks should be uniform if posterior is calibrated.
    Uses KS-test and chi-squared test for uniformity.
    """

    @property
    def name(self) -> str:
        return "sbc"

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run SBC validation.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments (unused)

        Returns:
            ValidationResult with rank uniformity tests
        """
        self._set_seed()
        device = self.config.device

        num_sims = self.config.num_simulations
        num_samples = self.config.num_posterior_samples
        batch_size = self.config.batch_size

        print(f"Running SBC with {num_sims} simulations, {num_samples} posterior samples each...")

        # Generate simulations
        theta_true = simulator.sample_prior(num_sims)  # (num_sims, theta_dim)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_sim = simulate_fn(theta_true)  # (num_sims, obs_dim)

        # Sample from posterior for each simulation
        all_ranks = []

        # Process in batches for memory efficiency
        for start in tqdm(range(0, num_sims, batch_size), desc="SBC sampling"):
            end = min(start + batch_size, num_sims)
            x_batch = x_sim[start:end].to(device)
            theta_batch = theta_true[start:end]

            with torch.no_grad():
                # Sample from posterior: shape (num_samples, batch_size, theta_dim)
                samples = posterior.sample(x_batch, num_samples, device, batch_size=batch_size)

            # Compute ranks for this batch
            ranks = compute_ranks(theta_batch.to(samples.device), samples)  # (batch_size, theta_dim)
            all_ranks.append(ranks.cpu())

        # Concatenate all ranks
        all_ranks = torch.cat(all_ranks, dim=0)  # (num_sims, theta_dim)

        # Perform uniformity tests
        results = self._test_uniformity(all_ranks, num_samples)

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir("sbc")
            plot_path = self._create_rank_histogram(all_ranks, num_samples, output_dir)
            plots["rank_histogram"] = plot_path

        # Determine pass/fail
        # Pass if KS-test p-value > 0.01 for all dimensions
        passed = all(results["ks_pvalues"][i] > 0.01 for i in range(len(results["ks_pvalues"])))

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics={
                "ks_statistics": results["ks_statistics"],
                "ks_pvalues": results["ks_pvalues"],
                "chi2_statistics": results["chi2_statistics"],
                "chi2_pvalues": results["chi2_pvalues"],
            },
            details={
                "num_simulations": num_sims,
                "num_posterior_samples": num_samples,
                "theta_dim": all_ranks.shape[1],
            },
            plots=plots,
        )

    def _test_uniformity(self, ranks: Tensor, num_samples: int) -> dict:
        """Test uniformity of ranks using KS and chi-squared tests.

        Args:
            ranks: Rank values, shape (num_sims, theta_dim)
            num_samples: Number of posterior samples (determines max rank)

        Returns:
            Dictionary with test statistics and p-values
        """
        num_sims, theta_dim = ranks.shape

        # Normalize ranks to [0, 1]
        normalized_ranks = ranks.float() / num_samples

        ks_statistics = []
        ks_pvalues = []
        chi2_statistics = []
        chi2_pvalues = []

        for d in range(theta_dim):
            dim_ranks = normalized_ranks[:, d].numpy()

            # KS test against uniform distribution
            ks_stat, ks_p = stats.kstest(dim_ranks, "uniform")
            ks_statistics.append(float(ks_stat))
            ks_pvalues.append(float(ks_p))

            # Chi-squared test with histogram bins
            num_bins = min(20, num_samples)
            observed, _ = np.histogram(dim_ranks, bins=num_bins, range=(0, 1))
            expected = np.ones(num_bins) * num_sims / num_bins
            chi2_stat, chi2_p = stats.chisquare(observed, expected)
            chi2_statistics.append(float(chi2_stat))
            chi2_pvalues.append(float(chi2_p))

        return {
            "ks_statistics": ks_statistics,
            "ks_pvalues": ks_pvalues,
            "chi2_statistics": chi2_statistics,
            "chi2_pvalues": chi2_pvalues,
        }

    def _create_rank_histogram(
        self, ranks: Tensor, num_samples: int, output_dir: Path
    ) -> Path:
        """Create and save rank histogram plots.

        Args:
            ranks: Rank values, shape (num_sims, theta_dim)
            num_samples: Number of posterior samples
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        num_sims, theta_dim = ranks.shape
        normalized_ranks = ranks.float() / num_samples

        # Create subplots for each dimension
        fig, axes = plt.subplots(
            1, theta_dim, figsize=(4 * theta_dim, 4), squeeze=False
        )

        num_bins = min(20, num_samples)

        for d in range(theta_dim):
            ax = axes[0, d]
            dim_ranks = normalized_ranks[:, d].numpy()

            # Plot histogram
            ax.hist(dim_ranks, bins=num_bins, range=(0, 1), density=True, alpha=0.7)

            # Plot expected uniform distribution
            ax.axhline(y=1.0, color="r", linestyle="--", label="Uniform")

            # Add confidence band (95%)
            n_per_bin = num_sims / num_bins
            std = np.sqrt(n_per_bin * (1 - 1/num_bins)) / num_sims * num_bins
            ax.axhspan(1 - 1.96 * std, 1 + 1.96 * std, alpha=0.2, color="r", label="95% CI")

            ax.set_xlabel("Normalized Rank")
            ax.set_ylabel("Density")
            ax.set_title(f"Dimension {d + 1}")
            ax.legend()

        plt.suptitle(f"SBC Rank Histogram (n={num_sims})")
        plt.tight_layout()

        plot_path = output_dir / "sbc_ranks.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

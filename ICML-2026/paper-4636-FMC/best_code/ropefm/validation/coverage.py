"""Expected coverage validation check.

For a well-calibrated posterior, the α-level credible interval should
contain the true parameter with probability α. This check computes
empirical coverage at various credible levels and compares to expected.

Reuses the existing acauc() metric from utils/metrics.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator
from utils.metrics import acauc

from .base import (
    ValidationCheck,
    ValidationConfig,
    ValidationResult,
    compute_credible_interval_coverage,
)


class CoverageCheck(ValidationCheck):
    """Expected coverage check.

    For α ∈ [0.1, 0.2, ..., 0.9], computes credible intervals and checks
    empirical coverage vs expected. Uses ACAUC metric from existing codebase.
    """

    @property
    def name(self) -> str:
        return "coverage"

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run coverage validation.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments (unused)

        Returns:
            ValidationResult with coverage calibration metrics
        """
        self._set_seed()
        device = self.config.device

        num_sims = self.config.num_simulations
        num_samples = self.config.num_posterior_samples
        alpha_levels = self.config.alpha_levels
        batch_size = self.config.batch_size

        print(f"Running coverage check with {num_sims} simulations...")

        # Generate simulations
        theta_true = simulator.sample_prior(num_sims)  # (num_sims, theta_dim)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_sim = simulate_fn(theta_true)  # (num_sims, obs_dim)

        # Sample from posterior for each simulation
        all_samples = []

        for start in tqdm(range(0, num_sims, batch_size), desc="Coverage sampling"):
            end = min(start + batch_size, num_sims)
            x_batch = x_sim[start:end].to(device)

            with torch.no_grad():
                samples = posterior.sample(x_batch, num_samples, device, batch_size=batch_size)

            all_samples.append(samples.cpu())

        # Concatenate all samples: (num_samples, num_sims, theta_dim)
        all_samples = torch.cat(all_samples, dim=1)

        # Compute coverage at each alpha level
        empirical_coverage = {}
        for alpha in alpha_levels:
            coverage = compute_credible_interval_coverage(theta_true, all_samples, alpha)
            empirical_coverage[alpha] = coverage

        # Compute ACAUC using existing metric
        acauc_value = acauc(
            theta_true, all_samples, alphas=torch.tensor(alpha_levels)
        ).item()

        # Compute mean calibration error
        calibration_errors = [
            abs(empirical_coverage[alpha] - alpha) for alpha in alpha_levels
        ]
        mean_calibration_error = np.mean(calibration_errors)

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir("coverage")
            plot_path = self._create_calibration_plot(
                alpha_levels, empirical_coverage, output_dir
            )
            plots["calibration_plot"] = plot_path

        # Pass if mean calibration error < 0.05
        passed = mean_calibration_error < 0.05

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics={
                "acauc": acauc_value,
                "mean_calibration_error": mean_calibration_error,
                "empirical_coverage": empirical_coverage,
                "expected_coverage": {alpha: alpha for alpha in alpha_levels},
            },
            details={
                "num_simulations": num_sims,
                "num_posterior_samples": num_samples,
                "alpha_levels": alpha_levels,
            },
            plots=plots,
        )

    def _create_calibration_plot(
        self, alpha_levels: list, empirical_coverage: dict, output_dir: Path
    ) -> Path:
        """Create and save calibration plot.

        Args:
            alpha_levels: List of credible levels
            empirical_coverage: Dict mapping alpha to empirical coverage
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # Sort alpha levels for plotting
        alphas = sorted(alpha_levels)
        empirical = [empirical_coverage[a] for a in alphas]

        # Plot empirical vs expected
        ax.plot(alphas, empirical, "o-", label="Empirical", markersize=8)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        # Add confidence band (approximate 95% CI using binomial)
        n = self.config.num_simulations
        for alpha in alphas:
            std = np.sqrt(alpha * (1 - alpha) / n)
            ax.fill_between(
                [alpha - 0.01, alpha + 0.01],
                [alpha - 1.96 * std, alpha - 1.96 * std],
                [alpha + 1.96 * std, alpha + 1.96 * std],
                alpha=0.1,
                color="blue",
            )

        ax.set_xlabel("Expected Coverage (α)")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Coverage Calibration")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / "coverage_calibration.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

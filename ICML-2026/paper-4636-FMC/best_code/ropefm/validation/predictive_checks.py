"""Posterior predictive checks.

Posterior predictive checks validate whether data generated from posterior
samples matches the distribution of observed data. If the posterior is
accurate, simulated data from posterior draws should be indistinguishable
from the original observations.

Uses C2ST and MMD metrics from existing utils/metrics.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator
from utils.metrics import MMD, classifier_two_samples_test_torch

from .base import ValidationCheck, ValidationResult


class PredictiveCheck(ValidationCheck):
    """Posterior predictive check.

    Samples θ from posterior, generates x' from simulator, and compares
    x' distribution to original x using C2ST and MMD.
    """

    @property
    def name(self) -> str:
        return "predictive"

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run posterior predictive check.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments (unused)

        Returns:
            ValidationResult with predictive check metrics
        """
        self._set_seed()
        device = self.config.device

        num_sims = self.config.num_simulations
        batch_size = self.config.batch_size

        print(f"Running posterior predictive check with {num_sims} simulations...")

        # Generate original data
        theta_true = simulator.sample_prior(num_sims)  # (num_sims, theta_dim)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_original = simulate_fn(theta_true)  # (num_sims, obs_dim)

        # Sample from posterior for each observation
        all_theta_posterior = []

        for start in tqdm(range(0, num_sims, batch_size), desc="Predictive sampling"):
            end = min(start + batch_size, num_sims)
            x_batch = x_original[start:end].to(device)

            with torch.no_grad():
                # Sample 1 theta per observation
                samples = posterior.sample(x_batch, 1, device, batch_size=batch_size)
                all_theta_posterior.append(samples.squeeze(0).cpu())

        # Concatenate posterior samples: (num_sims, theta_dim)
        theta_posterior = torch.cat(all_theta_posterior, dim=0)

        # Generate predictive samples using posterior draws
        x_predictive = simulate_fn(theta_posterior)  # (num_sims, obs_dim)

        # Flatten x for comparison if needed
        if x_original.dim() > 2:
            x_original_flat = x_original.flatten(start_dim=1)
            x_predictive_flat = x_predictive.flatten(start_dim=1)
        else:
            x_original_flat = x_original
            x_predictive_flat = x_predictive

        # Compute C2ST between original and predictive data
        c2st_score = classifier_two_samples_test_torch(
            x_original_flat,
            x_predictive_flat,
            seed=self.config.seed,
            n_folds=5,
            scoring="accuracy",
            z_score=True,
            training_kwargs={"device": device, "epochs": 50},
        )

        # Compute MMD between original and predictive data
        mmd_score = MMD(
            x_original_flat.to(device),
            x_predictive_flat.to(device),
            kernel="rbf",
        ).item()

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir("predictive")
            plot_path = self._create_predictive_plot(
                x_original_flat, x_predictive_flat, output_dir
            )
            plots["predictive_comparison"] = plot_path

        # Pass if C2ST is between 0.45-0.55 (random guessing level)
        passed = 0.45 <= c2st_score <= 0.55

        warnings = []
        if c2st_score < 0.45:
            warnings.append(
                f"C2ST score {c2st_score:.3f} < 0.45 suggests predictive samples "
                "are too similar to original (potential overfitting)"
            )
        elif c2st_score > 0.55:
            warnings.append(
                f"C2ST score {c2st_score:.3f} > 0.55 suggests predictive samples "
                "differ significantly from original (potential model misfit)"
            )

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics={
                "c2st": c2st_score,
                "mmd": mmd_score,
            },
            details={
                "num_simulations": num_sims,
                "c2st_pass_range": [0.45, 0.55],
            },
            plots=plots,
            warnings=warnings,
        )

    def _create_predictive_plot(
        self, x_original: Tensor, x_predictive: Tensor, output_dir: Path
    ) -> Path:
        """Create predictive comparison plot.

        Args:
            x_original: Original observations
            x_predictive: Predictive samples from posterior
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        x_orig_np = x_original.cpu().numpy()
        x_pred_np = x_predictive.cpu().numpy()

        # Plot first 2 dimensions (or all if <= 2)
        n_dims = min(2, x_orig_np.shape[1])

        fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5), squeeze=False)

        for d in range(n_dims):
            ax = axes[0, d]

            # Histogram comparison
            ax.hist(
                x_orig_np[:, d],
                bins=50,
                alpha=0.5,
                density=True,
                label="Original",
                color="blue",
            )
            ax.hist(
                x_pred_np[:, d],
                bins=50,
                alpha=0.5,
                density=True,
                label="Predictive",
                color="orange",
            )

            ax.set_xlabel(f"Dimension {d + 1}")
            ax.set_ylabel("Density")
            ax.legend()

        plt.suptitle("Posterior Predictive Check")
        plt.tight_layout()

        plot_path = output_dir / "posterior_predictive.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

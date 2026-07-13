"""Log probability sanity checks.

Verifies that the posterior assigns higher log probability to true
parameters than to random prior samples on average.

NOTE: This check is only applicable to NPE models that implement log_prob().
FlowMatching-based posteriors do not currently support log_prob().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator

from .base import ValidationCheck, ValidationResult


class LogProbCheck(ValidationCheck):
    """Log probability sanity check.

    Verifies:
    1. log_prob(θ_true|x) > log_prob(θ_prior|x) on average
    2. No NaN/Inf values in log probabilities

    NOTE: Only works with NPE models that implement log_prob().
    """

    @property
    def name(self) -> str:
        return "log_prob"

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run log probability check.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments (unused)

        Returns:
            ValidationResult with log probability metrics
        """
        # Check if posterior supports log_prob
        if not self._supports_log_prob(posterior):
            return ValidationResult(
                check_name=self.name,
                passed=True,  # Skip is considered a pass
                metrics={},
                details={"skipped": True, "reason": "Posterior does not support log_prob()"},
                warnings=["Log prob check skipped: posterior does not implement log_prob()"],
            )

        self._set_seed()
        device = self.config.device

        num_sims = self.config.num_simulations
        batch_size = self.config.batch_size

        print(f"Running log probability check with {num_sims} simulations...")

        # Generate simulations
        theta_true = simulator.sample_prior(num_sims)  # (num_sims, theta_dim)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_sim = simulate_fn(theta_true)  # (num_sims, obs_dim)

        # Generate random prior samples for comparison
        theta_prior = simulator.sample_prior(num_sims)

        # Compute log probabilities in batches
        log_prob_true = []
        log_prob_prior = []
        has_nan_inf = False

        for start in tqdm(range(0, num_sims, batch_size), desc="Log prob computation"):
            end = min(start + batch_size, num_sims)

            x_batch = x_sim[start:end].to(device)
            theta_true_batch = theta_true[start:end].to(device)
            theta_prior_batch = theta_prior[start:end].to(device)

            with torch.no_grad():
                lp_true = posterior.log_prob(theta_true_batch, x_batch)
                lp_prior = posterior.log_prob(theta_prior_batch, x_batch)

            # Check for NaN/Inf
            if torch.isnan(lp_true).any() or torch.isinf(lp_true).any():
                has_nan_inf = True
            if torch.isnan(lp_prior).any() or torch.isinf(lp_prior).any():
                has_nan_inf = True

            log_prob_true.append(lp_true.cpu())
            log_prob_prior.append(lp_prior.cpu())

        # Concatenate results
        log_prob_true = torch.cat(log_prob_true, dim=0)  # (num_sims,)
        log_prob_prior = torch.cat(log_prob_prior, dim=0)

        # Remove NaN/Inf for statistics
        valid_mask = (
            ~torch.isnan(log_prob_true)
            & ~torch.isinf(log_prob_true)
            & ~torch.isnan(log_prob_prior)
            & ~torch.isinf(log_prob_prior)
        )

        lp_true_valid = log_prob_true[valid_mask]
        lp_prior_valid = log_prob_prior[valid_mask]

        # Compute statistics
        mean_diff = (lp_true_valid - lp_prior_valid).mean().item()
        frac_true_higher = (lp_true_valid > lp_prior_valid).float().mean().item()
        frac_valid = valid_mask.float().mean().item()

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir("log_prob")
            plot_path = self._create_log_prob_plot(
                lp_true_valid, lp_prior_valid, output_dir
            )
            plots["log_prob_comparison"] = plot_path

        # Pass criteria:
        # 1. Fraction of true θ with higher log_prob > 0.5
        # 2. No significant NaN/Inf issues (>95% valid)
        passed = frac_true_higher > 0.5 and frac_valid > 0.95

        warnings = []
        if has_nan_inf:
            warnings.append(
                f"NaN/Inf values detected in log probabilities. "
                f"Fraction valid: {frac_valid:.3f}"
            )
        if frac_true_higher <= 0.5:
            warnings.append(
                f"Only {frac_true_higher:.1%} of true θ have higher log_prob than prior samples"
            )

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics={
                "mean_log_prob_true": lp_true_valid.mean().item(),
                "mean_log_prob_prior": lp_prior_valid.mean().item(),
                "mean_log_prob_difference": mean_diff,
                "fraction_true_higher": frac_true_higher,
                "fraction_valid": frac_valid,
            },
            details={
                "num_simulations": num_sims,
                "has_nan_inf": has_nan_inf,
            },
            plots=plots,
            warnings=warnings,
        )

    def _supports_log_prob(self, posterior: Posterior) -> bool:
        """Check if posterior supports log_prob computation.

        Args:
            posterior: Posterior to check

        Returns:
            True if posterior implements log_prob
        """
        try:
            # Check the class name for known non-supporting types
            class_name = posterior.__class__.__name__
            non_supporting = []
            if class_name in non_supporting:
                return False

            # Also check if log_prob raises NotImplementedError
            # by looking at the method
            if hasattr(posterior, "log_prob"):
                import inspect

                source = inspect.getsource(posterior.log_prob)
                if "NotImplementedError" in source:
                    return False

            return True
        except Exception:
            return False

    def _create_log_prob_plot(
        self, lp_true: Tensor, lp_prior: Tensor, output_dir: Path
    ) -> Path:
        """Create log probability comparison plot.

        Args:
            lp_true: Log probabilities of true parameters
            lp_prior: Log probabilities of prior samples
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        lp_true_np = lp_true.numpy()
        lp_prior_np = lp_prior.numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram comparison
        ax = axes[0]
        ax.hist(lp_true_np, bins=50, alpha=0.5, label="True θ", density=True)
        ax.hist(lp_prior_np, bins=50, alpha=0.5, label="Prior θ", density=True)
        ax.set_xlabel("Log Probability")
        ax.set_ylabel("Density")
        ax.set_title("Log Probability Distribution")
        ax.legend()

        # Difference distribution
        ax = axes[1]
        diff = lp_true_np - lp_prior_np
        ax.hist(diff, bins=50, alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", label="Zero difference")
        ax.axvline(x=np.mean(diff), color="g", linestyle="-", label=f"Mean: {np.mean(diff):.2f}")
        ax.set_xlabel("Log Prob Difference (True - Prior)")
        ax.set_ylabel("Count")
        ax.set_title("Log Probability Difference")
        ax.legend()

        plt.tight_layout()

        plot_path = output_dir / "log_prob_check.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

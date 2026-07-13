"""SBI library comparison validation.

Trains sbi.inference.NPE and sbi.inference.FMPE on the same data and
compares metrics to verify ropefm is competitive with the sbi library.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from posteriors import Posterior
from simulator.base import Simulator
from utils.metrics import MMD, acauc, classifier_two_samples_test_torch

from .base import ValidationCheck, ValidationResult


class SBIComparisonCheck(ValidationCheck):
    """Compare ropefm posterior against sbi library implementations.

    Trains sbi SNPE_C and FMPE on the same data and compares:
    - MMD between posterior samples
    - C2ST between posterior samples
    - ACAUC coverage metric
    """

    @property
    def name(self) -> str:
        return "sbi_comparison"

    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run sbi comparison check.

        Args:
            posterior: Trained ropefm posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments:
                - training_data: Dict with theta_sim, x_sim for training sbi models
                - sbi_method: "snpe" or "fmpe" (default: "snpe")
                - skip_training: If True, load pre-trained sbi model

        Returns:
            ValidationResult with comparison metrics
        """
        # Check if sbi is available
        try:
            import sbi
            from sbi.inference import SNPE, FMPE as SBI_FMPE
            from sbi.utils import BoxUniform
        except ImportError:
            return ValidationResult(
                check_name=self.name,
                passed=True,  # Skip is considered a pass
                metrics={},
                details={"skipped": True, "reason": "sbi library not installed"},
                warnings=["SBI comparison skipped: sbi library not installed. "
                          "Install with: pip install sbi"],
            )

        self._set_seed()
        device = self.config.device

        # Get training data
        training_data = kwargs.get("training_data", None)
        sbi_method = kwargs.get("sbi_method", "snpe")
        skip_training = kwargs.get("skip_training", False)

        if training_data is None:
            # Generate training data
            print("Generating training data for sbi comparison...")
            n_train = 10000
            theta_train = simulator.sample_prior(n_train)
            simulate_fn = simulator.get_simulator(misspecified=True)
            x_train = simulate_fn(theta_train)
            training_data = {"theta": theta_train, "x": x_train}

        theta_train = training_data["theta"]
        x_train = training_data["x"]

        # Train or load sbi model
        sbi_posterior = self._train_sbi_model(
            theta_train, x_train, simulator, sbi_method, device, skip_training
        )

        if sbi_posterior is None:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                metrics={},
                details={"error": "Failed to train sbi model"},
                warnings=["SBI model training failed"],
            )

        # Generate test data
        num_test = self.config.num_simulations
        num_samples = self.config.num_posterior_samples
        batch_size = self.config.batch_size

        print(f"Comparing posteriors on {num_test} test observations...")

        theta_test = simulator.sample_prior(num_test)
        simulate_fn = simulator.get_simulator(misspecified=True)
        x_test = simulate_fn(theta_test)

        # Sample from ropefm posterior
        ropefm_samples = []
        for start in tqdm(range(0, num_test, batch_size), desc="ropefm sampling"):
            end = min(start + batch_size, num_test)
            x_batch = x_test[start:end].to(device)

            with torch.no_grad():
                samples = posterior.sample(x_batch, num_samples, device, batch_size=batch_size)
            ropefm_samples.append(samples.cpu())

        ropefm_samples = torch.cat(ropefm_samples, dim=1)  # (num_samples, num_test, theta_dim)

        # Sample from sbi posterior
        sbi_samples = []
        for start in tqdm(range(0, num_test, batch_size), desc="sbi sampling"):
            end = min(start + batch_size, num_test)
            x_batch = x_test[start:end].to(device)

            samples_batch = []
            for i in range(end - start):
                sbi_posterior.set_default_x(x_batch[i])
                s = sbi_posterior.sample((num_samples,))
                samples_batch.append(s.unsqueeze(1))

            samples_batch = torch.cat(samples_batch, dim=1)  # (num_samples, batch, theta_dim)
            sbi_samples.append(samples_batch.cpu())

        sbi_samples = torch.cat(sbi_samples, dim=1)  # (num_samples, num_test, theta_dim)

        # Compute comparison metrics
        metrics = self._compute_comparison_metrics(
            ropefm_samples, sbi_samples, theta_test, x_test, device
        )

        # Create plots if requested
        plots = {}
        if self.config.save_plots and self.config.output_dir is not None:
            output_dir = self._ensure_output_dir(f"sbi_comparison_{sbi_method}")
            plot_path = self._create_comparison_plot(
                ropefm_samples, sbi_samples, theta_test, metrics, output_dir
            )
            plots["sbi_comparison"] = plot_path

        # Pass if ropefm MMD is within 10% of sbi MMD, or ropefm is better
        mmd_ratio = metrics["ropefm_mmd"] / (metrics["sbi_mmd"] + 1e-10)
        passed = mmd_ratio <= 1.1 or metrics["ropefm_mmd"] <= metrics["sbi_mmd"]

        return ValidationResult(
            check_name=self.name,
            passed=passed,
            metrics=metrics,
            details={
                "num_test": num_test,
                "num_samples": num_samples,
                "sbi_method": sbi_method,
                "mmd_ratio": mmd_ratio,
            },
            plots=plots,
        )

    def _train_sbi_model(
        self,
        theta: Tensor,
        x: Tensor,
        simulator: Simulator,
        method: str,
        device: torch.device,
        skip_training: bool,
    ):
        """Train or load an sbi model.

        Args:
            theta: Training parameters
            x: Training observations
            simulator: Simulator for prior specification
            method: "snpe" or "fmpe"
            device: Device for training
            skip_training: If True, skip training (not implemented)

        Returns:
            Trained sbi posterior or None if failed
        """
        try:
            from sbi.inference import SNPE, FMPE as SBI_FMPE
            from sbi.utils import BoxUniform, process_prior

            # Define prior for sbi (must be on same device as training)
            prior_dist = simulator.get_prior_dist()

            # Try to get prior bounds, else use estimated bounds
            try:
                low = simulator.prior_params.get("low", torch.ones(simulator.theta_dim) * -10)
                high = simulator.prior_params.get("high", torch.ones(simulator.theta_dim) * 10)
                # Move to training device for sbi compatibility
                prior = BoxUniform(low=low.to(device), high=high.to(device))
            except Exception:
                # Fall back to a wide prior on the training device
                prior = BoxUniform(
                    low=-10 * torch.ones(simulator.theta_dim, device=device),
                    high=10 * torch.ones(simulator.theta_dim, device=device),
                )

            # Initialize inference
            if method == "snpe":
                inference = SNPE(prior=prior, device=str(device))
            elif method == "fmpe":
                inference = SBI_FMPE(prior=prior, device=str(device))
            else:
                raise ValueError(f"Unknown sbi method: {method}")

            # Append simulations (move to device for sbi compatibility)
            inference.append_simulations(theta.to(device), x.to(device))

            print(f"Training sbi {method.upper()}...")
            density_estimator = inference.train(
                training_batch_size=256,
                max_num_epochs=100,
                show_train_summary=True,
            )

            # Build posterior
            posterior = inference.build_posterior(density_estimator)

            return posterior

        except Exception as e:
            print(f"Error training sbi model: {e}")
            return None

    def _compute_comparison_metrics(
        self,
        ropefm_samples: Tensor,
        sbi_samples: Tensor,
        theta_true: Tensor,
        x_test: Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute comparison metrics between ropefm and sbi.

        Args:
            ropefm_samples: Samples from ropefm posterior
            sbi_samples: Samples from sbi posterior
            theta_true: True parameter values
            x_test: Test observations
            device: Compute device

        Returns:
            Dictionary of comparison metrics
        """
        num_samples, num_test, theta_dim = ropefm_samples.shape

        # Flatten samples for global comparison
        ropefm_flat = ropefm_samples.reshape(-1, theta_dim)
        sbi_flat = sbi_samples.reshape(-1, theta_dim)

        # C2ST between ropefm and sbi samples
        c2st_between = classifier_two_samples_test_torch(
            ropefm_flat[:10000],  # Subsample for speed
            sbi_flat[:10000],
            seed=self.config.seed,
            n_folds=3,
            scoring="accuracy",
            training_kwargs={"device": device, "epochs": 30},
        )

        # MMD between ropefm and sbi
        mmd_between = MMD(
            ropefm_flat[:5000].to(device),
            sbi_flat[:5000].to(device),
            kernel="rbf",
        ).item()

        # ACAUC for each method
        acauc_ropefm = acauc(theta_true, ropefm_samples).item()
        acauc_sbi = acauc(theta_true, sbi_samples).item()

        # Compute MMD to true theta for comparison
        # (how close are samples to ground truth on average)
        theta_expanded = theta_true.unsqueeze(0).expand_as(ropefm_samples)

        ropefm_mmd = MMD(
            ropefm_flat[:5000].to(device),
            theta_expanded.reshape(-1, theta_dim)[:5000].to(device),
            kernel="rbf",
        ).item()

        sbi_mmd = MMD(
            sbi_flat[:5000].to(device),
            theta_expanded.reshape(-1, theta_dim)[:5000].to(device),
            kernel="rbf",
        ).item()

        return {
            "c2st_between": c2st_between,
            "mmd_between": mmd_between,
            "ropefm_acauc": acauc_ropefm,
            "sbi_acauc": acauc_sbi,
            "ropefm_mmd": ropefm_mmd,
            "sbi_mmd": sbi_mmd,
        }

    def _create_comparison_plot(
        self,
        ropefm_samples: Tensor,
        sbi_samples: Tensor,
        theta_true: Tensor,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> Path:
        """Create comparison plots.

        Args:
            ropefm_samples: Samples from ropefm
            sbi_samples: Samples from sbi
            theta_true: True parameters
            metrics: Computed metrics
            output_dir: Directory to save plots

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Sample comparison for first observation
        ax = axes[0, 0]
        idx = 0
        ropefm_0 = ropefm_samples[:, idx, 0].numpy()
        sbi_0 = sbi_samples[:, idx, 0].numpy()
        true_0 = theta_true[idx, 0].item()

        ax.hist(ropefm_0, bins=50, alpha=0.5, label="ropefm", density=True)
        ax.hist(sbi_0, bins=50, alpha=0.5, label="sbi", density=True)
        ax.axvline(x=true_0, color="r", linestyle="--", label=f"True θ = {true_0:.2f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("Density")
        ax.set_title("Posterior Comparison (First Observation)")
        ax.legend()

        # Plot 2: ACAUC comparison
        ax = axes[0, 1]
        methods = ["ropefm", "sbi"]
        acauc_values = [metrics["ropefm_acauc"], metrics["sbi_acauc"]]
        colors = ["blue", "orange"]
        bars = ax.bar(methods, acauc_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color="r", linestyle="--", label="Perfect calibration")
        ax.set_ylabel("ACAUC")
        ax.set_title("Calibration Comparison")
        ax.legend()
        for bar, val in zip(bars, acauc_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: MMD comparison
        ax = axes[1, 0]
        mmd_values = [metrics["ropefm_mmd"], metrics["sbi_mmd"]]
        bars = ax.bar(methods, mmd_values, color=colors, alpha=0.7)
        ax.set_ylabel("MMD to True θ")
        ax.set_title("MMD Comparison (lower is better)")
        for bar, val in zip(bars, mmd_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = (
            f"Summary:\n\n"
            f"C2ST between ropefm and sbi: {metrics['c2st_between']:.3f}\n"
            f"  (0.5 = identical, 1.0 = completely different)\n\n"
            f"MMD between ropefm and sbi: {metrics['mmd_between']:.4f}\n"
            f"  (lower = more similar)\n\n"
            f"ACAUC:\n"
            f"  ropefm: {metrics['ropefm_acauc']:.4f}\n"
            f"  sbi: {metrics['sbi_acauc']:.4f}\n\n"
            f"MMD to true θ:\n"
            f"  ropefm: {metrics['ropefm_mmd']:.4f}\n"
            f"  sbi: {metrics['sbi_mmd']:.4f}"
        )
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, va="top", fontsize=11, family="monospace")

        plt.tight_layout()

        plot_path = output_dir / "sbi_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

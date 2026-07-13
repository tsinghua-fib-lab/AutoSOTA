"""Base classes for validation checks.

This module provides the abstract base class for validation checks
and associated data structures for configuration and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from posteriors import Posterior
from simulator.base import Simulator


@dataclass
class ValidationConfig:
    """Configuration for validation checks.

    Attributes:
        num_simulations: Number of simulations for SBC and other checks
        num_posterior_samples: Number of posterior samples per observation
        n_obs_c2st: Number of observations for conditional C2ST (expensive, typically ~5)
        alpha_levels: Credible levels for coverage tests
        seed: Random seed for reproducibility
        device: Device to run validation on
        batch_size: Batch size for sampling
        save_plots: Whether to save diagnostic plots
        output_dir: Directory to save results
    """

    num_simulations: int = 1000
    num_posterior_samples: int = 1000
    n_obs_c2st: int = 5
    alpha_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    seed: int = 42
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 500
    save_plots: bool = True
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        check_name: Name of the validation check
        passed: Whether the check passed
        metrics: Dictionary of computed metrics
        details: Additional details about the check
        plots: Dictionary of plot paths (if saved)
        warnings: List of warnings encountered
    """

    check_name: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Path] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "metrics": self.metrics,
            "details": {
                k: str(v) if isinstance(v, Path) else v for k, v in self.details.items()
            },
            "plots": {k: str(v) for k, v in self.plots.items()},
            "warnings": self.warnings,
        }


class ValidationCheck(ABC):
    """Abstract base class for validation checks.

    Subclasses must implement the `run` method to perform the validation.
    """

    def __init__(self, config: ValidationConfig):
        """Initialize the validation check.

        Args:
            config: Validation configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the validation check."""
        pass

    @abstractmethod
    def run(
        self,
        posterior: Posterior,
        simulator: Simulator,
        **kwargs,
    ) -> ValidationResult:
        """Run the validation check.

        Args:
            posterior: Trained posterior estimator
            simulator: Simulator for generating data
            **kwargs: Additional arguments for specific checks

        Returns:
            ValidationResult containing metrics and pass/fail status
        """
        pass

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _ensure_output_dir(self, subdir: str) -> Path:
        """Ensure output directory exists and return path.

        Args:
            subdir: Subdirectory name for this check

        Returns:
            Path to output directory
        """
        if self.config.output_dir is None:
            raise ValueError("output_dir must be set in config to save results")

        output_path = self.config.output_dir / subdir
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def compute_ranks(true_values: Tensor, samples: Tensor) -> Tensor:
    """Compute ranks of true values within samples.

    For SBC, the rank of the true parameter within posterior samples
    should be uniformly distributed if the posterior is calibrated.

    Args:
        true_values: True parameter values, shape (N, D)
        samples: Posterior samples, shape (M, N, D)

    Returns:
        Ranks of true values, shape (N, D)
    """
    # samples shape: (M, N, D) - M samples per N observations, D dimensions
    # true_values shape: (N, D)

    # Count how many samples are less than the true value
    # Broadcast true_values to (1, N, D) for comparison with (M, N, D)
    ranks = (samples < true_values.unsqueeze(0)).sum(dim=0)  # (N, D)

    return ranks


def compute_credible_interval_coverage(
    true_values: Tensor, samples: Tensor, alpha: float
) -> float:
    """Compute empirical coverage of credible intervals.

    Args:
        true_values: True parameter values, shape (N, D)
        samples: Posterior samples, shape (M, N, D)
        alpha: Credible level (e.g., 0.9 for 90% interval)

    Returns:
        Empirical coverage (fraction of true values within intervals)
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q

    # Compute quantiles along sample dimension
    lower = torch.quantile(samples, lower_q, dim=0)  # (N, D)
    upper = torch.quantile(samples, upper_q, dim=0)  # (N, D)

    # Check coverage
    covered = (true_values >= lower) & (true_values <= upper)  # (N, D)

    return covered.float().mean().item()

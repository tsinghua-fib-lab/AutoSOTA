"""Bootstrap utilities for confidence interval estimation."""
from __future__ import annotations

import numpy as np
import scipy.special
import logging

# Fix import path for consistency
from core.net_benefit import net_benefit_for_prevalences
from stats.isotonic import get_calibration_model
from reproducibility.seed_manager import get_random_generator
from reproducibility.config import get_config

# Set up logging
logger = logging.getLogger(__name__)

def bootstrap_net_benefit_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prevalence_grid: np.ndarray,
    cost_ratio: float,
    n_bootstrap: int = 100,
    ci_levels: tuple[float, float] = (2.5, 97.5),
    train_prevalence: float | None = None,
    normalize: bool = False,
    random_seed: int | None = None,
    calibrate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute net benefit curve confidence intervals using bootstrap.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_pred_proba : np.ndarray
        Probabilistic predictions for the positive class.
    prevalence_grid : np.ndarray
        Grid of prevalence values to compute net benefit for.
    cost_ratio : float
        Cost ratio parameter for net benefit calculation.
    n_bootstrap : int, default=100
        Number of bootstrap samples.
    ci_levels : tuple[float, float], default=(2.5, 97.5)
        Lower and upper percentiles for confidence interval.
    train_prevalence : float | None, default=None
        Training prevalence. If None, computed from y_true.
    normalize : bool, default=False
        Whether to normalize net benefit.
    random_seed : int | None, default=None
        Random seed for reproducibility.
    calibrate : bool, default=False
        Whether to calibrate predictions for each bootstrap sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (lower_ci, median, upper_ci) arrays.
    """
    # Use the new seed management system
    config = get_config()
    if random_seed is None:
        random_seed = config.seed
    rng = get_random_generator(random_seed, component_name="bootstrap_net_benefit_ci")
    
    # Log this analysis for reproducibility
    config.log_analysis(
        "bootstrap_net_benefit_ci", 
        {
            "seed": random_seed, 
            "n_bootstrap": n_bootstrap,
            "calibrate": calibrate,
        }
    )
    
    logger.debug("Running bootstrap with seed %s", random_seed)
    
    # Initialize bootstrap results array
    boot = np.empty((n_bootstrap, prevalence_grid.size))

    # Get number of samples
    n_samples = len(y_true)

    # If train_prevalence is None, compute from data
    if train_prevalence is None:
        train_prevalence = float(np.mean(y_true))

    # Perform bootstrap sampling
    b = 0
    while b < n_bootstrap:
        # Sample with replacement
        idx_sample = rng.integers(0, n_samples, size=n_samples)
        idx_sample = np.sort(idx_sample)

        # Get bootstrap sample
        boot_y_true = y_true[idx_sample]
        boot_scores = y_pred_proba[idx_sample]
        boot_train_prev = float(np.mean(boot_y_true))

        # Apply calibration if requested
        if calibrate:
            ir = get_calibration_model(boot_y_true, boot_scores)

            # Resample for calibration evaluation
            idx_sample = rng.integers(0, n_samples, size=n_samples)
            idx_sample = np.sort(idx_sample)
            boot_y_true = y_true[idx_sample]
            boot_scores = ir.transform(y_pred_proba[idx_sample])
            boot_train_prev = float(np.mean(boot_y_true))

        try:
            # Compute net benefit for bootstrap sample
            boot[b] = net_benefit_for_prevalences(
                boot_y_true,
                boot_scores,
                prevalence_grid=prevalence_grid,
                cost_ratio=cost_ratio,
                train_prevalence=boot_train_prev,
                normalize=normalize
            )
            b += 1
        except ValueError:
            # Skip this bootstrap sample if there's an error
            continue

    # Compute percentiles
    lower = np.percentile(boot, ci_levels[0], axis=0)
    median = np.percentile(boot, 50, axis=0)
    upper = np.percentile(boot, ci_levels[1], axis=0)

    # Apply logit transformation for better CI estimation
    # Use errstate to suppress expected warnings from logit/expit transformations
    with np.errstate(invalid='ignore'):
        lower, median, upper = scipy.special.logit([lower, median, upper])
        lower, upper = 2 * median - upper, 2 * median - lower
        lower, upper = scipy.special.expit([lower, upper])

    return lower, median, upper

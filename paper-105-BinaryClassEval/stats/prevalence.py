"""Utilities for working with prevalence grids."""
from __future__ import annotations

import numpy as np


def default_prevalence_grid(num: int = 150) -> np.ndarray:
    """Return a prevalence grid over (0, 1) with more points at the extremes.

    The grid concentrates points near 0 and 1 to better show the asymptotic
    behavior at the extremes.

    Parameters
    ----------
    num : int, default=150
        Number of points in the grid.

    Returns
    -------
    np.ndarray
        Array of prevalence values between 0 and 1.
    """
    # Use very small epsilon to get closer to 0 and 1
    eps = 1e-6

    # Create a non-linear grid with more points near 0 and 1
    # Transform a uniform grid using a beta distribution shape
    uniform = np.linspace(0, 1, num)

    # Use a beta distribution shape to concentrate points at both ends
    # Beta(0.3, 0.3) puts more weight at 0 and 1
    beta_grid = np.power(uniform, 0.3) * np.power(1 - uniform, 0.3)
    beta_grid = beta_grid / np.max(beta_grid)  # Normalize

    # Invert and scale to [eps, 1-eps]
    beta_grid = 1 - beta_grid  # Invert so small values -> close to 0 or 1
    prevalence = eps + (1 - 2*eps) * beta_grid

    # Ensure we include some exact endpoint values
    prevalence = np.sort(np.concatenate([
        prevalence,
        np.array([eps, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1-eps])
    ]))

    return prevalence


def log_odds_grid(prevalence_grid: np.ndarray) -> np.ndarray:
    """Convert prevalence grid to log-odds grid.

    Parameters
    ----------
    prevalence_grid : np.ndarray
        Array of prevalence values between 0 and 1.

    Returns
    -------
    np.ndarray
        Array of log-odds values.
    """
    return np.log(prevalence_grid / (1 - prevalence_grid))


def prob_ticks_for_axis() -> tuple[list[float], list[float]]:
    """Return probability ticks and corresponding log-odds for axis display.

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple of (probability ticks, log-odds ticks).
    """
    prob_ticks = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    logodds_ticks = [np.log(p / (1 - p)) for p in prob_ticks]
    return prob_ticks, logodds_ticks
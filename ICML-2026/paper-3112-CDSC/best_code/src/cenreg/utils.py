import numpy as np


def create_bins(max_y: float, min_y: float = 0.0, num_bins=10, algorithm: str = "even") -> np.ndarray:
    """
    Create bins for discretizing a continuous variable.  This function generates
    evenly spaced bins between `min_y` and `max_y` with num_bins-2 intervals, and
    generates an additional bin at the end to include the value exceeding max_y.

    Parameters
    ----------
    max_y : float
        Maximum value of y.
    min_y : float
        Minimum value of y.
    num_bins : int
        Number of bins.

    Returns
    -------
    bins : np.ndarray
        Array of bin edges.
    """
    if num_bins <= 1:
        raise ValueError("Number of bins must be greater than 1.")
    if max_y <= min_y:
        raise ValueError("Maximum value must be strictly greater than minimum value.")

    if algorithm != "even":
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'even'.")

    w = max_y / (num_bins - 1)
    return np.linspace(0.0, max_y + w, num_bins + 1)


def create_discretized_labels(bins: np.ndarray, num_risks: int, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Create discretized labels for survival analysis.

    Parameters
    ----------
    bins : np.ndarray
        Array of bin edges.
    num_risks : int
        Number of risks (or categories) to be assigned to each bin.
    t : np.ndarray
        Array of time values.
    e : np.ndarray
        Array of event indicators (1 for event, 0 for censored).

    Returns
    -------
    np.ndarray
        Array of discretized labels, where each label corresponds to a bin and a risk category.
    """

    e = e.astype(int)
    idx = np.searchsorted(bins, t.reshape(-1, 1), side="right")
    idx = np.clip(idx, 1, len(bins) - 1)
    return (idx.reshape(-1) - 1) * num_risks + e


def adjust_exact_observations(lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjust exact observations (where lb == ub) to be slightly less than the upper bound.
    This is necessary for the models that cannot handle exact observations.

    Parameters
    ----------
    lb : np.ndarray
        Array of lower bounds.
    ub : np.ndarray
        Array of upper bounds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Adjusted lower and upper bounds.
    """

    lb = lb.astype(float)
    ub = ub.astype(float)
    mask_exact = lb == ub
    diff = ub[~mask_exact] - lb[~mask_exact]
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        min_diff = 0.00001
    else:
        min_diff = min(diff.min(), 0.0001) / 10.0
    lb[mask_exact] -= min_diff
    return lb, ub

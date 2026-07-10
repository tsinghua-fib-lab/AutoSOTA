import numpy as np


def d_calibration(
    dist,
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    boundaries: np.ndarray | int | None = None,
) -> float:
    """
    Compute D-Calibration.

    Parameters
    ----------
    dist: distribution object
        Predicted distribution.
    observed_times : ndarray (float) of shape [batch_size]
        Observation time (event time or censored time)
    uncensored : ndarray (bool) of shape [batch_size]
        Indicator (censored (False) or uncensored (True))
    boundaries: ndarray (float) of shape [num_bin+1] or int or None
        Bin boundaries used to compute D-calibration.
        boundaries[0] = 0.0 and boundaries[-1] = 1.0.
        If boundaries is int, then boundaries = np.linspace(0.0, 1.0, boundaries + 1).
        If boundaries is None, then boundaries = np.linspace(0.0, 1.0, 11).

    Returns
    -------
    D-Calibration : float
        Value of D-Calibration.
    """

    if boundaries is None:
        boundaries = np.linspace(0.0, 1.0, 11)
    elif isinstance(boundaries, int):
        boundaries = np.linspace(0.0, 1.0, boundaries + 1)
    len_bin = boundaries[1:] - boundaries[:-1]
    num_bin = len(boundaries) - 1

    quantiles = dist.cdf(observed_times.reshape(-1, 1))

    # compute count_unc for uncensored data points
    count_unc = np.zeros((quantiles[uncensored].shape[0], num_bin))
    t = quantiles[uncensored].reshape(-1, 1)
    t_in_C = (boundaries[:-1] <= t) & (t < boundaries[1:])
    count_unc[t_in_C] += 1.0
    count_unc[:, -1] = 1.0 - np.sum(count_unc[:, :-1], 1)

    # compute count_cen for censored data points
    count_cen = np.zeros((quantiles[~uncensored].shape[0], num_bin))
    v = quantiles[~uncensored].reshape(-1, 1)
    v_in_C = (boundaries[:-1] <= v) & (v < boundaries[1:])
    denominator = np.clip(1.0 - v, 0.0001, 1.0)
    count_cen[v_in_C] += ((boundaries[1:] - v) / denominator)[v_in_C]
    v_leq_C = v < boundaries[:-1]
    count_cen[v_leq_C] += (len_bin / denominator)[v_leq_C]
    count_cen[:, -1] = 1.0 - np.sum(count_cen[:, :-1], 1)

    # compute square loss
    diff = (np.sum(count_unc, 0) + np.sum(count_cen, 0)) / quantiles.shape[0]
    diff -= len_bin
    return np.sum(diff * diff)


def ic_calibration(
    dist,
    lb: np.ndarray,
    ub: np.ndarray,
    p: float = 2.0,
    b: np.ndarray | None = None,
    EPS: float = 0.0001,
) -> float:
    """
    Interval-Censored Calibration (IC-Cal).

    Parameters
    ----------
    dist: distribution object
        Predicted distribution.
    lb : ndarray (float) of shape [batch_size]
        Lower bound of the interval.
    ub : ndarray (float) of shape [batch_size]
        Upper bound of the interval.
    p : float
        Power for the calibration error. Only p=2 is implemented.
    b : ndarray (float) or None
        Bin boundaries used to compute IC-Cal.
        If None, default boundaries are used.
    EPS : float
        Small value to avoid division by zero.

    Returns
    -------
    IC-Cal : float
        Value of IC-Cal.
    """

    if len(lb.shape) != 1:
        raise ValueError("lb must be of shape [batch_size]")
    if len(ub.shape) != 1:
        raise ValueError("ub must be of shape [batch_size]")
    if lb.shape[0] != ub.shape[0]:
        raise ValueError("lb and ub must have the same length")

    if p != 2.0:
        raise NotImplementedError("Only p=2 is implemented.")

    F_lb = dist.cdf(lb.reshape(-1, 1))
    F_ub = dist.cdf(ub.reshape(-1, 1))
    if b is None:
        temp01 = np.array([[0.0], [1.0]])
        u = np.unique(np.concatenate([temp01, F_lb, F_ub]))
    else:
        u = b

    mask = (F_ub - F_lb).reshape(-1) < EPS
    pi = np.zeros((len(lb), len(u)))
    if np.any(mask):
        pi[mask, :] = (u > F_lb[mask, :]).astype(float)
    if np.any(~mask):
        pi[~mask, :] = (u - F_lb[~mask, :]) / (F_ub[~mask] - F_lb[~mask])
    pi = np.clip(pi, 0.0, 1.0)
    diff = pi.mean(axis=0) - u
    a = diff[:-1]
    b = diff[1:]
    return np.sum(np.diff(u) * (a * a + a * b + b * b) / 3.0)

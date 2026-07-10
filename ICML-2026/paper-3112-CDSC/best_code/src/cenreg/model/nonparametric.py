import warnings

import numpy as np

from cenreg.distribution.cdf import CumulativeDist


def _set_ymin_ymax(
    y: np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[float, float]:
    y = y[np.isfinite(y)]
    temp_min = np.min(y)
    temp_max = np.max(y)
    if temp_min == temp_max:
        if temp_min == 0.0:
            temp_min = -0.1
            temp_max = 0.1
        elif temp_min > 0.0:
            temp_min *= 0.9
            temp_max *= 1.1
        else:
            temp_min *= 1.1
            temp_max *= 0.9
    else:
        width = temp_max - temp_min
        temp_min -= 0.1 * width
        temp_max += 0.1 * width
    if y_min is not None:
        temp_min = y_min
    if y_max is not None:
        temp_max = y_max
    return temp_min, temp_max


def _validate_weights(weights: np.ndarray, y: np.ndarray):
    if weights is None:
        return np.ones_like(y)
    if len(weights.shape) != 1:
        raise ValueError("weight must be one-dimensional array.")
    if weights.shape[0] != y.shape[0]:
        raise ValueError("weight and y must have the same length.")
    if np.any(weights < 0.0):
        raise ValueError("weight must be non-negative.")


def _validate_cdf_inputs(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
):
    if len(y.shape) != 1:
        raise ValueError("y must be one-dimensional array.")
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if y_min is not None:
        if y_min > np.min(y):
            raise ValueError("y_min must be less than or equal to min(y).")
    if y_max is not None:
        if y_max < np.max(y):
            raise ValueError("y_max must be greater than or equal to max(y).")
    if weights is not None:
        _validate_weights(weights, y)


def _adjust_bins(bins: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    if y_min is not None:
        if y_min > bins[0]:
            raise ValueError("y_min must be less than or equal to the minimum observed value.")
        elif y_min < bins[0]:
            bins = np.append(y_min, bins)
    if y_max is not None:
        if y_max < bins[-1]:
            raise ValueError("y_max must be greater than or equal to the maximum observed value.")
        elif y_max > bins[-1]:
            bins = np.append(bins, y_max)
    return bins


def empirical_cdf_estimator(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> CumulativeDist:
    """
    Compute CDF based on y and weight.
    The observed values y are weighted by weight.

    Parameters
    -------
    y : np.ndarray
        One-dimensional ndarray containing observed values.
    weights : np.ndarray | None
        One-dimensional ndarray containing non-negative weight for each value in y.
        If None, all weights are set to 1.0.
    y_min : float | None
        The lower bound for the CDF. If None, it is set to the minimum observed value.
    y_max : float | None
        The upper bound for the CDF. If None, it is set to the maximum observed value.

    Returns
    -------
    dist : CumulativeDist
        Empirical CDF object.
    """

    # Input validation
    _validate_cdf_inputs(y, weights, y_min, y_max)
    if weights is None:
        weights = np.ones_like(y)

    # Set y_min and y_max if not provided
    y_min, y_max = _set_ymin_ymax(y, y_min, y_max)

    # Compute empirical CDF
    bins, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv, weights)
    s = weights.sum()
    if s <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    p = np.clip(counts / s, 0.0, 1.0)
    cum_p = np.cumsum(p)
    cum_p = np.append(0.0, cum_p)  # CDF starts from 0.0
    cum_p[-1] = 1.0  # Ensure last value is exactly 1.0

    # Adjust bins for y_min and y_max
    bins = _adjust_bins(bins, y_min, y_max)

    return CumulativeDist(b=bins, cum_p=cum_p, interpolate="right")


def _validate_kaplan_meier_inputs_weights(weights: np.ndarray, observed_times: np.ndarray):
    if len(weights.shape) != 1:
        raise ValueError("weights must be one-dimensional array.")
    if observed_times.shape[0] != weights.shape[0]:
        raise ValueError("observed_times and weights must have the same length.")


def _validate_kaplan_meier_inputs(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    weights: np.ndarray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
):
    if len(observed_times.shape) != 1:
        raise ValueError("observed_times must be one-dimensional array.")
    if len(uncensored.shape) != 1:
        raise ValueError("uncensored must be one-dimensional array.")
    if observed_times.shape[0] != uncensored.shape[0]:
        raise ValueError("observed_times and uncensored must have the same length.")
    if y_min is not None:
        if y_min > np.min(observed_times):
            raise ValueError("y_min must be less than or equal to min(observed_times).")
    if y_max is not None:
        if y_max < np.max(observed_times):
            raise ValueError("y_max must be greater than or equal to max(observed_times).")
    if weights is not None:
        _validate_kaplan_meier_inputs_weights(weights, observed_times)


def kaplan_meier_estimator(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    weights: np.ndarray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> CumulativeDist:
    """
    Compute Kaplan-Meier estimator.

    Parameters
    ----------
    observed_times : np.ndarray
        Observed times (both censored and uncensored).
    uncensored : np.ndarray
        Indicator for uncensored data (1: uncensored, 0: censored).
    weights : np.ndarray | None
        Weights for each data point.
    y_min : float | None
        Minimum value for the EmpiricalCDF.  If None, y_min is set to 0.0.
    y_max : float | None
        Maximum value for the EmpiricalCDF.  If None, y_max is set to observed_times.max().

    Returns
    -------
    dist : CumulativeDist
        Cumulative distribution function.
    """

    if not np.issubdtype(observed_times.dtype, np.floating):
        warnings.warn("observed_times is not float, converting to float.", stacklevel=2)
        observed_times = observed_times.astype(float)
    _validate_kaplan_meier_inputs(observed_times, uncensored, weights, y_min, y_max)

    uncensored = uncensored.astype(int)
    if np.sum(uncensored) == 0:
        raise ValueError("At least one data point must be uncensored.")
    if weights is None:
        weights = np.ones_like(observed_times)

    # sort based on uncensored and observed_times
    temp = np.concatenate(
        [
            observed_times.reshape(-1, 1),
            uncensored.reshape(-1, 1),
            weights.reshape(-1, 1),
        ],
        axis=1,
    )
    temp = temp[np.argsort(temp[:, 1])[::-1]]
    temp = temp[np.argsort(temp[:, 0])]

    # count alive and dead
    num_alive = np.sum(temp[:, 2]) - np.cumsum(temp[:, 2]) + temp[:, 2]
    temp = np.concatenate([temp, num_alive.reshape(-1, 1)], axis=1)
    dead = temp[temp[:, 1] == 1]
    cumsum_death = np.concatenate([[0.0], np.cumsum(dead[:, 2])])
    cut_index = np.concatenate(([True], dead[1:, 0] != dead[:-1, 0], [True])).nonzero()[0]
    num_death = cumsum_death[cut_index[1:]] - cumsum_death[cut_index[:-1]]
    num_alive = dead[cut_index[:-1], 3]
    time_points = dead[cut_index[:-1], 0]

    # create CDF
    b = time_points
    rate = 1.0 - num_death / num_alive
    survival_rates = np.cumprod(rate)
    if y_min is None:
        y_min = 0.0
    if y_min < b[0]:
        b = np.append(y_min, b)
        survival_rates = np.append(1.0, survival_rates)
    if y_max is None:
        y_max = np.max(observed_times)
    if y_max > b[-1]:  # last observation is censored
        b = np.append(b, y_max)
    else:
        survival_rates = survival_rates[:-1]
    dist = CumulativeDist(b=b, cum_p=1.0 - survival_rates, interpolate="right")
    # dist.alive = num_alive
    # dist.dead = num_death
    return dist


def _binary_search_F_single(F_lb: float, F_ub: float, G_cur: float, copula, target: float, eps=0.0001):
    F_cur = (F_lb + F_ub) / 2.0
    if F_ub - F_lb < eps:
        return F_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        F_lb = F_cur
    else:
        F_ub = F_cur
    return _binary_search_F_single(F_lb, F_ub, G_cur, copula, target)


def _binary_search_G_single(G_lb: float, G_ub: float, F_cur: float, copula, target: float, eps=0.0001):
    G_cur = (G_lb + G_ub) / 2.0
    if G_ub - G_lb < eps:
        return G_cur
    u = np.array([[F_cur, G_cur]])
    temp = 1.0 - F_cur - G_cur + copula.cdf(u)
    if temp > target:
        G_lb = G_cur
    else:
        G_ub = G_cur
    return _binary_search_G_single(G_lb, G_ub, F_cur, copula, target)


def _process_zheng_klein_loop(temp: np.ndarray, copula, total_weight: float):
    """Process main loop for zheng_klein_estimator."""
    f = 0.0
    g = 0.0
    times = []
    survival_rates = []
    cum_weight = 0.0

    for i in range(temp.shape[0]):
        cum_weight += temp[i, 2]
        if i + 1 < temp.shape[0] and temp[i, 0] == temp[i + 1, 0] and temp[i, 1] == temp[i + 1, 1]:
            continue  # skip duplicates
        target = 1.0 - cum_weight / total_weight
        if temp[i, 1] > 0:  # uncensored
            f = _binary_search_F_single(f, 1.0, g, copula, target)
            times.append(temp[i, 0])
            survival_rates.append(1.0 - f)
        else:
            g = _binary_search_G_single(g, 1.0, f, copula, target)

    return times, survival_rates


def _adjust_cdf_bounds(
    b: np.ndarray, survival_rates: np.ndarray, y_min: float | None, y_max: float | None, observed_times: np.ndarray
):
    """Adjust CDF bounds with y_min and y_max."""
    if y_min is None:
        y_min = 0.0
    if y_min < b[0]:
        b = np.append(y_min, b)
        survival_rates = np.append(1.0, survival_rates)

    if y_max is None:
        y_max = np.max(observed_times)
    if y_max > b[-1]:  # last observation is censored
        b = np.append(b, y_max)
    else:
        survival_rates = survival_rates[:-1]

    return b, survival_rates


def zheng_klein_estimator(
    observed_times: np.ndarray,
    uncensored: np.ndarray,
    copula,
    weights: np.ndarray | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> CumulativeDist:
    """
    Compute copula-graphic estimator proposed by Zheng and Klein.
    This method receives any copula.

    Parameters
    ----------
    observed_times : np.ndarray
        Observed times (both censored and uncensored).
    uncensored : np.ndarray
        Indicator for uncensored data (1: uncensored, 0: censored).
    copula : object
        Copula function.
    weights : np.ndarray | None
        Weights for each data point.
    y_min : float | None
        Minimum value for the EmpiricalCDF.  If None, y_min is set to 0.0.
    y_max : float | None
        Maximum value for the EmpiricalCDF.  If None, y_max is set to observed_times.max().

    Returns
    -------
    dist : CumulativeDist
        Cumulative distribution function.
    """

    if len(observed_times.shape) != 1:
        raise ValueError("observed_times must be one-dimensional array.")
    if len(uncensored.shape) != 1:
        raise ValueError("uncensored must be one-dimensional array.")
    if observed_times.shape[0] != uncensored.shape[0]:
        raise ValueError("observed_times and uncensored must have the same length.")
    if not np.issubdtype(observed_times.dtype, np.floating):
        warnings.warn("observed_times is not float, converting to float.", stacklevel=2)
        observed_times = observed_times.astype(float)

    uncensored = uncensored.astype(int)
    if np.sum(uncensored) == 0:
        raise ValueError("At least one data point must be uncensored.")
    if weights is None:
        weights = np.ones_like(observed_times)
    else:
        if len(weights.shape) != 1:
            raise ValueError("weights must be one-dimensional array.")
        if observed_times.shape[0] != weights.shape[0]:
            raise ValueError("observed_times and weights must have the same length.")
        if not np.issubdtype(weights.dtype, np.floating):
            warnings.warn("weights is not float, converting to float.", stacklevel=2)
            weights = weights.astype(float)

    # sort based on uncensored and observed_times
    temp = np.concatenate(
        [
            observed_times.reshape(-1, 1),
            uncensored.reshape(-1, 1),
            weights.reshape(-1, 1),
        ],
        axis=1,
    )
    temp = temp[np.argsort(temp[:, 1])[::-1]]
    temp = temp[np.argsort(temp[:, 0])]

    # compute
    total_weight = np.sum(weights)
    times, survival_rates = _process_zheng_klein_loop(temp, copula, total_weight)

    # create CDF
    b = np.array(times)
    survival_rates = np.array(survival_rates)
    b, survival_rates = _adjust_cdf_bounds(b, survival_rates, y_min, y_max, observed_times)

    dist = CumulativeDist(b=b, cum_p=1.0 - survival_rates, interpolate="right")
    return dist


def _validate_interval_inputs(
    lb: np.ndarray,
    ub: np.ndarray,
    weights: np.ndarray | None = None,
):
    if len(lb.shape) != 1 or len(ub.shape) != 1:
        raise ValueError("lb and ub must be one-dimensional arrays.")
    if lb.shape[0] != ub.shape[0]:
        raise ValueError("lb and ub must have the same length.")
    if np.any(lb == np.inf):
        raise NotImplementedError("lb containing np.inf is not supported.")
    if np.any(ub == -np.inf):
        raise NotImplementedError("ub containing -np.inf is not supported.")
    lb = lb.astype(float)
    ub = ub.astype(float)
    lb = np.where(np.isnan(lb), -np.inf, lb)
    ub = np.where(np.isnan(ub), np.inf, ub)
    if np.any(lb > ub):
        raise ValueError("Each element of lb must be less than or equal to the corresponding element of ub.")

    if weights is None:
        weights = np.ones_like(lb).astype(float).reshape(-1, 1)
    else:
        weights = weights.astype(float).reshape(-1, 1)
        if len(weights.shape) != 1:
            raise ValueError("weights must be a one-dimensional array.")
        if lb.shape[0] != weights.shape[0]:
            raise ValueError("weights must have the same length as lb and ub.")

    return lb, ub, weights


def turnbull_estimator(
    lb: np.ndarray,
    ub: np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
    weights: np.ndarray | None = None,
    eps: float = 1e-8,
    max_iter: int = 100,
):
    """
    Turnbull estimator for interval-censored data.

    Parameters
    ----------
    lb : np.ndarray
        Lower bounds of observed intervals.
    ub : np.ndarray
        Upper bounds of observed intervals.
    y_min : float | None
        Minimum value for the CDF.
    y_max : float | None
        Maximum value for the CDF.
    weights : np.ndarray | None
        Weights for each data point.
    eps : float
        Convergence threshold.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    cdf : cenreg.distribution.cdf.CumulativeDist
        Cumulative distribution function object.
    """

    lb, ub, weights = _validate_interval_inputs(lb, ub, weights)
    if np.any(lb == ub):
        raise NotImplementedError("Exact observations (lb == ub) are not supported in turnbull_estimator.")

    # Set y_min and y_max if not provided
    vals = np.concatenate([lb, ub])
    y_min, y_max = _set_ymin_ymax(vals, y_min, y_max)

    # initialize distribution
    omega = np.unique(vals[np.isfinite(vals)])
    omega = np.concatenate([[-np.inf], omega, [np.inf]])
    num_bin = len(omega) - 1
    s = np.full((num_bin,), 1.0 / num_bin)

    # iterate EM algorithm
    n = lb.shape[0]
    batch_size = int(1000000 / num_bin + 1)
    for _ in range(max_iter):
        pi = np.zeros_like(s)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mask_l = (omega[:-1] >= lb[start:end].reshape(-1, 1)).astype(float)
            mask_r = (omega[1:] <= ub[start:end].reshape(-1, 1)).astype(float)
            mask = mask_l * mask_r
            temp = mask * s.reshape(1, -1)
            mu = temp / temp.sum(axis=1).reshape(-1, 1)
            mu *= weights[start:end]
            pi += mu.sum(axis=0)
        pi /= weights.sum()
        diff = pi - s
        loss = np.mean(diff * diff)
        s = pi
        if loss < num_bin * eps:
            break

    # create CDF
    omega[0] = y_min
    omega[-1] = y_max
    cdf = CumulativeDist(b=omega, p=s, interpolate="right")
    return cdf


def li_watkins_yu_estimator(
    lb: np.ndarray,
    ub: np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
    weights: np.ndarray | None = None,
    eps: float = 1e-8,
    max_iter: int = 100,
):
    """
    Li-Watkins-Yu estimator for interval-censored data.

    Parameters
    ----------
    lb : np.ndarray
        Lower bounds of observed intervals.
    ub : np.ndarray
        Upper bounds of observed intervals.
    y_min : float | None
        Minimum value for the CDF.
    y_max : float | None
        Maximum value for the CDF.
    weights : np.ndarray | None
        Weights for each data point.
    eps : float
        Convergence threshold.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    cdf : cenreg.distribution.cdf.CumulativeDist
        Cumulative distribution function object.
    """

    lb, ub, weights = _validate_interval_inputs(lb, ub, weights)

    # Set y_min and y_max if not provided
    vals = np.concatenate([lb, ub])
    y_min, y_max = _set_ymin_ymax(vals, y_min, y_max)

    # initialize distribution
    omega = np.unique(vals[np.isfinite(vals)])
    omega = np.concatenate([[y_min], omega, [y_max]])
    num_bin = len(omega) - 1
    s = np.full((num_bin,), 1.0 / num_bin)
    dist = CumulativeDist(b=omega, p=s, interpolate="left")

    # iterate EM algorithm
    n = lb.shape[0]
    batch_size = int(1000000 / num_bin + 1)
    for _ in range(max_iter):
        # update distribution
        F_t = dist.cdf(omega.reshape(-1, 1)).reshape(1, -1)
        F_lb = dist.cdf(lb.reshape(-1, 1))
        F_ub = dist.cdf(ub.reshape(-1, 1))
        pi = np.zeros_like(omega)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mask = (F_ub[start:end] - F_lb[start:end]).reshape(-1) < 1e-7
            temp = np.zeros((mask.shape[0], F_t.shape[1]))
            if np.any(mask):
                temp[mask, :] = (F_t >= F_lb[start:end][mask, :]).astype(float)
            if np.any(~mask):
                temp[~mask, :] = (F_t - F_lb[start:end][~mask, :]) / (F_ub[start:end][~mask] - F_lb[start:end][~mask])
            temp = np.clip(temp, 0.0, 1.0) * weights[start:end]
            pi += temp.sum(axis=0)
        pi_mean = pi / weights.sum()
        dist = CumulativeDist(b=omega, cum_p=pi_mean[1:], interpolate="left")

        # compute loss
        diff = pi_mean - F_t.reshape(-1)
        loss = np.mean(diff * diff)
        if loss < eps:
            break

    return dist

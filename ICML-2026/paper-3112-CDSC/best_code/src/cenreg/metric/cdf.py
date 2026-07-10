import warnings

import numpy as np

from cenreg.model.nonparametric import kaplan_meier_estimator

'''
def negative_loglikelihood(
    dist,
    observed_times: np.ndarray,
    uncensored: np.ndarray | None = None,
    boundaries: np.ndarray | None = None,
    simplified: bool = True,
    eps: float = 0.0001,
) -> np.ndarray:
    """
    Compute negative log-likelihood (NLL).

    Parameters
    ----------
    dist: distribution object
        Prediction results.
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    uncensored : ndarray(bool) or None
        Indicator (censored (False) or uncensored (True))
        Array shape is [batch_size].
    boundaries : ndarray or None
        Boundaries of the prediction to be evaluated.
        The first element must be at most the smallest observed time.
        The last element must be strictly larger than the largest observed time.
        If None, dist.b is used.
    simplified : bool
        Use simplified version for censored data points.
    eps : float
        Small positive value for numerical stability.

    Returns
    -------
    NLL : ndarray
        Value of negative log-likelihood.
    """

    if not simplified:
        raise NotImplementedError("Only simplified version is implemented.")

    # set default values
    if uncensored is None:
        uncensored = np.ones(observed_times.shape, dtype=bool)
    else:
        uncensored = uncensored.astype(bool)
    if boundaries is None:
        try:
            boundaries = dist.b
        except AttributeError as err:
            raise ValueError("boundaries must be provided if pred does not have b.") from err

    # check idx
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right")
    idx = idx.reshape(-1) - 1
    if np.any(idx < 0):
        raise ValueError("observed_times must be at least boundaries[0]")
    if np.any(idx >= len(boundaries) - 1):
        raise ValueError("observed_times must be strictly smaller than boundaries[-1]")

    # compute loss
    loss = np.zeros(observed_times.shape)
    b_lb = boundaries[idx]
    b_ub = boundaries[idx + 1]
    F_lb = dist.cdf(b_lb.reshape(-1, 1)).reshape(-1)
    F_ub = dist.cdf(b_ub.reshape(-1, 1)).reshape(-1)
    loss[uncensored] = -np.log(F_ub[uncensored] - F_lb[uncensored] + eps)
    last_idx = idx == len(boundaries) - 2
    loss[~uncensored & ~last_idx] = -np.log(1.0 - F_ub[~uncensored & ~last_idx] + eps)
    loss[~uncensored & last_idx] = -np.log(1.0 - F_lb[~uncensored & last_idx] + eps)
    return loss.sum()


def negative_log_likelihood_interval(
    dist,
    lb: np.ndarray,
    ub: np.ndarray,
    eps: float = 0.0001,
) -> np.ndarray:
    """
    Compute Negative log-likelihood for interval-censored data.

    Parameters
    ----------
    dist: distribution object
        Prediction results.
    lb: ndarray
        Lower bounds of the intervals.
    ub: ndarray
        Upper bounds of the intervals.
    eps: float
        Small positive value for numerical stability.

    Returns
    -------
    loss : ndarray
        Value of negative log-likelihood.
    """

    if np.any(lb >= ub):
        raise ValueError("lb must be strictly smaller than ub for all data points.")

    mask = (dist.b >= lb.reshape(-1, 1)) & (dist.b < ub.reshape(-1, 1))
    p = np.diff(dist.cum_p, axis=0)
    loss = -np.log(np.sum(p * mask.astype(np.float32), axis=1) + eps)
    return loss.sum()
'''


def _negative_log_likelihood(
    dist,
    lb: np.ndarray,
    ub: np.ndarray,
    proportional: bool = True,
    eps: float = 0.0001,
) -> np.ndarray:
    """
    Compute Negative log-likelihood.

    Parameters
    ----------
    dist: predicted distribution

    lb: ndarray of shape [batch_size]
        lower bound of the interval-censored data

    ub: ndarray of shape [batch_size]
        upper bound of the interval-censored data

    proportional: bool
        whether to distribute the probability mass proportionally for censored data

    eps: float
        small value to avoid numerical issues

    Returns
    -------
    loss : ndarray of shape [batch_size]
    """

    y_bins = dist.b
    idx_lb = np.searchsorted(y_bins, lb.reshape(-1, 1), side="right")
    idx_lb = np.clip(idx_lb, 1, len(y_bins) - 1)
    idx_ub = np.searchsorted(y_bins, ub.reshape(-1, 1), side="left")
    idx_ub = np.clip(idx_ub, 1, len(y_bins) - 1)
    b_lb = y_bins[idx_lb - 1]
    b_ub = y_bins[idx_ub]
    F_lb = dist.cdf(b_lb)
    F_ub = dist.cdf(b_ub)

    if proportional:
        interval = (idx_lb + 1 == idx_ub)[:, 0]
        F_lb[interval] = dist.cdf(lb.reshape(-1, 1))[interval]
        F_ub[interval] = dist.cdf(ub.reshape(-1, 1))[interval]

    loss = -np.log(F_ub - F_lb + eps)
    return loss.reshape(-1)


def negative_log_likelihood_survival(
    dist,
    y: np.ndarray,
    uncensored: np.ndarray | None = None,
    proportional: bool = True,
    eps: float = 0.0001,
) -> np.ndarray:
    """
    Compute Negative log-likelihood for survival data.

    Parameters
    ----------
    dist: predicted distribution
        Must be an instance of cenreg.distribution.cdf.CumulativeDist with linear interpolation.

    y: ndarray of shape [batch_size]
        observed times

    uncensored: ndarray of shape [batch_size]
        boolean array indicating whether the event was observed (True) or censored (False)

    proportional: bool
        whether to distribute the probability mass proportionally for censored data

    eps: float
        small value to avoid numerical issues

    Returns
    -------
    loss : ndarray of shape [batch_size]
    """

    if len(y.shape) != 1:
        raise ValueError("y must be of shape [batch_size]")
    if uncensored is None:
        uncensored = np.ones(y.shape, dtype=bool)
    if dist.interpolate != "linear":
        raise Warning("dist must be an instance of cenreg.distribution.cdf.CumulativeDist with linear interpolation.")

    lb = y.astype(float)
    ub = y.astype(float)
    ub[~uncensored] = np.inf
    return _negative_log_likelihood(dist, lb, ub, proportional, eps)


def negative_log_likelihood_interval(
    dist,
    lb: np.ndarray,
    ub: np.ndarray,
    proportional: bool = True,
    eps: float = 0.0001,
) -> np.ndarray:
    """
    Compute Negative log-likelihood for interval-censored data.

    Parameters
    ----------
    dist: predicted distribution
        Must be an instance of cenreg.distribution.cdf.CumulativeDist with linear interpolation.

    lb: ndarray of shape [batch_size]
        lower bounds of the intervals

    ub: ndarray of shape [batch_size]
        upper bounds of the intervals

    proportional: bool
        whether to distribute the probability mass proportionally for censored data

    eps: float
        small value to avoid numerical issues

    Returns
    -------
    loss : ndarray of shape [batch_size]
    """

    if len(lb.shape) != 1:
        raise ValueError("lb must be of shape [batch_size]")
    if len(ub.shape) != 1:
        raise ValueError("ub must be of shape [batch_size]")
    if len(lb) != len(ub):
        raise ValueError("lb and ub must have the same length")
    if dist.interpolate != "linear":
        raise Warning("dist must be an instance of cenreg.distribution.cdf.CumulativeDist with linear interpolation.")

    return _negative_log_likelihood(dist, lb, ub, proportional, eps)


def brier(
    dist,
    y: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    """
    Compute the (original) Brier score.

    Parameters
    ----------
    dist: predicted distribution

    y: Array of shape [batch_size]

    y_bins: Array of shape [num_bins+1]

    Returns
    -------
    loss : Array of shape [batch_size]
    """

    if len(y.shape) != 1:
        raise ValueError("y must be of shape [batch_size]")
    if len(y_bins.shape) != 1:
        raise ValueError("y_bins must be of shape [num_bins+1]")

    F_pred = dist.cdf(y_bins)
    if len(F_pred.shape) == 1:
        F_pred = F_pred.reshape(1, -1)
    pred = F_pred[:, 1:] - F_pred[:, :-1]
    y = y.reshape(-1, 1)
    idx = np.searchsorted(y_bins, y, side="right") - 1
    onehot = np.identity(len(y_bins) - 1)[idx.reshape(-1)]
    diff = onehot - pred
    return (diff * diff).sum(axis=1)


def ranked_probability_score(
    dist,
    y: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    """
    Compute the ranked probability score.

    Parameters
    ----------
    dist: predicted distribution

    y: Array of shape [batch_size]

    y_bins: Array of shape [num_col]

    Returns
    -------
    loss : Array of shape [batch_size]
    """

    if len(y.shape) != 1:
        raise ValueError("y must be of shape [batch_size]")
    if len(y_bins.shape) != 1:
        raise ValueError("y_bins must be of shape [num_col]")

    F_pred = dist.cdf(y_bins[1:-1])
    y = y.reshape(-1, 1)
    idx = np.searchsorted(y_bins, y, side="right") - 1
    num_cls = len(y_bins) - 1
    label = np.triu(np.ones((num_cls, num_cls)))[idx.reshape(-1)]
    diff = label[:, :-1] - F_pred
    return (diff * diff).sum(axis=1)


def nll_sc(
    list_dist,
    observed_times: np.ndarray,
    events: np.ndarray,
    survival_copula,
    eps: float = 0.0001,
):
    """
    Compute Negative Log-Likelihood based on Survival Copula (NLL-SC).
    """

    if len(observed_times.shape) != 1:
        raise ValueError("observed_times must be of shape [batch_size]")
    if len(events.shape) != 1:
        raise ValueError("events must be of shape [batch_size]")
    if observed_times.shape[0] != events.shape[0]:
        raise ValueError("observed_times and events must have the same length")
    if not np.issubdtype(observed_times.dtype, np.floating):
        warnings.warn("observed_times is not float, converting to float.", stacklevel=2)
        observed_times = observed_times.astype(float)

    Sl_list = []
    Sr_list = []
    diff = np.zeros(observed_times.shape[0])
    for k in range(len(list_dist)):
        mask = events == k

        b = list_dist[k].b
        idx = np.searchsorted(b, observed_times, side="right")
        idx = np.clip(idx, 1, len(b) - 1)
        left = b[idx - 1]
        right = b[idx]
        diff[mask] = np.clip(right[mask] - left[mask], eps, None)

        Sl = np.zeros(observed_times.shape[0])
        Sr = np.zeros(observed_times.shape[0])
        Sl[mask] = 1.0 - list_dist[k].cdf(left.reshape(-1, 1)).reshape(-1)[mask]
        Sr[mask] = 1.0 - list_dist[k].cdf(right.reshape(-1, 1)).reshape(-1)[mask]
        S_pred = 1.0 - list_dist[k].cdf(observed_times.reshape(-1, 1)).reshape(-1)[~mask]
        Sl[~mask] = S_pred
        Sr[~mask] = S_pred
        Sl_list.append(Sl.reshape(-1, 1))
        Sr_list.append(Sr.reshape(-1, 1))

    Sl = np.concatenate(Sl_list, axis=1)
    Sr = np.concatenate(Sr_list, axis=1)
    sl = survival_copula.cdf(Sl)
    sr = survival_copula.cdf(Sr)
    log_fl = np.log(np.clip((sl - sr) / diff, eps, None))
    return -log_fl


def km_calibration(
    dist,
    observed_times: np.ndarray,
    uncensored: np.ndarray | None = None,
    y_bins: np.ndarray | None = None,
    eps: float = 0.0001,
) -> float:
    """
    Compute KM-Calibration

    Parameters
    ----------
    dist: distribution object
        Prediction results.
    observed_times : ndarray
        Observation time (event time or censored time)
        Array shape is [batch_size].
    uncensored : ndarray
        Censored (False) or uncensored (True)
        Array shape is [batch_size].
    y_bins: ndarray
        Bins for the prediction to be evaluated.
        Each element in observed_times must be STRICTLY smaller than y_bins[-1]
    eps : float
        Small positive value for numerical stability

    Returns
    -------
    KM-Calibration : float
        A non-negative float number.
    """

    # set default values
    if uncensored is None:
        uncensored = np.ones(observed_times.shape, dtype=bool)
    else:
        uncensored = uncensored.astype(bool)
    if y_bins is None:
        try:
            y_bins = dist.b
        except AttributeError as err:
            raise ValueError("y_bins must be provided if dist does not have attribute 'b'.") from err

    # compute Kaplan-Meier distribution and prediction distribution
    km = kaplan_meier_estimator(observed_times, uncensored)
    last_idx = np.searchsorted(y_bins, km.b[-1], side="right") - 1
    F_km = km.cdf(y_bins)
    F_km[-1] = 1.0
    f_km = F_km[1:] - F_km[:-1]
    F_pred = dist.cdf(y_bins.reshape(-1))
    f_pred_mean = np.mean(F_pred[:, 1:] - F_pred[:, :-1], 0)

    # compute logarithmic loss for KM valid region
    log_empirical = np.log(f_km[:last_idx] + eps)
    log_mean_pred = np.log(f_pred_mean[:last_idx] + eps)
    loss_valid = np.sum(f_km[:last_idx] * (log_empirical - log_mean_pred))

    # compute logarithmic loss for KM invalid region
    sum_empirical = np.sum(f_km[last_idx:])
    log_sum_empirical = np.log(sum_empirical + eps)
    log_sum_pred = np.log(np.sum(f_pred_mean[last_idx:]) + eps)
    loss_invalid = sum_empirical * (log_sum_empirical - log_sum_pred)

    return loss_valid + loss_invalid

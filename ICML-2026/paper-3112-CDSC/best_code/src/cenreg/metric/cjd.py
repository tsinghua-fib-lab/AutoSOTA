import numpy as np


def brier(
    observed_times: np.ndarray,
    events: np.ndarray,
    num_risks: int,
    f_pred: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    """
    Compute Brier score for censoree joint distribution.

    Parameters
    ----------
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    events : ndarray
        Event index (0 usually corresponds to censoring)
        Array shape is [batch_size].
    num_risks : int
        Number of risks.
    f_pred : ndarray
        Estimated joint distribution.
        Array shape is [batch_size, num_bin*num_risks].
    boundaries: ndarray
        Bin boundaries used to represent f_pred.
        boundaries[0] = 0.0 and boundaries[-1] = max_time.
        Array shape is [num_bin+1].
    Returns
    -------
    Brier : float
        Value of Brier score.
    """

    events = events.astype(int)
    num_bin = len(boundaries) - 1
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right").reshape(-1)
    idx = events * num_bin + (idx - 1)

    num_bin = len(boundaries) - 1
    onehot = np.identity(num_bin * num_risks)[idx].astype(float)
    diff = onehot - f_pred
    return np.sum(diff * diff, axis=1).reshape(-1)


def negative_loglikelihood(
    observed_times: np.ndarray,
    events: np.ndarray,
    f_pred: np.ndarray,
    boundaries: np.ndarray,
    epsilon: float = 0.0001,
) -> np.ndarray:
    """
    Compute negative log-likelihood (NLL).

    Parameters
    ----------
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    events : ndarray
        Event index (0 usually corresponds to censoring)
        Array shape is [batch_size].
    f_pred : ndarray
        Estimated joint distribution.
        Array shape is [batch_size, num_bin*num_risks].
    boundaries: ndarray
        Bin boundaries used to represent f_pred.
        boundaries[0] = 0.0 and boundaries[-1] = max_time.
        Array shape is [num_bin+1].
    epsilon : float
        Small value to avoid log(0).
    Returns
    -------
    NLL : ndarray
        Per-sample negative log-likelihood values.
        Array shape is [batch_size].
    """
    events = events.astype(int).reshape(-1, 1)
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right")
    num_bin = len(boundaries) - 1
    idx = events * num_bin + (idx - 1)
    p = np.take_along_axis(f_pred, idx, 1)
    return -np.log(np.clip(p, epsilon, 1.0)).reshape(-1)


def ranked_probability_score(
    observed_times: np.ndarray,
    events: np.ndarray,
    num_risks: int,
    f_pred: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    """
    Ranked probability score (RPS).

    Parameters
    ----------
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    events : ndarray
        Event index (0 usually corresponds to censoring)
        Array shape is [batch_size].
    num_risks : int
        Number of risks.
    f_pred : ndarray
        Estimated joint distribution.
        Array shape is [batch_size, num_bin*num_risks].
    boundaries: ndarray
        Bin boundaries used to represent f_pred.
        boundaries[0] = 0.0 and boundaries[-1] = max_time.
        Array shape is [num_bin+1].
    Returns
    -------
    RPS : float
        Value of ranked probability score.
    """
    num_bin = len(boundaries) - 1
    num_cls = num_bin * num_risks
    triu = np.triu(np.ones((num_cls, num_cls)))

    events = events.astype(int).reshape(-1, 1)
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right")
    idx = events * num_bin + (idx - 1)
    label = triu[idx.reshape(-1)]

    F_pred = np.cumsum(f_pred, axis=1)
    diff = label - F_pred
    return (diff * diff).sum(axis=1)


def kolmogorov_smirnov_calibration_error(
    observed_times: np.ndarray,
    events: np.ndarray,
    f_pred: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    """
    Sum of Kolmogorov-Sminov calibration error.

    Parameters
    ----------
    observed_times : ndarray
        Observed times (event times or censoring times).
        Array shape is [batch_size].
    events : ndarray
        Event index (0 usually corresponds to censoring)
        Array shape is [batch_size].
    f_pred : ndarray
        Estimated joint distribution.
        Array shape is [batch_size, num_bin*num_risks].
    boundaries: ndarray
        Bin boundaries used to represent f_pred.
        boundaries[0] = 0.0 and boundaries[-1] = max_time.
        Array shape is [num_bin+1].
    Returns
    -------
    KS : float
        Sum of Kolmogorov-Sminov calibration error.
    """

    if len(observed_times.shape) != 1:
        raise ValueError("observed_times must be of shape [batch_size]")
    if len(events.shape) != 1:
        raise ValueError("events must be of shape [batch_size]")
    if len(f_pred.shape) != 2:
        raise ValueError("f_pred must be of shape [batch_size, num_bin*num_risks]")
    if len(boundaries.shape) != 1:
        raise ValueError("boundaries must be of shape [num_bin+1]")
    if f_pred.shape[0] != observed_times.shape[0]:
        raise ValueError("f_pred and observed_times must have the same length")

    events = events.astype(int).reshape(-1, 1)
    idx = np.searchsorted(boundaries, observed_times.reshape(-1, 1), side="right")
    num_bin = len(boundaries) - 1
    idx = events * num_bin + (idx - 1)

    ret = 0.0
    for k in range(f_pred.shape[1]):
        pred = f_pred[:, k]
        ind = np.argsort(pred)
        pred_sorted = pred[ind]
        idx_sorted = idx[ind]
        flag = idx_sorted == k
        pred_cumsum = np.cumsum(pred_sorted) / f_pred.shape[0]
        flag_cumsum = np.cumsum(flag) / f_pred.shape[0]
        ks = np.max(np.abs(pred_cumsum - flag_cumsum))
        ret += ks
    return ret

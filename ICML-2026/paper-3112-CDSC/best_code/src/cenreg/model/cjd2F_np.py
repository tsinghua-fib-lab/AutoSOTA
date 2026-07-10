import numpy as np


def _integral(jd_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the integral of the joint distribution predictions to estimate the cumulative distribution function (CDF).
    This function is based on this paper: A. Tsiatis, A nonidentifiability aspect of the problem of competing risks,
    1975.

    Parameters
    ----------
    jd_pred: np.ndarray of shape [batch_size, num_risks, num_bin_predictions]

    Returns
    -------
    F_pred: estimated CDF.
        np.ndarray of shape [batch_size, num_risks, num_bin_predictions+1]
    """
    if len(jd_pred.shape) != 3:
        raise ValueError(f"Expected jd_pred to have 3 dimensions, but got {len(jd_pred.shape)} dimensions.")

    # w = boundaries[1:] - boundaries[:-1]
    Q = np.cumsum(jd_pred, axis=2)
    Q = np.concatenate([np.zeros((Q.shape[0], Q.shape[1], 1)), Q], axis=2)
    Q = Q[:, :, -1].reshape(Q.shape[0], Q.shape[1], 1) - Q[:, :, :]
    denominator = Q[:, :, :-1].sum(axis=1)
    denominator = np.clip(denominator, 0.000001, 1.0)
    h = -jd_pred / denominator.reshape(denominator.shape[0], 1, denominator.shape[1])
    H = np.exp(np.cumsum(h, axis=2))
    ret = np.concatenate([np.zeros((H.shape[0], H.shape[1], 1)), 1.0 - H], axis=2)
    ret[:, :, -1] = 1.0
    return ret


def cjd2surv(jd_pred: np.ndarray, algorithm: str = "integral") -> np.ndarray:
    """
    Estimate the survival function from the CJD representation.

    Parameters
    ----------
    jd_pred: np.ndarray of shape [batch_size, num_risks, num_bin_predictions]
    algorithm: str, optional
        Algorithm to use for estimation. Currently only "integral" is supported.

    Returns
    -------
    F_pred: estimated CDF.
        np.ndarray of shape [batch_size, num_risks, num_bin_predictions+1]
    """

    if algorithm == "integral":
        return _integral(jd_pred)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'integral'.")

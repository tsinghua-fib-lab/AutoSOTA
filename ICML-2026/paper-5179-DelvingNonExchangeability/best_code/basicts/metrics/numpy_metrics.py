import numpy as np
from torchmetrics.utilities.checks import _check_same_shape


def masked_coverage(y_hat, y, mask, axis=None):
    """

    :param y_hat: intervals, shape [2, ...]
    :param y: ground truth, shape [...]
    :return: Whether each prediction is covered or not
    """
    y_hat_lower, y_hat_upper = y_hat
    res = np.logical_and(y >= y_hat_lower, y <= y_hat_upper).astype('float')
    res = np.where(mask, res, np.nan)
    return np.nanmean(res, axis=axis)

def masked_pi_width(y_hat, y, mask, axis=None):
    y_hat_lower, y_hat_upper = y_hat
    res = y_hat_upper - y_hat_lower
    res = np.where(mask, res, np.nan)
    return np.nanmean(res, axis=axis)

def masked_pi_width_median(y_hat, y, mask, axis=None):
    y_hat_lower, y_hat_upper = y_hat
    res = y_hat_upper - y_hat_lower
    res = np.where(mask, res, np.nan)
    return np.nanmedian(res, axis=axis)

def masked_winkler_score(y_hat, y, mask, alpha, axis=None):
    y_hat_lower, y_hat_upper = y_hat
    width = y_hat_upper - y_hat_lower
    score = width + \
            (2 / alpha) * ((y_hat_lower - y) * (y < y_hat_lower) + \
                           (y - y_hat_upper) * (y > y_hat_upper))
    score = np.where(mask, score, np.nan)
    return np.nanmean(score, axis=axis)

from math import sqrt

from scipy.stats import norm
import numpy as np


def clip(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def point_estimator(p, q0, q1):
    """Compute the bias-adjusted point estimate.

    Args:
        p: Proportion judged "correct" on test set (Pr(Predict = correct)).
        q0: Specificity (Pr(Predict = incorrect | True = incorrect)).
        q1: Sensitivity (Pr(Predict = correct | True = correct)).
    Returns:
        float in [0, 1]
    """
    assert 0 <= p <= 1, "p must be in [0, 1]"
    assert 0 <= q0 <= 1, "q0 must be in [0, 1]"
    assert 0 <= q1 <= 1, "q1 must be in [0, 1]"

    th = (p + q0 - 1) / (q0 + q1 - 1)
    return clip(th)


def confidence_interval(p, q0, q1, n, m0, m1, alpha=0.05):
    """Compute the adjusted (1 - alpha) confidence interval.

    Uses a plug-in approach with smoothing to reflect uncertainty from
    both the test set (p) and the calibration set (q0, q1).
    """
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    assert 0 <= p <= 1, "p must be in [0, 1]"
    assert 0 <= q0 <= 1, "q0 must be in [0, 1]"
    assert 0 <= q1 <= 1, "q1 must be in [0, 1]"
    assert isinstance(n, (int, np.integer)) and n > 0, "n must be a positive integer"
    assert isinstance(m0, (int, np.integer)) and m0 > 0, "m0 must be a positive integer"
    assert isinstance(m1, (int, np.integer)) and m1 > 0, "m1 must be a positive integer"

    z = norm.ppf(1 - alpha / 2)
    p, q0, q1 = (n*p + z**2/2)/(n + z**2), (m0*q0 + 1)/(m0 + 2), (m1*q1 + 1)/(m1 + 2)
    n, m0, m1 = n+z**2, m0+2, m1+2
    th = (p + q0 - 1)/(q0 + q1 - 1)
    dth = 2*z**2 * (-(1-th)*q0*(1-q0)/m0 + th*q1*(1-q1)/m1)
    se = sqrt(p*(1-p)/n + (1-th)**2*q0*(1-q0)/m0 + th**2*q1*(1-q1)/m1) / (q0+q1-1)
    return clip(th + dth - z*se), clip(th + dth + z*se)
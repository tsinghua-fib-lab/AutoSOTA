import warnings

from math import sqrt
import numpy as np


def allocate_calibration_sample(m, p, q0_pilot=0.9, q1_pilot=0.9, m_pilot=0, eps=1e-6):
    """Allocate calibration budget between true-negative (m0) and true-positive (m1) subsets.

    Args:
        m: total calibration budget (int > 0)
        p: test-set judged-correct rate in [0,1]
        q0_pilot, q1_pilot: pilot estimates (or beliefs) for specificity and sensitivity
        m_pilot: number of pilot samples already collected per side combined
        eps: small positive value for numerical stability (default: 1e-6)
    Returns:
        (m0, m1): integers summing to m, with m0, m1 >= m_pilot
    """
    assert 0 <= p <= 1, "p must be in [0, 1]"
    assert 0 <= q0_pilot <= 1, "q0_pilot must be in [0, 1]"
    assert 0 <= q1_pilot <= 1, "q1_pilot must be in [0, 1]"
    assert isinstance(m, (int, np.integer)) and m > 0, "m must be a positive integer"
    assert isinstance(m_pilot, (int, np.integer)) and m_pilot >= 0, "m_pilot must be a non-negative integer"

    if p < eps:
        m1 = m - m_pilot
    else:
        if m_pilot == 0:
            warnings.warn(
                "If 'm_pilot' is 0, compute kappa using q0 and q1 as given values."
            )
            q1_pilot = min(q1_pilot, 1-eps)
            kappa = (1-q0_pilot) / (1-q1_pilot)
        else:
            kappa = (m_pilot*(1-q0_pilot) + 1) / (m_pilot*(1-q1_pilot) + 1)
        m1 = m / (1 + (1/p-1)*sqrt(kappa))
        m1 = max(m_pilot, round(min(m-m_pilot, m1)))

    m0 = m - m1
    return m0, m1

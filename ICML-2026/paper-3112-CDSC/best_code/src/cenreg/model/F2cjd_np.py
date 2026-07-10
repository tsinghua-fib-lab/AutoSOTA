import itertools
from collections.abc import Sequence

import numpy as np


def _copula_sum_sub(
    F_pred: np.ndarray,
    c,
    idx_list: list[int],
    i: int,
    K: int,
    idx_list_use_Ft: Sequence[int],
) -> np.ndarray:
    if i == K:
        temp_list = []
        for k in range(K):
            temp_list.append(F_pred[:, k, idx_list[k]])
        return c.cdf(np.stack(temp_list, axis=1))

    if i in idx_list_use_Ft:
        idx_list.append(1)
    else:
        idx_list.append(2)
    temp1 = _copula_sum_sub(F_pred, c, idx_list, i + 1, K, idx_list_use_Ft)
    idx_list.pop()
    idx_list.append(0)
    temp2 = _copula_sum_sub(F_pred, c, idx_list, i + 1, K, idx_list_use_Ft)
    idx_list.pop()
    return temp1 - temp2


def convert(
    F_pred: np.ndarray,
    copula,
    w: list[int],
    num_risks: int,
    k: int,
    list_K: list[int],
) -> np.ndarray:
    """
    Convert F_pred into jd_pred.

    Parameters
    ----------
    F_pred: np.ndarray of shape [batch_size, num_risks, 3]
        The three elements in the last axis are the lower bound, the current value, and the upper bound.

    copula: function

    w: list of weight parameters (between 0 and 1)

    num_risks: int
        The number of risks.

    k: int
        The index of the risk.

    list_K: list of int
        The list of indices of the risks.

    Returns
    -------
    jd_pred : np.ndarray of shape [batch_size, num_risks]
    """

    q = _copula_sum_sub(F_pred, copula, [], 0, num_risks, [k])
    sign = 1.0
    for i in range(num_risks + 1):
        if i < 2 or w[i] == 0.0:
            continue
        for v in itertools.combinations(list_K, i):
            if k in v:
                q -= sign * w[i] * _copula_sum_sub(F_pred, copula, [], 0, num_risks, v)
        sign *= -1.0
    return q

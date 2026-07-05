#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/13 16:53
 @File    : fisher_z_test.py
 """

import numpy as np
from scipy.stats import norm



def get_partial_matrix(S, X, Y):
    S = S[X, :]
    S = S[:, Y]
    return S


def partial_corr_coef(S, i, j, Y):
    S = np.matrix(S)
    X = [i, j]
    inv_syy = np.linalg.inv(get_partial_matrix(S, Y, Y))
    i2 = 0
    j2 = 1
    S2 = get_partial_matrix(S, X, X) - get_partial_matrix(S, X, Y) * inv_syy * get_partial_matrix(S, Y, X)
    c = S2[i2, j2]
    r = c / np.sqrt((S2[i2, i2] * S2[j2, j2]))

    return r


def cond_indep_fisher_z(data, var1, var2, cond=[], alpha=0.05):

    """
    COND_INDEP_FISHER_Z Test if var1 indep var2 given cond using Fisher's Z test
    CI = cond_indep_fisher_z(X, Y, S, C, N, alpha)
    C is the covariance (or correlation) matrix
    N is the sample size
    alpha is the significance level (default: 0.05)
    transfromed from matlab
    See p133 of T. Anderson, "An Intro. to Multivariate Statistical Analysis", 1984

    Parameters
    ----------
    data: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    cond: list
        List of variable names in given variables.

    Returns
    -------

    CI: int
        The  conditional independence of the fisher z test.
    r: float
        partial correlation coefficient
    p_value: float
        The p-value of the test
    """

    N, k_var = np.shape(data)
    list_z = [var1, var2] + list(cond)
    list_new = []
    for a in list_z:
        list_new.append(int(a))
    data_array = np.array(data)
    array_new = np.transpose(np.matrix(data_array[:, list_new]))
    cov_array = np.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    z = 0.5 * np.log((1+r) / (1-r))
    z0 = 0
    W = np.sqrt(N - len(S1) - 3) * (z - z0)
    cutoff = norm.ppf(1 - 0.5 * alpha)
    if abs(W) < cutoff:
        CI = 1
    else:
        CI = 0
    p = norm.cdf(W)
    r = abs(r)

    return CI, r, p







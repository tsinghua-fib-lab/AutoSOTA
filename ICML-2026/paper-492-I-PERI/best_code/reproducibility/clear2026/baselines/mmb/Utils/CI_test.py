import numpy as np
from scipy.stats import norm
from math import log, sqrt


    
from scipy.stats import pearsonr
import statsmodels.api as sm
from causallearn.utils.cit import CIT

def fisherz(X, Y, S, data, alpha):
    """
    Tests whether X is conditionally independent of Y given a conditioning set S,
    using partial correlation and a Pearson test.

    Args:
        data: numpy array of shape (n_samples, n_variables)
        X (int): column index of the first variable.
        Y (int): column index of the second variable.
        S (list[int] or None): list of column indices of conditioning variables.
        alpha (float): significance level threshold.

    Returns:
        float: p-value of the conditional independence test.
    """
    if S is None or len(S) == 0:
        r, pval = pearsonr(data[:, X], data[:, Y])
    else:
        # Extract conditioning set
        S_data = data[:, S]

        # Regress X ~ S
        model_X = sm.OLS(data[:, X], sm.add_constant(S_data)).fit()
        resid_X = model_X.resid

        # Regress Y ~ S
        model_Y = sm.OLS(data[:, Y], sm.add_constant(S_data)).fit()
        resid_Y = model_Y.resid

        # Correlation of residuals = partial correlation
        r, pval = pearsonr(resid_X, resid_Y)
    
    CI = pval > alpha
    return CI, pval

def gsq(X, Y, S, data, alpha):
    """
    Tests whether X is conditionally independent of Y given a conditioning set S,
    using the G² (likelihood ratio chi-squared) test for categorical variables.

    Args:
        data: numpy array of shape (n_samples, n_variables) with categorical/binary data
        X (int): column index of the first variable
        Y (int): column index of the second variable
        S (list[int] or None): list of column indices of conditioning variables
        alpha (float): significance level threshold

    Returns:
        tuple: (CI, pval)
            CI (bool): True if X is conditionally independent of Y given S
            pval (float): p-value of the test
    """
    S = [] if S is None else S

    # Drop rows with missing data in X, Y, or S
    cols = [X, Y] + S
    data_clean = data[~np.isnan(data[:, cols]).any(axis=1)]

    # Create G² CIT object
    gsq_obj = CIT(data_clean, "gsq")

    # Compute p-value
    pval = gsq_obj(X, Y, S)

    CI = pval > alpha
    return CI, pval


def CI_Test(X, Y, S, D, alpha, test):
    """
    Wrapper for conditional independence tests.

    Args:
        X (int): column index of first variable
        Y (int): column index of second variable
        S (list[int] or None): conditioning set column indices
        D (np.ndarray): data array
        alpha (float): significance level
        test (str): "fisherz" or "gsq"

    Returns:
        tuple: (CI, pval)
            CI (bool): True if X ⟂ Y | S
            pval (float): p-value of the test
    """
    if test == "fisherz":
        CI, pval = fisherz(X, Y, S, D, alpha)
    elif test == "gsq":
        CI, pval = gsq(X, Y, S, D, alpha)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return CI, pval

    
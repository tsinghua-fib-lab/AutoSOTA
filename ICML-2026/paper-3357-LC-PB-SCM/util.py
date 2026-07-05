import numpy as np
from itertools import combinations
import sympy as sp
import pickle
import re
import pandas as pd
from copy import deepcopy
from sympy import Mul, Add, symbols
from sympy.core.numbers import One

# none:0, tail:-1, arrow:1, circle:2
# Some edge marks are not supported.
def mag2graph(mag) -> str:
    rec = ""
    for u in range(len(mag)):
        for v in range(u+1, len(mag)):
            if mag[u][v] == 0:
                continue

            # Directed edge.
            if mag[u][v] == 1 and mag[v][u] == -1:
                rec += (f"[X{u+1}]->[X{v+1}]\n")
            elif mag[v][u] == 1 and mag[u][v] == -1:
                rec += (f"[X{v+1}]->[X{u+1}]\n")

            # Bidirected edge.
            if mag[u][v] == 1 and mag[v][u] == 1:
                rec += (f"[X{u+1}]<->[X{v+1}]\n")

            # Contains circle marks.
            if mag[u][v] == 2 and mag[v][u] == 2:
                rec += (f"[X{u+1}].-.[X{v+1}]\n")
            elif mag[v][u] == 2 and mag[u][v] == -1:
                rec += (f"[X{u+1}].--[X{v+1}]\n")
            elif mag[u][v] == 2 and mag[v][u] == -1:
                rec += (f"[X{u+1}]--.[X{v+1}]\n")
            elif mag[u][v] == 1 and mag[v][u] == 2:
                rec += (f"[X{u+1}].->[X{v+1}]\n")
            elif mag[v][u] == 1 and mag[u][v] == 2:
                rec += (f"[X{u+1}]<-.[X{v+1}]\n")
    return rec

def interactions2terms(interactions) -> set[frozenset]:
    terms = set()
    for term in interactions:
        mterm = {}
        for var in term:
            if var+1 in mterm:
                mterm[var+1] += 1
            else:
                mterm[var+1] = 1
        terms.add(frozenset(mterm.items()))
    return terms

def get_epgf(data, z):
    """
    Estimate the empirical probability generating function (EPGF) using numpy.
    """
    z = np.array(z)
    power_matrix = np.power(z[:, np.newaxis], data)
    product_per_sample = np.prod(power_matrix, axis=0)
    epgf = np.mean(product_per_sample)

    return epgf


def topo(graph):
    """
    Topological order of an adjacency matrix.
    """
    from collections import deque
    in_degree = {node: 0 for node in range(len(graph))}  # Initialize each node's in-degree to 0.
    for u in range(len(graph)):
        for v in range(len(graph)):
            if graph[u][v] != 0 :in_degree[v] += 1  # Compute each node's in-degree.
    queue = deque([node for node in range(len(graph)) if in_degree[node] == 0])  # Add nodes with in-degree 0 to the queue.
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)

        for v in range(len(graph)):
            if graph[u][v] != 0:
                in_degree[v] -= 1  # Remove the current node's outgoing edge and decrement the neighbor's in-degree.
                if in_degree[v] == 0:  # Add the neighbor to the queue if its in-degree becomes 0.
                    queue.append(v)

    return result

def pbscm(graph:np.ndarray, mu:np.ndarray, sample:int, seed=None):
    """
    Generate data with the PBSCM model.
    """
    if seed is not None:
        np.random.seed(seed)
    data = np.column_stack([np.random.poisson(u, sample) for u in mu])
    ord = topo(graph)
    for u in ord:
        for v in range(len(graph)):
            if graph[u][v] > 0:
                data[:, v] += np.random.binomial(data[:, u], graph[u][v])
    return data


def save_csv(res_df:pd.DataFrame, file_name):
    """ save result, append if exists"""
    try:
        saved_df = pd.read_csv(file_name)
        saved_df = saved_df._append(res_df, ignore_index=True)
        saved_df.to_csv(file_name, index=False)
    except FileNotFoundError:
        res_df.to_csv(file_name, index=False)

def partial_derivative(f, z_list, order_list, h=1e-5):
    """
    Recursively compute the mixed partial derivative of a multivariate function
    using forward differences.

    :param f: Multivariate function, e.g. f(x, y, z).
    :param z_list: Point where the derivative is evaluated, e.g. [x0, y0, z0].
    :param order_list: Partial derivative order for each variable.
                       For example, [2, 1, 0] means second order for the first
                       variable and first order for the second variable.
    :param h: Difference step size, a small positive number.
    :return: Partial derivative value at z_list.
    """
    # Recursive base case: if all orders are 0, return the function value at this point.
    if all(order == 0 for order in order_list):
        return f(*z_list)

    # Find the first dimension with a nonzero derivative order.
    dim_to_diff = -1
    for i, order in enumerate(order_list):
        if order > 0:
            dim_to_diff = i
            break

    # Build the next recursive order list by decreasing the current dimension's order by 1.
    next_order_list = list(order_list)
    next_order_list[dim_to_diff] -= 1

    # Prepare the two points in the forward-difference formula.
    z_plus_h = list(z_list)
    z_plus_h[dim_to_diff] += h

    # Recursively apply the forward-difference formula: (g(z+h) - g(z)) / h,
    # where g is f after reducing the derivative order in the current dimension.
    val1 = partial_derivative(f, z_plus_h, next_order_list, h)
    val2 = partial_derivative(f, z_list, next_order_list, h)

    return (val1 - val2) / h

def bootstrap_test(data, order_list, round, seed=None, one=True):
    """Construct multiple bootstrap samples and compute partial derivatives. Pass seed for reproducibility."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if one == True:
        def get_log_epgf(*z):
            return np.log(get_epgf(data.T, z))
        z_list = [.01] * data.shape[1]
        h = 1e-1
        rec = partial_derivative(get_log_epgf, z_list, order_list, h)
        return rec
    a = np.zeros(round)
    for i in range(round):
        sample_indices = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        sample_data = data[sample_indices, :]
        def get_log_epgf(*z):
            return np.log(get_epgf(sample_data.T, z))
        z_list = [.01] * data.shape[1]
        h = 1e-1
        rec = partial_derivative(get_log_epgf, z_list, order_list, h)
        a[i] = rec
    return a

def hypothesis_test(bootstrap_samples, p, interval, verbose=False, special_judge=False):
    """
    Test whether the (1-p) confidence interval lies within a specified interval.

    :param bootstrap_samples: Bootstrap sample array.
    :param p: Significance level, e.g. 0.05 corresponds to 95% confidence.
    :param interval: Checked interval [lower_bound, upper_bound].

    When there is only one sample, this is equivalent to checking whether that
    sample lies within [0, threshold].
    """
    lower_quantile = p / 2
    upper_quantile = 1 - p / 2

    ci_lower = np.quantile(bootstrap_samples, lower_quantile)
    ci_upper = np.quantile(bootstrap_samples, upper_quantile)

    interval_lower, interval_upper = interval

    # Test whether the confidence interval lies within the specified interval.
    # Special handling for cases with too many negative values.
    if special_judge and np.count_nonzero(bootstrap_samples<=0) >= np.count_nonzero(bootstrap_samples>0) and np.mean(bootstrap_samples) < 0:
        test_passed = True
    else:
        test_passed = (ci_lower >= interval_lower) and (ci_upper <= interval_upper)

    if verbose:
        print(f"Bootstrap sample mean: {np.mean(bootstrap_samples):.6f}")
        print(f"Bootstrap sample standard deviation: {np.std(bootstrap_samples, ddof=1):.6f}")
        print(f"{(1-p)*100}% confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"Test interval: [{interval_lower}, {interval_upper}]")

        if test_passed:
            print(f"Test passed: the confidence interval lies within the test interval.")
        else:
            print(f"Test failed: the confidence interval exceeds the test interval.")

    return test_passed

def hypothesis_test_significance(bootstrap_samples, p, verbose=False):
    """
    Standard significance test: test whether the statistic is significantly nonzero.

    This function implements a classical significance test for determining
    whether a coefficient is significantly different from 0.

    Statistical hypotheses:
        - Null hypothesis H0: theta = 0 (the coefficient equals 0)
        - Alternative hypothesis H1: theta != 0 (the coefficient is not 0)

    Test logic:
        - Compute the (1-p) confidence interval.
        - If 0 is outside the confidence interval, reject H0 and treat the
          coefficient as significantly nonzero.
        - If 0 is inside the confidence interval, fail to reject H0 and treat
          the coefficient as not significant.

    Parameters
    ----
    bootstrap_samples : numpy.ndarray
        Bootstrap sample array containing statistic estimates obtained by
        repeated sampling from the original data.
    p : float
        Significance level, e.g. 0.05 corresponds to 95% confidence.
    verbose : bool, optional (default=False)
        Whether to print details, including the confidence interval and test result.

    Returns
    ----
    bool
        - True: reject H0; the coefficient is significantly nonzero.
        - False: fail to reject H0; the coefficient is not significant and may be 0.

    Examples
    ----
    >>> bootstrap_samples = np.array([0.001, 0.002, 0.0015, 0.0018])
    >>> flag = hypothesis_test_significance(bootstrap_samples, p=0.05)
    >>> if flag:
    ...     print("The coefficient is significantly nonzero and should be added to the model")
    ... else:
    ...     print("The coefficient is not significant and can be ignored")
    """
    lower_quantile = p / 2
    upper_quantile = 1 - p / 2

    ci_lower = np.quantile(bootstrap_samples, lower_quantile)
    ci_upper = np.quantile(bootstrap_samples, upper_quantile)

    # Check whether 0 lies inside the confidence interval.
    # If 0 is outside the CI (ci_lower > 0 or ci_upper < 0), reject H0.
    test_passed = (ci_lower > 0) or (ci_upper < 0)
    # test_passed=True means H0 is rejected and the coefficient is significantly nonzero.

    if verbose:
        print(f"Bootstrap sample mean: {np.mean(bootstrap_samples):.6f}")
        print(f"Bootstrap sample standard deviation: {np.std(bootstrap_samples, ddof=1):.6f}")
        print(f"{(1-p)*100}% confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")

        if test_passed:
            print(f"Test result: reject H0; the coefficient is significantly nonzero (significance level alpha={p})")
        else:
            print(f"Test result: fail to reject H0; the coefficient is not significant (significance level alpha={p})")

    return not test_passed


def hypothesis_test_significance_one_side(bootstrap_samples, p, verbose=False):
    """
    Right-sided one-sided significance test: test whether the statistic is
    significantly greater than 0.

    This function implements a right-sided one-sided significance test for
    determining whether a coefficient is significantly greater than 0.

    Statistical hypotheses:
        - Null hypothesis H0: theta <= 0 (the coefficient is less than or equal to 0)
        - Alternative hypothesis H1: theta > 0 (the coefficient is greater than 0)

    Test logic:
        - Compute the p-quantile of the bootstrap samples.
        - If the p-quantile is greater than 0, at least a (1-p) fraction of
          samples are greater than 0.
        - Reject H0 and treat the coefficient as significantly greater than 0.
        - Otherwise, fail to reject H0 and treat the coefficient as not significant.

    For example, when p=0.05:
        - Compute the 5% quantile.
        - If the 5% quantile is greater than 0, at least 95% of samples are greater than 0.
        - Treat the effect as significant.

    Parameters
    ----
    bootstrap_samples : numpy.ndarray
        Bootstrap sample array containing statistic estimates obtained by
        repeated sampling from the original data.
    p : float
        Significance level, e.g. 0.05 requires at least 95% of samples to be greater than 0.
    verbose : bool, optional (default=False)
        Whether to print details, including the quantile and test result.

    Returns
    ----
    bool
        - True: fail to reject H0; the coefficient is not significant and may be <= 0.
        - False: reject H0; the coefficient is significantly greater than 0.

    """
    # Compute the p-quantile.
    quantile_p = np.quantile(bootstrap_samples, p)

    # Right-sided test: if the p-quantile is greater than 0, at least a (1-p)
    # fraction of samples are greater than 0.
    # Reject H0: theta <= 0, and treat the coefficient as significantly greater than 0.
    test_passed = quantile_p > 0
    # test_passed=True means H0 is rejected and the coefficient is significantly greater than 0.

    if verbose:
        print(f"Bootstrap sample mean: {np.mean(bootstrap_samples):.6f}")
        print(f"Bootstrap sample standard deviation: {np.std(bootstrap_samples, ddof=1):.6f}")
        print(f"{p*100}% quantile: {quantile_p:.6f}")
        print(f"At least {(1-p)*100}% of samples are greater than 0: {test_passed}")

        if test_passed:
            print(f"Test result: reject H0; the coefficient is significantly greater than 0 (significance level alpha={p})")
        else:
            print(f"Test result: fail to reject H0; the coefficient is not significant (significance level alpha={p})")

    return not test_passed

def hypothesis_test_practical(bootstrap_samples, p, threshold, verbose=False):
    """
    Practical significance test: test whether the statistic is practically far from 0.

    This function implements an effect-size-based significance test for
    determining whether a coefficient is practically nonzero. Unlike a standard
    significance test, this method accounts for the practical size of the effect.

    Statistical hypotheses:
        - Null hypothesis H0: |theta| <= threshold
          (the coefficient's absolute value is at most the threshold)
        - Alternative hypothesis H1: |theta| > threshold
          (the coefficient's absolute value is greater than the threshold)

    Test logic:
        - Compute the (1-p) confidence interval.
        - If the confidence interval lies completely outside [-threshold, threshold], reject H0.
        - That is, ci_upper < -threshold or ci_lower > threshold.
        - This means there is enough evidence that the coefficient is practically far from 0.

    Parameters
    ----
    bootstrap_samples : numpy.ndarray
        Bootstrap sample array containing statistic estimates obtained by
        repeated sampling from the original data.
    p : float
        Significance level, e.g. 0.05 corresponds to 95% confidence.
    threshold : float
        Practical significance threshold. For example, 1e-4 means a coefficient
        is significant only when its absolute value is greater than 0.0001.
    verbose : bool, optional (default=False)
        Whether to print details, including the confidence interval and test result.

    Returns
    ----
    bool
        - True: reject H0; the coefficient is practically nonzero and has a large enough effect.
        - False: fail to reject H0; the coefficient may be practically close to 0.

    Examples
    ----
    >>> bootstrap_samples = np.array([0.0002, 0.0003, 0.00025, 0.00028])
    >>> flag = hypothesis_test_practical(bootstrap_samples, p=0.05, threshold=1e-4)
    >>> if flag:
    ...     print("The coefficient is practically significant and should be added to the model")
    ... else:
    ...     print("The coefficient may be nonzero, but the effect is too small and can be ignored")
    """
    lower_quantile = p / 2
    upper_quantile = 1 - p / 2

    ci_lower = np.quantile(bootstrap_samples, lower_quantile)
    ci_upper = np.quantile(bootstrap_samples, upper_quantile)

    # Check whether the confidence interval lies completely outside [-threshold, threshold].
    # That is, ci_upper < -threshold (entire CI is negative) or ci_lower > threshold (entire CI is positive).
    test_passed = (ci_upper < -threshold) or (ci_lower > threshold)
    # test_passed=True means the coefficient is practically far from 0.

    if verbose:
        print(f"Bootstrap sample mean: {np.mean(bootstrap_samples):.6f}")
        print(f"Bootstrap sample standard deviation: {np.std(bootstrap_samples, ddof=1):.6f}")
        print(f"{(1-p)*100}% confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"Practical significance threshold: [-{threshold}, {threshold}]")

        if test_passed:
            print(f"Test result: reject H0; the coefficient is practically nonzero (significance level alpha={p})")
            if ci_upper < -threshold:
                print(f"  -> The coefficient is significantly negative and |theta| > {threshold}")
            else:
                print(f"  -> The coefficient is significantly positive and theta > {threshold}")
        else:
            print(f"Test result: fail to reject H0; the coefficient may be practically close to 0 (significance level alpha={p})")
            if ci_lower <= -threshold and ci_upper >= threshold:
                print(f"  -> The confidence interval crosses [-{threshold}, {threshold}], so the effect is uncertain")
            elif ci_lower > -threshold and ci_upper < threshold:
                print(f"  -> The confidence interval lies completely within [-{threshold}, {threshold}], so the effect is small")
            else:
                print(f"  -> The confidence interval is partly outside [-{threshold}, {threshold}], but evidence is insufficient")

    return test_passed

def diff_get_epgf_gradient_np(data, z_list, order_list):
    """Compute the gradient of the empirical probability generating function (EPGF) using numpy."""
    z_list = np.array(z_list)
    n, m = data.shape
    data_ = deepcopy(data)

    tmp = np.ones_like(data_)
    for i in range(n):
        o = order_list[i]
        for j in range(o):
            tmp[i, :] *= data_[i, :] - j
        data_[i, :] -= o

    power_matrix = np.power(z_list[:, np.newaxis], data_) * tmp
    product_per_sample = np.prod(power_matrix, axis=0)
    epgf_gradient = np.mean(product_per_sample)
    return epgf_gradient

# Get a PGF without coefficients.
# Borrowed from DFS.
def graph2pgf(graph:np.ndarray, l_vars:list) -> str:
    from PGF import PGF
    pgf = PGF(graph)
    n = len(graph)
    tar = pgf.compute(
        mu=[1 for _ in range(n)],
        s=[pgf.s[k] if k not in l_vars else 1 for k in range(n)],
        e=0.5*np.array(graph)
    )
    new_terms = []
    for term in tar.as_expr().as_ordered_terms():
        coeff, var_part = term.as_coeff_Mul()
        if type(var_part) != One:
            new_terms.append(var_part)

    new_poly = Add(*new_terms, evaluate=False)  # Prevent automatic simplification.

    s = str(new_poly)

    return s

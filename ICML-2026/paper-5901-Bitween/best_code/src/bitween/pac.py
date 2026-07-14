# PAC learning sample size bound calculation

from math import comb

import numpy as np


def sample_size_bound(n, d, b, k, p, epsilon, delta):
    """
    n: the number of variables in the polynomial
    d: the maximum degree of the polynomial
    b: the bound on the coefficients of the polynomial [-b, b]
    k: the maximum subset of terms selected in a hypothesis (UGLY_FACTOR of DIG)
    p: the precision of the coeefficients

    return: the sample size bound, m
    """

    # https://math.stackexchange.com/questions/551214/the-number-of-monomials-of-a-given-degree
    def number_of_terms(variables, max_degree):
        # The number of terms in a polynomial with n variables and degree d
        return comb(variables + max_degree, max_degree)

    monomials = number_of_terms(n, d)
    print("Number of monomials is", monomials)
    subsets = comb(monomials, k)
    print("Number of subsets is", subsets)
    # H = (2 * b + 1) * subsets
    H = (2 * b + 1) * (10**p) * subsets
    print("Hypothesis Space is", H)
    m = int(np.log(H / delta) / epsilon)
    print("Sample size bound is", m)
    return m


if __name__ == "__main__":
    # vtrace1(x, y, z, a, b), 5 variables, up to degree 5,
    # bound 20, selected max terms 20 -- DIG puts bound and subset as the same value and call it as UGLY_FACTOR
    print(sample_size_bound(n=5, d=3, b=20, k=20, p=2, epsilon=0.1, delta=0.2))

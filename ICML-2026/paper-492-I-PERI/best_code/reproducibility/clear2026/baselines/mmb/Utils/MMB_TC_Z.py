import math
import numpy as np
from scipy.stats import norm
from math import log, sqrt

def CI_Test(X, Y, S, D, alpha):
    X = int(X)
    Y = int(Y)
    if isinstance(S, np.ndarray):
        S = S.tolist()
    n = D.shape[0]
    DD = D[:, [X, Y, *S]]
    corr_matrix = np.corrcoef(DD, rowvar=False)
    try:
        inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
    r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
    if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)

    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(n - len(S) - 3) * abs(Z)
    p_value = 2 * (1 - norm.cdf(abs(X)))
    CI = p_value > alpha
    return CI, p_value
def MMB_TC(D, A, alpha, is_MMB):
    # Total Conditioning, see Pellet and Elisseeff
    # Parameters:
    # - D: Data matrix (number_of_samples * n).
    # - A: Target variable to find the MMB.
    # - alpha: The parameter of Fisher's Z transform.
    # - is_MMB: Matrix to track the MMB status.
    #
    # Returns:
    # - A_MMB: List of variables in the MMB of A.
    # - nTests: Number of tests performed.
    # - is_MMB: Updated matrix tracking MMB status.

    n = D.shape[1]
    do_n = n / 10
    ceil_result = math.floor(do_n)
    if ceil_result > 0:
        alpha = alpha/(ceil_result*10)
    nVars = D.shape[1]
    nTests = 0
    A_MMB = []
    X = A
    tmp = list(range(0, nVars))
    tmp.remove(X)
    for Y in tmp:
        if is_MMB[X][Y] == 0:
            S = list(range(0, nVars))
            S.remove(X)
            S.remove(Y)
            CI, p = CI_Test(X, Y, S, D, alpha)
            nTests += 1
            if not CI:
                A_MMB.append(Y)
                is_MMB[X][Y] = 1
                is_MMB[Y][X] = 1
            else:
                is_MMB[X][Y] = -1
                is_MMB[Y][X] = -1
        elif is_MMB[X][Y] == 1:
            A_MMB.append(Y)
        elif is_MMB[X][Y] == -1:
            continue

    return A_MMB,nTests,is_MMB


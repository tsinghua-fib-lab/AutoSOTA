import numpy as np
import pandas as pd

####################################################################################

def autoc(ranked):
    # mean effect (varying k)
    cate = ranked['cate'].to_numpy()
    n = len(cate)
    cumsum = np.cumsum(cate)
    top_means = cumsum / np.arange(1, n+1)

    # improvement over ate
    ate = cate.mean()
    toc = top_means - ate
    return toc.mean()

####################################################################################

def policy_value(ranked):
    # potential outcomes
    M0 = ranked['M0'].to_numpy()
    M1 = ranked['M1'].to_numpy()
    n = M0.size

    # cumulative outcomes
    csum_M1 = np.cumsum(M1)
    csum_M0 = np.cumsum(M0)
    total_M0 = csum_M0[-1]

    # mean policy value
    ks = np.arange(1, n + 1, dtype=int)
    treated_sum   = csum_M1[ks - 1]
    untreated_sum = total_M0 - csum_M0[ks - 1]
    pv = (treated_sum + untreated_sum) / n
    return pv.mean()

####################################################################################

def pehe(ranked):
    cate = ranked['cate'].to_numpy(dtype=float)
    est  = ranked['est'].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((cate - est) ** 2)))
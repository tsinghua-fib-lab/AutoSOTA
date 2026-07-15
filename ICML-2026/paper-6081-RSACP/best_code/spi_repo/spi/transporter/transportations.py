import numpy as np
from math import comb


def get_hg_prob_value(m, N, i, k):
    b1 = comb(k + i - 2, i - 1)
    b2 = comb(N + m - k - i + 1, m - i)
    b3 = comb(N + m, m)
    return b1 * b2 / b3


def get_hg_cdf_value(m, N, i, start, end):
    cdf = 0
    for k in range(start, end):
        cdf += get_hg_prob_value(m, N, i, k)
    return cdf


def get_hg_quantiles_window_matrix(N, m, beta, weights=False, inflated=False, epsilon=0.01):
    T = np.zeros((m, N), dtype=float)
    if inflated and beta / 2 > epsilon:
        l_q = np.max([beta / 2 - 1 / m, epsilon])
    else:
        l_q = beta / 2
    u_q = 1 - (beta - l_q)
    for i in range(m):
        curr_r = 1
        curr_q = get_hg_cdf_value(m, N, i+1, 1, curr_r+1)
        while curr_q <= l_q:
            curr_r += 1
            pdf = get_hg_prob_value(m, N, i+1, curr_r)
            curr_q += pdf
        start = curr_r
        while curr_q < u_q:
            curr_r += 1
            pdf = get_hg_prob_value(m, N, i+1, curr_r)
            curr_q += pdf
            if curr_r == N + 1:
                break
        end = curr_r + 1
        # sanity check
        cdf = get_hg_cdf_value(m, N, i+1, start, end)
        if cdf < 1 - beta:
            print(f'cdf = {cdf}, N={N}, m+1={m}, r={i+1} -> start = {start}, end = {end}', flush=True)
        assert cdf >= 1 - beta

        if weights:
            for k in range(start, min(end, N+1)):
                pdf = get_hg_prob_value(m, N, i+1, k)
                T[i, k-1] = pdf
        else:
            T[i, start-1:end-1] = 1
    return T


def buckets_by_hg_dist(A, B_sorted, beta=0.1, weights=False, **kwargs):
    T = get_hg_quantiles_window_matrix(len(B_sorted), len(A), beta=beta, weights=weights)
    # Rearrange matrix rows and columns to match original indices of A
    original_A_indices = np.argsort(A)
    T = T[np.argsort(original_A_indices), :]
    return T

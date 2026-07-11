import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats


def pearsonr_ci(df, col_x, col_y, alpha=0.05):
    '''
        https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    '''
    
    df_new = df[[col_x, col_y]].dropna()
    n = len(df_new) - 3

    r, p = stats.pearsonr(df_new[col_x], df_new[col_y])
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))

    return r, p, lo, hi


def spearmanr_ci(df, col_x, col_y, alpha=0.05):
    '''
        https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    '''
    
    df_new = df[[col_x, col_y]].dropna()
    n = len(df_new) - 3

    res = stats.spearmanr(df_new[col_x], df_new[col_y])
    r = res.statistic
    p = res.pvalue
    return r, p


def rbf_kernel(X, sigma=1):
    X = np.atleast_2d(X)
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    return K


def rbf_kernel_two_matrices(X1, X2, sigma=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    sq_dists = (
        np.sum(X1 ** 2, axis=1).reshape(-1, 1)  
        + np.sum(X2 ** 2, axis=1).reshape(1, -1)  
        - 2 * np.dot(X1, X2.T)  
    )
    
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K


def laplace_kernel_two_matrices(X1, X2, sigma=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    l1_dists = np.sum(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)
    K = np.exp(-l1_dists / sigma)
    
    return K


def legendre_mat(X, d):
    matrix = np.zeros((X.shape[0], d))
    
    for n in range(d):
        Pn = Legendre([0] * (n + 1) + [1])
        matrix[:, n] = Pn(X)
    
    return matrix


def sample_cov_matrix(d, temp=100):
    A = np.random.rand(d, d)
    symmetric_matrix = A + A.T

    positive_semi_definite_matrix = np.dot(symmetric_matrix, symmetric_matrix.T)
    D = np.sqrt(np.diag(positive_semi_definite_matrix))

    normalization_matrix = np.outer(D, D)
    covariance_matrix = positive_semi_definite_matrix / normalization_matrix

    covariance_matrix = covariance_matrix / temp

    np.fill_diagonal(covariance_matrix, 1)
    return covariance_matrix


def mmr_test(df, Kxx, B=100):
    
    n = len(df)
    np.fill_diagonal(Kxx, 0)
    psi = np.array(df['psi'])

    mmr_stat = psi @ Kxx @ psi / (n - 1)

    h0_sample = np.zeros(B)
    for k in range(B):
        wpsi = psi * (np.random.multinomial(n, [1 / n] * n) - 1) 
        wprod = wpsi @ Kxx @ wpsi / n
        h0_sample[k] = wprod

    pval = (np.sum(mmr_stat < h0_sample) + 1) / (len(h0_sample) + 1)
    return int(pval < 0.05), pval

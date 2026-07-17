import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from jax.scipy.special import logsumexp
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
from sklearn.decomposition import PCA

is_cuda = True

def PCA_transform(X, n_components=4):
    pc = PCA(n_components)
    pc.fit(X)
    X = pc.transform(X)
    return X

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """
    Compute kernel matrix for a given kernel and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return jnp.exp(-(d**2) / 2)
    elif kernel == "laplace" and l == "l1":
        return jnp.exp(-d * jnp.sqrt(2))
    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (
        kernel == "matern_0.5_l2" and l == "l2"
    ):
        return jnp.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (
        kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (
        kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (
        kernel == "matern_3.5_l2" and l == "l2"
    ):
        return (
            1 + jnp.sqrt(7) * d + 2 * 7 / 5 * d**2 + 7 * jnp.sqrt(7) / 3 / 5 * d**3
        ) * jnp.exp(-jnp.sqrt(7) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (
        kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
            1
            + 3 * d
            + 3 * (6**2) / 28 * d**2
            + (6**3) / 84 * d**3
            + (6**4) / 1680 * d**4
        ) * jnp.exp(-3 * d)
    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')

def jax_distances(X, Y, l, max_samples=None, matrix=False):
    if l == "l1":

        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))

    elif l == "l2":

        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))

    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]

@partial(jit, static_argnums=(2, 3, 4))
def compute_bandwidths(X, Y, l, number_bandwidths, only_median=False):
    Z = jnp.concatenate((X, Y))
    distances = jax_distances(Z, Z, l, matrix=False)
    median = jnp.median(distances)
    if only_median:
        return median
    distances = distances + (distances == 0) * median
    dd = jnp.sort(distances)
    lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
    lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
    bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def mmdfuse(
    X,
    Y,
    key,
    alpha=0.05,
    kernels=("laplace","gaussian"),
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=2000,
    return_p_val=False,
):
    """
    Two-Sample MMD-FUSE test.

    Given data from one distribution and data from another distribution,
    return 0 if the test fails to reject the null
    (i.e. data comes from the same distribution),
    or return 1 if the test rejects the null
    (i.e. data comes from different distributions).

    Fixing the two sample sizes and the dimension, the first time the function is
    run it is getting compiled. After that, the function can fastly be evaluated on
    any data with the same sample sizes and dimension (with the same other parameters).

    Parameters
    ----------
    X : array_like
        The shape of X must be of the form (m, d) where m is the number
        of samples and d is the dimension.
    Y: array_like
        The shape of X must be of the form (n, d) where m is the number
        of samples and d is the dimension.
    key:
        Jax random key (can be generated by jax.random.PRNGKey(seed) for an integer seed).
    alpha: scalar
        The value of alpha (level of the test) must be between 0 and 1.
    kernels: str or list
        The list should contain strings.
        The value of the strings must be: "gaussian", "laplace", "imq", "matern_0.5_l1",
        "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1",
        "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2",
        "matern_4.5_l2".
    lambda_multiplier: scalar
        The value of lambda_multiplier must be positive.
        The regulariser lambda is taken to be jnp.sqrt(minimum_m_n * (minimum_m_n - 1)) * lambda_multiplier
        where minimum_m_n is the minimum of the sample sizes of X and Y.
    number_bandwidths: int
        The number of bandwidths per kernel to include in the collection.
    number_permutations: int
        Number of permuted test statistics to approximate the quantiles.
    return_p_val: bool
        If true, the p-value is returned.
        If false, the test output Indicator(p_val <= alpha) is returned.

    Returns
    -------
    output : int
        0 if the aggregated MMD-FUSE test fails to reject the null
            (i.e. data comes from the same distribution)
        1 if the aggregated MMD-FUSE test rejects the null
            (i.e. data comes from different distributions)
    """
    # Assertions
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m = X.shape[0]
    n = Y.shape[0]
    assert n <= m
    assert n >= 2 and m >= 2
    assert 0 < alpha and alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int
    if type(kernels) is str:
        # convert to list
        kernels = (kernels,)
    for kernel in kernels:
        assert kernel in (
            "imq",
            "rq",
            "gaussian",
            "matern_0.5_l2",
            "matern_1.5_l2",
            "matern_2.5_l2",
            "matern_3.5_l2",
            "matern_4.5_l2",
            "laplace",
            "matern_0.5_l1",
            "matern_1.5_l1",
            "matern_2.5_l1",
            "matern_3.5_l1",
            "matern_4.5_l1",
        )

    # Lists of kernels for l1 and l2
    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Setup for permutations
    key, subkey = random.split(key)
    B = number_permutations
    # (B, m+n): rows of permuted indices
    idx = random.permutation(
        subkey,
        jnp.array([[i for i in range(m + n)]] * (B + 1)),
        axis=1,
        independent=True,
    )
    # 11
    v11 = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))  # (m+n, )
    V11i = jnp.tile(v11, (B + 1, 1))  # (B, m+n)
    V11 = jnp.take_along_axis(
        V11i, idx, axis=1
    )  # (B, m+n): permute the entries of the rows
    V11 = V11.at[B].set(v11)  # (B+1)th entry is the original MMD (no permutation)
    V11 = V11.transpose()  # (m+n, B+1)
    # 10
    v10 = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
    V10i = jnp.tile(v10, (B + 1, 1))
    V10 = jnp.take_along_axis(V10i, idx, axis=1)
    V10 = V10.at[B].set(v10)
    V10 = V10.transpose()
    # 01
    v01 = jnp.concatenate((jnp.zeros(m), -jnp.ones(n)))
    V01i = jnp.tile(v01, (B + 1, 1))
    V01 = jnp.take_along_axis(V01i, idx, axis=1)
    V01 = V01.at[B].set(v01)
    V01 = V01.transpose()

    # Compute all permuted MMD estimates
    N = number_bandwidths * number_kernels
    M = jnp.zeros((N, B + 1))
    kernel_count = -1  # first kernel will have kernel_count = 0
    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            # Pairwise distance matrix
            Z = jnp.concatenate((X, Y))
            pairwise_matrix = jax_distances(Z, Z, l, matrix=True)

            # Collection of bandwidths
            def compute_bandwidths(distances, number_bandwidths):
                median = jnp.median(distances)
                distances = distances + (distances == 0) * median
                dd = jnp.sort(distances)
                lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
                lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
                bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
                return bandwidths

            distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
            bandwidths = compute_bandwidths(distances, number_bandwidths)

            # Compute all permuted MMD estimates for either l1 or l2
            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1
                for i in range(number_bandwidths):
                    # compute kernel matrix and set diagonal to zero
                    bandwidth = bandwidths[i]
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    K = K.at[jnp.diag_indices(K.shape[0])].set(0)
                    # compute standard deviation
                    unscaled_std = jnp.sqrt(jnp.sum(K**2))
                    # compute MMD permuted values 
                    # with lambda = jnp.sqrt(n * (n - 1))
                    M = M.at[kernel_count * number_bandwidths + i].set(
                        # following the reasoning of
                        # Schrab et al. MMDAgg Appendix C
                        (
                            jnp.sum(V10 * (K @ V10), 0)
                            * (n - m + 1) 
                            * (n - 1)
                            / (m * (m - 1))
                            + jnp.sum(V01 * (K @ V01), 0)
                            * (m - n + 1)
                            / m
                            + jnp.sum(V11 * (K @ V11), 0) 
                            * (n - 1)
                            / m
                            # jnp.sum(V10 * (K @ V10), 0)
                            # * (n - m + 1) 
                            # / (m * (m - 1) * n)
                            # + jnp.sum(V01 * (K @ V01), 0)
                            # * (m - n + 1)
                            # / (m * n * (n - 1))
                            # + jnp.sum(V11 * (K @ V11), 0) 
                            # / (m * n)
                        )
                        / unscaled_std
                        * jnp.sqrt(n * (n - 1))
                        # * jnp.sqrt(m * (m - 1))
                    )

    # Compute permuted and original statistics
    all_statistics = logsumexp(lambda_multiplier * M, axis=0, b=1 / N)  # (B1+1,)
    original_statistic = all_statistics[-1]  # (1,)

    # Compute statistics and test output
    p_val = jnp.mean(all_statistics >= original_statistic)
    output = p_val <= alpha

    # Return output
    if return_p_val:
        return output.astype(int), p_val
    else:
        # return [all_statistics.astype(float),M.astype(float)]
        return output.astype(int)
    
def TST_MMDFuse(name, N1, rs, check, n_test, alpha):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)

    H_MMD_FUSE = np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0

    X = np.concatenate((X_train,X_test_all))
    Y = np.concatenate((Y_train,Y_test_all))

    Z = np.concatenate((X,Y))
    Z = PCA_transform(Z, n_components=9)
    X = Z[:len(Z)//2]
    Y = Z[len(Z)//2:]

    X_train = X[:len(X_train)]
    X_test_all = X[len(X_train):]
    Y_train = Y[:len(Y_train)]
    Y_test_all = Y[len(Y_train):]

    key = random.PRNGKey(42)
    # test by MMD FUSE
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]

        S_x = np.concatenate((X_train, X_test), axis=0)
        S_y = np.concatenate((Y_train, Y_test), axis=0)
 
        key, subkey = random.split(key)

        start_time = time.time()
        if name in ['cifar10']: 
            h_MMD_FUSE = mmdfuse(S_x, S_y, subkey, number_bandwidths=10,kernels='laplace',alpha=alpha)
        else:
            h_MMD_FUSE = mmdfuse(S_x, S_y, subkey, number_bandwidths=10,alpha=alpha)
        test_time += time.time() - start_time

        H_MMD_FUSE[k] = h_MMD_FUSE

    return H_MMD_FUSE, 0, test_time


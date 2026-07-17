"""
Jax implementation
This file contains our three tests:
MMDAggInc, HSICAggInc and KSDAggInc
which are implemented in the function agginc()
For details, see our paper:
Efficient Aggregated Kernel Tests using Incomplete U-statistics
Antonin Schrab, Ilmun Kim, Benjamin Guedj, Arthur Gretton
"""

import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.flatten_util import ravel_pytree
from functools import partial
import itertools
import psutil
import GPUtil as gputil
import warnings
import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.flatten_util import ravel_pytree
from functools import partial
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
is_cuda = True

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def agginc(
        agg_type,
        X,
        Y,
        R=200,
        alpha=0.05,
        weights_type="uniform",
        number_bandwidths=10,
        hsic_collection_parameters=(2, -2, 2),
        batch_size=400,
        memory_percentage=0.8,
        B1=500,
        B2=500,
        B3=50,
        seed=42,
        return_dictionary=False,
        bandwidths=None,
    ):
    """
    Efficient Aggregated tests for two-sample (MMD), independence (HSIC) 
    and goodness-of-fit (KSD) testing.
    
    Given the appropriate data for the type of testing, 
    return 0 if the test fails to reject the null (i.e. same distribution, independent, fits the data), 
    or return 1 if the test rejects the null (i.e. different distribution, dependent, does not fit the data).
    
    Parameters
    ----------
    agginc: str
        "mmd" or "hsic" or "ksd"
    X : array_like
        The shape of X must be of the form (N_X, d_X) where N_X is the number
        of samples and d_X is the dimension.
    Y : array_like
        The shape of Y must be of the form (N_Y, d_Y) 
        where N_Y is the number of samples and d_Y is the dimension.
        Case agginc = "mmd": Y is the second sample, we must have d_X = d_Y.
        Case agginc = "hsic": Y is the paired sample, we must have N_X = N_Y.
        Case agginc = "ksd": Y is the score of X, we must have N_Y = N_X and d_Y = d_X.
    R : int
        Number of superdiagonals to consider. 
        If R >= min(N_X, N_Y) - 1 then the complete U-statistic is computed in quadratic time.
    alpha: scalar
        The level alpha must be between 0 and 1.
    weights_type: str
        Must be "uniform", or "centred", or "increasing", or "decreasing".
    number_bandwidths: int
        The number of bandwidths to include in the collection when agginc is "mmd" or "ksd".
    hsic_collection_parameters : tuple
        Tuple of length either 3 or 5.
        Case tuple of length 3:
            hsic_collection_parameters = (power, l_minus, l_plus)
            collection = (power ** i * median_bandwidth_X, power ** j * median_bandwidth_Y)
                         for i, j = l_minus, ..., l_plus
        Case tuple of length 5:
            hsic_collection_parameters = (power, l_minus_X, l_plus_X, l_minus_Y, l_plus_Y)
            collection = (power ** i * median_bandwidth_X, power ** j * median_bandwidth_Y)
                         for i = l_minus_X, ..., l_plus_X
                         for j = l_minus_Y, ..., l_plus_Y
    batch_size : int or None or str
        The memory cost consists in storing an array of shape (batch_size, R * N - R * (R - 1) / 2)
        where batch_size is between 1 and B1 + B2.
        Using batch_size = "auto", calculates automatically the batch_size which uses memory_percentage of the cpu/gpu memory (default 80%).
        For faster runtimes but using more memory, use batch_size = None (equivalent to batch_size = B1 + B2)
        By decreasing batch_size from B1 + B2, the memory cost is reduced but the runtimes increase.
    memory_percentage: scalar
        The value of memory_percentage must be between 0 and 1.
        It is used when batch_size = "auto", the batch_size is calculated automatically to use memory_percentage of the cpu/gpu memory.
    B1: int
        B1 is the number of wild bootstrap samples to approximate the quantiles.
    B2: int
        B2 is the number of wild bootstrap samples to approximate the level correction.
    B3: int
        Number of steps of bissection method to perform to estimate the level correction.
    seed: int 
        Random seed used for the randomness of the Rademacher variables.
    return_dictionary: bool
        If true, a dictionary is also returned containing the overall test out and for each single test: 
        the test output, the kernel, the bandwidth, the statistic, the quantile, 
        the p-value and the p-value threshold value.
   bandwidths: array_like or list or None
        If bandwidths is None, the collection of bandwidths is computed automatically.
        Otherwise, the collection provided in bandwidths is used instead.
        If agg_type is "mmd" or "ksd", then bandwidths needs to be an array of one dimension.
        If agg_type is "hsic", then bandwidths should be a list containing 2 elements:
            first element is a list or array of bandwidths for X,
            second element is a list or array of bandwidths for Y.
        Note that number_bandwidths and hsic_collection_parameters are overwritten by bandwidths.

        
    Returns
    -------
    output : int
        0 if the AggInc test fails to reject the null (i.e. same distribution, independent, fits the data), 
        1 if the AggInc test rejects the null (i.e. different distribution, dependent, does not fit the data).
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the overall output of the AggInc test, and for each single test: 
        the test output, the kernel, the bandwidth, the statistic, the quantile, 
        the p-value and the p-value threshold.
    
    Examples
    --------
    Check out the notebooks demo.ipynb and speed.ipynb for examples and speed comparisons.
    
    >>> # MMDAggInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1
    >>> output = agginc("mmd", X, Y)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = agginc("mmd", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'MMDAggInc test reject': True,
     'Single test 1': {'Bandwidth': 0.8926196098327637,
      'Kernel Gaussian': True,
      'MMD': 0.3186362385749817,
      'MMD quantile': 0.0025616204366087914,
      'Reject': True,
      'p-value': 0.0019960079807788134,
      'p-value threshold': 0.04590817913413048},
      ...
    }
    
    >>> # HSICAggInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = 0.5 * X + random.uniform(subkeys[1], shape=(500, 10))
    >>> output = agginc("hsic", X, Y)
    >>> output
    Array(0, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = agginc("hsic", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'HSICAggInc test reject': True,
     'Single test 1': {'Bandwidth X': 0.31978243589401245,
      'Bandwidth Y': 0.3518877327442169,
      'HSIC': 3.8373030974980793e-07,
      'HSIC quantile': 8.487702416459797e-07,
      'Kernel Gaussian': True,
      'Reject': False,
      'p-value': 0.17365269362926483,
      'p-value threshold': 0.007984011434018612},
      ...
    }
    
    >>> # KSDAggInc
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> X = jnp.array(X)
    >>> score_X = jnp.array(score_X)
    >>> output = agginc("ksd", X, score_X)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = agginc("ksd", X, score_X, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'KSDAggInc test reject': True,
     'Single test 1': {'Bandwidth': 1.0,
      'KSD': 0.0005635482375510037,
      'KSD quantile': 0.0010665705194696784,
      'Kernel IMQ': True,
      'Reject': False,
      'p-value': 0.12974052131175995,
      'p-value threshold': 0.01596805267035961},
      ...
    }
    """
    # Warnings
    assert memory_percentage > 0 and memory_percentage <= 1
    if number_bandwidths != 10 and agg_type == "hsic":
        warnings.warn("Parameter 'number_bandwidths' is not at its default value. Note that this parameter has no effect on the HSICAggInc test. The collection of bandwidths for HSICAggInc can be varied by changing the parameter 'hsic_collection_parameters'. Alternatively, a custom collection can be provided using the parameter 'bandwidths'.")
    if agg_type != "hsic" and hsic_collection_parameters != (2, -2, 2):
        warnings.warn("Parameter 'hsic_collection_parameters' is not at its default value. Note that this parameter has no effect on the MMDAggInc and KSDAggInc tests. The size of the collection of bandwidths for those tests can be varied by changing the parameter 'number_bandwidths'. Alternatively, a custom collection can be provided using the parameter 'bandwidths'.")
    
    # function compute_h_values
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
    elif agg_type == "hsic":
        compute_h_values = compute_h_HSIC_values
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
    else:
        raise ValueError('The value of agg_type should be "mmd" or' '"hsic" or "ksd".')
        
    # collection of bandwidths
    if bandwidths != None:
        if agg_type == "mmd" or agg_type == "ksd":
            assert bandwidths.ndim == 1
            number_bandwidths = len(bandwidths)
        elif agg_type == "hsic":
            assert len(bandwidths) == 2 and len(bandwidths[0]) > 0 and len(bandwidths[0]) > 0
            number_bandwidths = len(bandwidths[0]) * len(bandwidths[1])
    elif agg_type == "mmd":
        Z = jnp.concatenate((X, Y))
        max_samples = 500
        distances = jax_distances(X, Y, max_samples)
        distances = distances + (distances == 0) * jnp.median(distances)
        dd = jnp.sort(distances)
        lambda_min = jax.lax.cond(
            jnp.min(distances) < 10 ** (-1), 
            lambda : jnp.maximum(dd[(jnp.floor(len(dd) * 0.05).astype(int))], 10 ** (-1)), 
            lambda : jnp.min(distances),
        )
        lambda_min = lambda_min / 2
        lambda_max = jnp.maximum(jnp.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = jnp.array([power ** i * lambda_min for i in range(number_bandwidths)])
    elif agg_type == "hsic":
        assert weights_type == "uniform"
        if len(hsic_collection_parameters) == 3:
            power, l_minus, l_plus = hsic_collection_parameters
            l_minus = (l_minus, l_minus)
            l_plus = (l_plus, l_plus)
        elif len(hsic_collection_parameters) == 5:
            power, l_minus_X, l_plus_X, l_minus_Y, l_plus_Y = hsic_collection_parameters
            l_minus = (l_minus_X, l_minus_Y)
            l_plus = (l_plus_X, l_plus_Y)
        else:
            raise ValueError(
                "hsic_collection_parameters should have length either 3 or 5."
            )
        max_samples=500
        distances = jax_distances(X, X, max_samples)
        median_bandwidth_X = jnp.median(distances)
        distances = jax_distances(Y, Y, max_samples)
        median_bandwidth_Y = jnp.median(distances)
        median_bandwidths = (median_bandwidth_X, median_bandwidth_Y)
        bandwidths = [
            [power ** i * median_bandwidths[j] for i in range(l_minus[j], l_plus[j] + 1)]
            for j in range(2)
        ]
        number_bandwidths = len(bandwidths[0]) * len(bandwidths[1])
    elif agg_type == "ksd":
        max_samples = 500
        distances = jax_distances(X, X, max_samples)
        distances = distances + (distances == 0) * jnp.median(distances)
        lambda_min = 1
        lambda_max = jnp.maximum(jnp.max(distances), 2)
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = jnp.array([power ** i * lambda_min / X.shape[1] for i in range(number_bandwidths)])

    # compute all h-values
    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    
    # compute bootstrap and original statistics

    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B1 + B2, seed, batch_size, return_original=True, memory_percentage=memory_percentage
    )
    
    # compute quantile
    quantiles, u_correction = compute_quantile(
        bootstrap_values, original_value, B1, B2, B3, alpha, weights_type
    )
    
    # compute test output and dictionary
    output = return_test_output(
        agg_type, 
        bootstrap_values, 
        original_value, 
        quantiles, 
        u_correction, 
        bandwidths, 
        weights_type, 
        B1, 
        return_dictionary,
    )
    
    return output

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9, 10))
def inc(
    agg_type, 
    X, 
    Y, 
    R=200,
    alpha=0.05,
    batch_size=400,
    memory_percentage=0.8,
    B=500, 
    seed=42,
    return_dictionary=False,
    bandwidth=None,
):
    """
    Efficient test for two-sample (MMD), independence (HSIC) and goodness-of-fit 
    (KSD) testing, using median bandwidth (no aggregation).
    
    Given the appropriate data for the type of testing, 
    return 0 if the test fails to reject the null (i.e. same distribution, independent, fits the data), 
    or return 1 if the test rejects the null (i.e. different distribution, dependent, does not fit the data).
    
    Parameters
    ----------
    agginc: str
        "mmd" or "hsic" or "ksd"
    X : array_like
        The shape of X must be of the form (N_X, d_X) where N_X is the number
        of samples and d_X is the dimension.
    Y : array_like
        The shape of Y must be of the form (N_Y, d_Y) 
        where N_Y is the number of samples and d_Y is the dimension.
        Case agginc = "mmd": Y is the second sample, we must have d_X = d_Y.
        Case agginc = "hsic": Y is the paired sample, we must have N_X = N_Y.
        Case agginc = "ksd": Y is the score of X, we must have N_Y = N_X and d_Y = d_X.
    R : int
        Number of superdiagonals to consider. 
        If R >= min(N_X, N_Y) - 1 then the complete U-statistic is computed in quadratic time.
    alpha: float
        The level alpha must be between 0 and 1.
    batch_size : int or None or str
        The memory cost consists in storing an array of shape (batch_size, R * N - R * (R - 1) / 2)
        where batch_size is between 1 and B.
        Using batch_size = "auto", calculates automatically the batch_size which uses 80% of the memory.
        For faster runtimes but using more memory, use batch_size = None (equivalent to batch_size = B)
        By decreasing batch_size from B, the memory cost is reduced but the runtimes increase.
    memory_percentage: float
        The value of memory_percentage must be between 0 and 1.
        It is used when batch_size = "auto", the batch_size is calculated automatically 
        to use memory_percentage of the memory.
    B: int
        B is the number of wild bootstrap samples to approximate the quantiles.
    seed: int 
        Random seed used for the randomness of the Rademacher variables.
    return_dictionary: bool
        If true, a dictionary is also returned containing the test out, the kernel, the bandwidth, 
        the statistic, the statistic quantile, the p-value and the p-value threshold value (level).
   bandwidth: float or list or None
        If bandwidths is None, the bandwidth used is the median heuristic.
        Otherwise, the bandwidth provided is used instead.
        If agg_type is "mmd" or "ksd", then bandwidth needs to be a float.
        If agg_type is "hsic", then bandwidths should be a list 
        containing 2 floats (bandwidths for X and Y).

        
    Returns
    -------
    output : int
        0 if the Inc test fails to reject the null (i.e. same distribution, independent, fits the data), 
        1 if the Inc test rejects the null (i.e. different distribution, dependent, does not fit the data).
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the output of the Inc test, the kernel, the bandwidth, 
        the statistic, the quantile, the p-value and the p-value threshold (level).
    
    
    Examples
    --------
    
    >>> # MMDInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1
    >>> output = inc("mmd", X, Y)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("mmd", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth': 3.391918659210205,
     'Kernel Gaussian': True,
     'MMD': 0.9845684170722961,
     'MMD quantile': 0.007270246744155884,
     'MMDInc test reject': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    
    >>> # HSICInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = 0.5 * X + random.uniform(subkeys[1], shape=(500, 10))
    >>> output = inc("hsic", X, Y)
    >>> output
    Array(0, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("hsic", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth X': 1.2791297435760498,
     'Bandwidth Y': 1.4075509309768677,
     'HSIC': 0.00903838686645031,
     'HSIC quantile': 0.0005502101266756654,
     'HSICInc test reject': True,
     'Kernel Gaussian': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    
    >>> # KSDInc
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> X = jnp.array(X)
    >>> score_X = jnp.array(score_X)
    >>> output = inc("ksd", X, score_X)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("ksd", X, score_X, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth': 10.13830852508545,
     'KSD': 2.4731751182116568e-05,
     'KSD quantile': 5.930277438892517e-06,
     'KSDInc test reject': True,
     'Kernel IMQ': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    """
    
    # function compute_h_values
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
    elif agg_type == "hsic":
        compute_h_values = compute_h_HSIC_values
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
    else:
        raise ValueError('The value of agg_type should be "mmd" or' '"hsic" or "ksd".')
        
    # bandwidth: use provided one or compute median heuristic
    if bandwidth is not None:
        if agg_type == "mmd" or agg_type == "ksd":
            bandwidths = jnp.array(bandwidth).reshape(1)
        elif agg_type == "hsic":
            assert len(bandwidth) == 2
            bandwidths = (jnp.array(bandwidth[0]).reshape(1), jnp.array(bandwidth[1]).reshape(1))
    elif agg_type == "mmd":
        max_samples=500
        distances = jax_distances(X, Y, max_samples)
        median_bandwidth = jnp.median(distances)
        bandwidths = jnp.array([median_bandwidth])
    elif agg_type == "hsic":
        max_samples=500
        distances = jax_distances(X, X, max_samples)
        median_bandwidth_X = jnp.median(distances)
        distances = jax_distances(Y, Y, max_samples)
        median_bandwidth_Y = jnp.median(distances)
        bandwidths = (jnp.array([median_bandwidth_X]), jnp.array([median_bandwidth_Y]))
    elif agg_type == "ksd":
        max_samples=500
        distances = jax_distances(X, X, max_samples)
        median_bandwidth = jnp.median(distances)
        bandwidths = jnp.array([median_bandwidth])
    else:
        raise ValueError('The value of agg_type should be "mmd" or "hsic" or "ksd".')
        
    # compute all h-values
    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    
    # compute bootstrap and original statistics
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B, seed, batch_size, return_original=True, memory_percentage=memory_percentage
    )
    original_value = original_value[0]
    
    # compute quantile
    assert bootstrap_values.shape[0] == 1
    bootstrap_1 = jnp.column_stack([bootstrap_values, original_value])
    bootstrap_1_sorted = jnp.sort(bootstrap_1)
    quantile = bootstrap_1_sorted[0, (jnp.ceil((B + 1) * (1 - alpha))).astype(int) - 1]
    
    # reject if p_val <= alpha
    p_val = jnp.mean((bootstrap_1 - original_value.reshape(-1, 1) >= 0), -1)[0]
    reject_p_val = p_val <= alpha

    # reject if original_value > quantile
    reject_stat_val = original_value > quantile

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary[agg_type.upper() + "Inc test reject"] = reject_p_val
    if agg_type == "ksd":
        reject_dictionary["Kernel IMQ"] = True
    else:
        reject_dictionary["Kernel Gaussian"] = True
    if agg_type == "hsic":
        reject_dictionary["Bandwidth X"] = bandwidths[0][0]
        reject_dictionary["Bandwidth Y"] = bandwidths[1][0]
    else:
        reject_dictionary["Bandwidth"] = bandwidths[0]
    reject_dictionary[agg_type.upper()] = original_value
    reject_dictionary[agg_type.upper() + " quantile"] = quantile
    reject_dictionary["p-value"] = p_val
    reject_dictionary["p-value threshold"] = alpha

    # return output and dictionary
    if return_dictionary:
        return (reject_dictionary[agg_type.upper() + "Inc test reject"]).astype(int), reject_dictionary
    else:
        return (reject_dictionary[agg_type.upper() + "Inc test reject"]).astype(int)

def create_indices(N, R):
    """
    Return lists of indices of R superdiagonals of N x N matrix
    
    This function can be modified to compute any type of incomplete U-statistic.
    """
    index_X = list(
        itertools.chain(*[[i for i in range(N - r)] for r in range(1, R + 1)])
    )
    index_Y = list(
        itertools.chain(*[[i + r for i in range(N - r)] for r in range(1, R + 1)])
    )
    return index_X, index_Y

def compute_h_MMD_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_MMD values.

    inputs:
        X (m,d)
        Y (n,d)
        R int
        bandwidths (#bandwidths,)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    N = min(X.shape[0], Y.shape[0])
    assert X.shape[1] == Y.shape[1]

    index_i, index_j = create_indices(N, R)
    
    norm_Xi_Xj = jnp.linalg.norm(X[jnp.array(index_i)] - X[jnp.array(index_j)], axis=1) ** 2
    norm_Xi_Yj = jnp.linalg.norm(X[jnp.array(index_i)] - Y[jnp.array(index_j)], axis=1) ** 2
    norm_Yi_Xj = jnp.linalg.norm(Y[jnp.array(index_i)] - X[jnp.array(index_j)], axis=1) ** 2
    norm_Yi_Yj = jnp.linalg.norm(Y[jnp.array(index_i)] - Y[jnp.array(index_j)], axis=1) ** 2

    h_values = jnp.zeros((bandwidths.shape[0], norm_Xi_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        K_Xi_Xj_b = jnp.exp(-norm_Xi_Xj / bandwidths[r] ** 2)
        K_Xi_Yj_b = jnp.exp(-norm_Xi_Yj / bandwidths[r] ** 2)
        K_Yi_Xj_b = jnp.exp(-norm_Yi_Xj / bandwidths[r] ** 2)
        K_Yi_Yj_b = jnp.exp(-norm_Yi_Yj / bandwidths[r] ** 2)
        h_values = h_values.at[r].set(K_Xi_Xj_b - K_Xi_Yj_b - K_Yi_Xj_b + K_Yi_Yj_b)

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values

def compute_h_HSIC_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_HSIC values.
    
    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (2, #bandwidths) (bandwidths for X and for Y)
        
    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape[0] == Y.shape[0]
    N = int(X.shape[0] / 2)

    bandwidths_X = jnp.array(bandwidths[0])
    bandwidths_Y = jnp.array(bandwidths[1])

    h_X_values, index_i, index_j, Nbis = compute_h_MMD_values(
        X[:N], X[N:], R, bandwidths_X, True
    )
    assert N == Nbis
    h_Y_values = compute_h_MMD_values(Y[:N], Y[N:], R, bandwidths_Y)

    # we need to consider all pairs of bandwidths
    h_XY_values = (
        jnp.expand_dims(h_X_values, 0) * jnp.expand_dims(h_Y_values, 1)
    ).reshape(h_X_values.shape[0] * h_Y_values.shape[0], -1)

    if return_indices_N:
        return h_XY_values, index_i, index_j, N
    else:
        return h_XY_values

def compute_h_KSD_values(
    X, score_X, R, bandwidths, return_indices_N=False, beta_imq=-0.5
):
    """
    Compute h_KSD values.

    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (#bandwidths,)
        beta_imq in (-1, 0)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape == score_X.shape
    N, d = X.shape

    index_i, index_j = create_indices(N, R)

    Xi_minus_Xj = X[jnp.array(index_i)] - X[jnp.array(index_j)]
    norm_Xi_Xj = jnp.linalg.norm(Xi_minus_Xj, axis=1) ** 2
    sXi = score_X[jnp.array(index_i)]
    sXj = score_X[jnp.array(index_j)]
    sXi_minus_sXj = sXi - sXj
    sXi_minus_sXj_dot_Xi_minus_Xj = jnp.einsum(
        "ij,ij->i", sXi_minus_sXj, Xi_minus_Xj, optimize=True
    )
    sXi_dot_sXj = jnp.einsum("ij,ij->i", sXi, sXj, optimize=True)

    h_values = jnp.zeros((bandwidths.shape[0], Xi_minus_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        b_norm_Xi_Xj = bandwidths[r] ** 2 + norm_Xi_Xj
        h_values = h_values.at[r].set(
            sXi_dot_sXj * b_norm_Xi_Xj ** beta_imq
            - sXi_minus_sXj_dot_Xi_minus_Xj
            * 2
            * beta_imq
            * b_norm_Xi_Xj ** (beta_imq - 1)
            - 2 * d * beta_imq * b_norm_Xi_Xj ** (beta_imq - 1)
            - 4
            * beta_imq
            * (beta_imq - 1)
            * norm_Xi_Xj
            * b_norm_Xi_Xj ** (beta_imq - 2)
        )

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values

def compute_bootstrap_values(
    h_values, index_i, index_j, N, B, seed, batch_size=400, return_original=False, memory_percentage=0.8
):
    """
    Compute B bootstrap values.

    inputs:
        h_values, index_i, index_j = compute_h_XXX_values(...)
        h_values (#bandwidths, R * N - R * (R - 1) / 2)
        N int
        B int
        seed int


    output (#bandwidths, B)
    """

    # print(B)

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    epsilon = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(N, B))
    
    # Bootstrap values can be computed as follows
    # with memory cost of storing an array (e_values) 
    # of size (R * N - R * (R - 1) / 2, B)
    
    # e_values = epsilon[index_i] * epsilon[index_j]
    # bootstrap_values = h_values @ e_values
    
    # Instead we use batches to store only arrays
    # of size (R * N - R * (R - 1) / 2, batch_size)
    # where batch_size is automatically chosen to use 80% of the memory
    # In the experiments of the paper, batch_size = None (i.e. batch_size = B) has been used
    # Larger batch_size increases the memory cost and decreases computational time
    
    if batch_size == None:
        batch_size = B
    elif batch_size == "auto":
        # Automatically compute the batch size depending on cpu/gpu memory 
        if "gpu" in str(jax.devices()[0]).lower() and len(gputil.getGPUs()) > 0:
            memory = gputil.getGPUs()[0].memoryTotal * 1048576 # bytes
        else:
            memory = psutil.virtual_memory().total # bytes
        memory_single_array = jnp.zeros(h_values.shape[0]).nbytes
        batch_size = int(memory * memory_percentage / memory_single_array)
    bootstrap_values = jnp.zeros((h_values.shape[0], epsilon.shape[1]))
    i = 0
    index = 0
    while index + batch_size < B or i == 0:
        index = i * batch_size
        epsilon_b = epsilon[:, index : index + batch_size]
        e_values_b = epsilon_b[jnp.array(index_i)] * epsilon_b[jnp.array(index_j)]
        bootstrap_values = bootstrap_values.at[:, index : index + batch_size].set(h_values @ e_values_b)
        i += 1
    bootstrap_values = bootstrap_values / len(index_i)

    if return_original:
        original_value = h_values @ jnp.ones(h_values.shape[1]) / len(index_i)
        return bootstrap_values, original_value
    else:
        return bootstrap_values

def compute_quantile(bootstrap_values, original_value, B1, B2, B3, alpha, weights_type):
    """
    Compute quantile.

    inputs:
        bootstrap_values (#bandwidths, B1 + B2)
        original_value (#bandwidths, )
        B1 int
        B2 int
        B3 int
        alpha in (0,1)
        weights_type "uniform" or "decreasing" or "increasing" or "centred"

    returns quantiles (#bandwidths, )
    """
    # this can be adjusted B1 for quantiles, B2 for P_u
    bootstrap_1 = jnp.column_stack([bootstrap_values[:, :B1], original_value])
    bootstrap_1_sorted = jnp.sort(bootstrap_1)  # sort each row
    bootstrap_2 = bootstrap_values[:, B1:]
    assert B2 == bootstrap_2.shape[1]

    weights = create_weights(bootstrap_values.shape[0], weights_type)
    # (1-u*w_lambda)-quantiles for the #bandwidths
    quantiles = jnp.zeros((bootstrap_values.shape[0], 1))
    u_min = 0.0
    u_max = jnp.min(1 / weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for i in range(bootstrap_values.shape[0]):
            quantiles = quantiles.at[i].set(
                bootstrap_1_sorted[
                    i, (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
                ]
            )
        P_u = jnp.sum(jnp.max(bootstrap_2 - quantiles, 0) > 0) / B2
        u_min, u_max = jax.lax.cond(P_u <= alpha, lambda: (u, u_max), lambda: (u_min, u))
    u = u_min
    for i in range(bootstrap_values.shape[0]):
            quantiles = quantiles.at[i].set(
                bootstrap_1_sorted[
                    i, (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
                ]
            )
    return quantiles, u

def return_test_output(agg_type, bootstrap_values, original_value, quantiles, u_correction, bandwidths, weights_type, B1, return_dictionary):
    """
    Compute test output and dictionary.
    
    inputs:
        agg_type "mmd" or "hsic" or "ksd"
        bootstrap_values (#bandwidths, B1 + B2)
        original_value (#bandwidths, )
        quantiles (#bandwidths, )
        u_correction scalar
        bandwidths (#bandwidths, ) for agg_type "mmd" or "hsic", 
                   list for agg_type "hsic" with first and second elements bandwidths_X and bandwidths_Y
        weights_type "uniform" or "decreasing" or "increasing" or "centred"
        B1 int
        return_dictionary bool
        
    returns 0 or 1 depending on test output
    also returns dictionary if return_dictionary is True
    """
    
    bootstrap_1 = jnp.column_stack([bootstrap_values[:, :B1], original_value])
    
    number_bandwidths = bootstrap_values.shape[0]
    weights = create_weights(number_bandwidths, weights_type)
    
    p_vals = jnp.mean((bootstrap_1 - original_value.reshape(-1, 1) >= 0), -1)
    thresholds = u_correction * weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    stat_vals = original_value
    quantiles = quantiles.reshape(-1)
    # reject if stat_val > quantile
    reject_stat_vals = stat_vals > quantiles
    
    if agg_type == "hsic":
        bandwidths = [(bandwidths[0][i], bandwidths[1][j]) for j in range(len(bandwidths[1])) for i in range(len(bandwidths[0]))]
        assert number_bandwidths == len(bandwidths)

    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary[agg_type.upper() + "AggInc test reject"] = False
    for i in range(number_bandwidths):
        index = "Single test " + str(i + 1)
        reject_dictionary[index] = {}
        reject_dictionary[index]["Reject"] = reject_p_vals[i]
        if agg_type == "ksd":
            reject_dictionary[index]["Kernel IMQ"] = True
        else:
            reject_dictionary[index]["Kernel Gaussian"] = True
        if agg_type == "hsic":
            reject_dictionary[index]["Bandwidth X"] = bandwidths[i][0]
            reject_dictionary[index]["Bandwidth Y"] = bandwidths[i][1]
        else:
            reject_dictionary[index]["Bandwidth"] = bandwidths[i]
        reject_dictionary[index][agg_type.upper()] = stat_vals[i]
        reject_dictionary[index][agg_type.upper() + " quantile"] = quantiles[i]
        reject_dictionary[index]["p-value"] = p_vals[i]
        reject_dictionary[index]["p-value threshold"] = thresholds[i]
        # Aggregated test rejects if one single test rejects
        reject_dictionary[agg_type.upper() + "AggInc test reject"] = jnp.any(ravel_pytree(
            (reject_dictionary[agg_type.upper() + "AggInc test reject"], 
            reject_p_vals[i])
        )[0])
    
    if return_dictionary:
        return (reject_dictionary[agg_type.upper() + "AggInc test reject"]).astype(int), reject_dictionary
    else:
        return (reject_dictionary[agg_type.upper() + "AggInc test reject"]).astype(int)

def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021).
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = jnp.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = jnp.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = jnp.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights

@partial(jit, static_argnums=(2,))
def jax_distances(X, Y, max_samples):
    def dist(x, y):
        z = x - y
        return jnp.sqrt(jnp.sum(jnp.square(z)))
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    output = output[jnp.triu_indices(output.shape[0])]
    return output

def human_readable_dict(dictionary):
    """
    Transform all jax arrays of one element into scalars.
    """
    meta_keys = dictionary.keys()
    for meta_key in meta_keys:
        if isinstance(dictionary[meta_key], jnp.ndarray):
            dictionary[meta_key] = dictionary[meta_key].item()
        elif isinstance(dictionary[meta_key], dict):
            for key in dictionary[meta_key].keys():
                if isinstance(dictionary[meta_key][key], jnp.ndarray):
                    dictionary[meta_key][key] = dictionary[meta_key][key].item()
                    
def TST_MMDAgg(name, N1, rs, check, n_test, alpha):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)

    H_MMDAGG = np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0
    # test by MMDAgg
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]

        S_x = np.concatenate((X_train, X_test), axis=0)
        S_y = np.concatenate((Y_train, Y_test), axis=0)

        start_time = time.time()
        h_MMDAGG = agginc("mmd", S_x, S_y, alpha=alpha)
        test_time += time.time() - start_time
        H_MMDAGG[k] = h_MMDAGG

    return H_MMDAGG, 0, test_time

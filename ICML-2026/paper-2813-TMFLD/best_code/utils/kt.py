'''
Algorithms from Shetty et al. 2022 for near-linear-time equal-weighted
compression.
'''


import numpy as np
import math
import logging

# from goodpoints.jax.kt import kernel_halve, \
    # kernel_split, kernel_swap
from goodpoints.jax.rounding import stratified_round,\
    log2_ceil, log4_ceil


def kt_compresspp(kernel, points, w, rng_gen, *,
                  inflate_size,
                  delta=0.5,
                  g=4):
    '''
    KT-Compress++ algorithm from Li et al. 2024 targeting a mean-zero kernel
    and a weighted input.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator state.
        inflate_size: Multiplicity, a power of 4.
        delta: Failure probability.
    Returns:
        A numpy array of indices of the coreset of length sqrt(inflate_size).
    '''

    log4_n = log4_ceil(inflate_size)
    assert(inflate_size == 4**log4_n)
    seq = stratified_round(w, inflate_size, rng_gen)
    logging.info(f'Compress++ with {seq.size} points...')
    coresets = compresspp(kernel, points.subset(seq), g, delta, rng_gen)
    coresets = [seq[coreset] for coreset in coresets]
    logging.info('KT-swap improvement...')
    coreset = kernel_swap(kernel, points, coresets,
                          rng_gen, mean_zero=True)
    return coreset


def compress(halve_fn, points, g, rng_gen,
             ordered=True, one_less_halve=False):
    '''
    Compress from Shetty et al. 2022.

    Args:
        halve_fn: Halving function.
        points: A SliceablePoints instance containing input information.
        g: Oversampling parameter.
        rng_gen: Random number generator state.
        ordered: Whether to return the coreset in the same order as the input.
        one_less_halve:
            Whether to skip the final halving and return a size 2^{g+1} sqrt(n)
            coreset.
    Returns: A numpy array representing the final coreset of size 2^g sqrt(n).
    '''
    n = points.length
    if n == 4**g:
        return np.arange(n)

    if ordered:
        perm = np.arange(n)
    else:
        perm = rng_gen.permutation(n)

    n_prime = n // 4
    S = []
    for i in range(4):
        inds = perm[i*n_prime:(i+1)*n_prime]
        S_prime = compress(halve_fn, points.subset(inds), g,
                           rng_gen, one_less_halve=False)
        S.append(inds[S_prime])
    S_cat = np.concatenate(S)
    if ordered:
        S_cat = np.sort(S_cat)
    if one_less_halve:
        return S_cat
    S_halve = halve_fn(points.subset(S_cat), rng_gen)
    return S_cat[S_halve]


def compresspp(kernel, points, g, delta, rng_gen,
                one_less_halve=True):
    '''
    Compress++ from Shetty et al. 2022 specialized to kernel thinning.
    Thins n inputs points to m points.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        g: Inflation parameter so that compress results in 2^g sqrt(n) points.
        delta: Failure probabilities, delta_i = delta/n.
        rng_gen: Random number generator state.
        one_less_halve:
            Whether to skip the final halving in compress and instead thin from
            2^{g+1} sqrt(n) to sqrt(n) with 2^{g+1} coresets.

    Returns:
        2^g corests of size m.
    '''

    n = points.length
    log2_n = log2_ceil(n)
    assert(n == 2**log2_n)
    assert(log2_n % 2 == 0)

    if g >= log2_n // 2:
        logging.info(f'g ({g}) is too large for n ({n})! Setting g to '
                     f'{log2_n // 2 - 1}.')
        g = log2_n // 2 - 1

    # Compute deltas for halving and thinning.
    beta_n = log2_n // 2 - (g + 1)
    halve_delta0 = delta / (4 * n * (2**g) * (g + (beta_n+1)*(2**g)))
    kt_delta = g / (g + (beta_n+1)*(2**g)) * delta

    def halve_fn(points, rng_gen):
        # The delta to halve is always proportional to input size squared.
        n = points.length
        halve_delta = (n ** 2) * halve_delta0
        S0, S1 = kernel_halve(kernel, points, delta, rng_gen)
        if rng_gen.random() < 0.5:
            return S0
        else:
            return S1
    logging.info('Compressing to 2^g sqrt(n) points...')
    S_c = compress(halve_fn, points, g, rng_gen,
                   one_less_halve=one_less_halve) # size: 2^g sqrt(n)
    assert((S_c.shape[0] ==
            (2 if one_less_halve else 1) * 2**g * (2**(log2_n//2))))

    logging.info('Kernel thinning to sqrt(n) points...')
    coresets_cpp = kernel_split(kernel, points.subset(S_c),
                                g+(1 if one_less_halve else 0),
                                kt_delta,
                                rng_gen)
    coresets_cpp = [S_c[coreset] for coreset in coresets_cpp]
    return coresets_cpp



'''
Kernel Thinning algorithm and its subrountines by Dwivedi et al. (2021).
'''

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import logging

from goodpoints.jax.sliceable_points import SliceablePoints
from goodpoints.jax.mmd import compute_mmd, compute_K_mean
from goodpoints.jax.st import stein_thin
from goodpoints.jax.rounding import stratified_round, log2_ceil


def kt(kernel, points, w, rng_gen, *,
       inflate_size, out_size, delta=0.5):
    '''
    Kernel thinning algorithm from Li et al. (2024) targeting a mean-zero
    kernel and a weighted input.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator state.
        inflate_size:
            Multiplicity with inflate_size/out_size being a power of 2.
        out_size: Output size.
        delta: Failure probability.
    Returns:
        A numpy array of indices of the coreset of length out_size.
    '''
    t = log2_ceil(inflate_size, out_size)
    seq = stratified_round(w, inflate_size, rng_gen)
    logging.info(f'KT splitting with {seq.size} points and thinning to '
                 f'{out_size} points...')
    coresets = kernel_split(kernel, points.subset(seq), t, delta,
                            rng_gen)
    coresets = [seq[coreset] for coreset in coresets]
    logging.info('KT-swap improvement...')
    coreset = kernel_swap(kernel, points, coresets,
                          rng_gen, mean_zero=True, inplace=True)
    return coreset


def kernel_split(kernel, points, t, delta, rng_gen):
    '''
    Thins inputs points x to floor(n/2^t) points using a kernel.
    This takes O(n^2) time.

    Args:
        kernel: A broadcastable JAX kernel function.
        points: A SliceablePoints instance containing input information.
        t: Thinning parameter.
        delta: Failure probability, delta_i = delta/n.
        rng_gen: Random number generator state.
    Returns:
        A list of coresets, each a numpy integer array of size floor(n/2^t).
    '''
    n = points.length
    assert(2**t <= n)

    cur_list = [np.arange(n)]
    for i in range(t):
        next_list = []
        for j, coreset in enumerate(cur_list):
            coreset0, coreset1 = kernel_halve(
                kernel,
                points.subset(coreset),
                delta,
                rng_gen)
            next_list.append(coreset[coreset0])
            next_list.append(coreset[coreset1])
        cur_list = next_list

    assert(len(cur_list) == 2**t)
    return cur_list


def kernel_halve(kernel, points, delta, rng_gen,
                 block_size=16):
    '''
    Exact halving using a kernel.

    Args:
        kernel: A broadcastable JAX kernel function.
        points: A SliceablePoints instance containing input information.
        delta: Failure probabilities, delta_i = delta/n.
        rng_gen: Random number generator state.
        block_size: Block size for the intermediate kernel matrix.
    Returns: (coreset0, coreset1), where
        coreset0: A numpy array containing indices of the first half.
        coreset1: A numpy array containing indices of the second half.
    '''

    n = points.length
    rand = rng_gen.random(size=n//2)
    rand = jnp.array(rand)

    n_remain = n % (2*block_size)
    n_pad = (2*block_size - n_remain) % (2*block_size)

    # Add padding to the points. These points will not affect the result
    # since they are added to the end of the sequence.
    points_padded = points.pad(n_pad)
    mask0 = kernel_halve_impl(kernel, points_padded, delta, rand,
                              block_size=block_size)
    if n_pad > 0:
        mask0 = np.array(mask0[:-n_pad])
    coreset0 = np.arange(n)[mask0]
    coreset1 = np.arange(n)[np.logical_not(mask0)]

    return coreset0, coreset1


@partial(jax.jit, static_argnames=('kernel', 'block_size'))
def kernel_halve_impl(kernel, points, delta, rand,
                      block_size):
    '''
    JAX-compiled version of kernel_halve.
    To save the number of calls to the kernel function while reducing the
    memory footprint, we access a block of entries of the kernel matrix with
    a given block size.
    '''
    n = points.length
    assert(n % (block_size * 2) == 0)
    if block_size == -1:
        block_size = n // 2
    K_diag = kernel(points, points)
    K_parity = kernel(points[::2], points[1::2])

    def inner_loop_body(j, args):
        k, mask0, sigma_sqr, K_coreset0, K_sum, K_block = args
        i = k * block_size + j
        b_sqr = K_diag[2*i] + K_diag[2*i+1] - 2 * K_parity[i]
        a, sigma_sqr = get_swap_params(sigma_sqr, b_sqr, delta)
        alpha = 0
        alpha += K_sum[2*i] - K_sum[2*i+1]
        alpha += -2 * (K_coreset0[2*i] - K_coreset0[2*i+1])
        threshold = jnp.minimum(1., 0.5 * jnp.maximum(0, 1 - alpha / a))
        swap = rand[i] < threshold
        l0 = jnp.where(swap, 2*i+1, 2*i)
        l1 = jnp.where(swap, 2*i, 2*i+1)
        mask0 = mask0.at[l0].set(True)

        Kl0 = K_block[l0 % (2*block_size)]
        Kl1 = K_block[l1 % (2*block_size)]
        K_coreset0 = K_coreset0 + Kl0
        K_sum = K_sum + Kl0 + Kl1
        return (k, mask0, sigma_sqr, K_coreset0, K_sum, K_block)

    def outer_loop_body(k, args):
        mask0, sigma_sqr, K_coreset0, K_sum = args
        i = k * 2*block_size
        points_block = points.dynamic_slice(i, 2*block_size)
        K_block = kernel(points_block[:, None], points[None, :])
        _, mask0, sigma_sqr, K_coreset0, K_sum, _ = jax.lax.fori_loop(
            0, block_size, inner_loop_body,
            (k, mask0, sigma_sqr, K_coreset0, K_sum, K_block))
        return (mask0, sigma_sqr, K_coreset0, K_sum)

    mask0 = jnp.zeros((n,), dtype=bool)
    mask0 = jax.lax.fori_loop(0, n // (block_size * 2),
                              outer_loop_body,
                              (mask0, 0,
                               jnp.zeros((n,)), jnp.zeros((n,))))[0]

    return mask0

def kernel_swap(kernel, points, coresets, rng_gen,
                num_repeat=1,
                random_swap_order=False,
                baseline=True,
                mean_zero=True,
                inplace=False):
    '''
    First choose the coreset that has the smallest MMD among all given
    corests, and then greedily swap points from the coreset to minimize MMD,
    allowing repeats.
    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        coresets: A list of coresets.
        rng_gen: Random number generator state.
        num_repeat: Number of times to repeat the greedy swapping.
        random_swap_order: If True, randomly permute the order of swapping.
        baseline:
            If True, add in either Stein thinning baseline (if mean-zero), or
            i.i.d. baseline otherwise.
        mean_zero:
            If True, the kernel is assumed to have zero mean in P and the swap
            is to minimize the MMD to P.
        inplace:
            If True, input coresets may be modified in place.
    Returns:
        (m,) a numpy array of indices of the improved coreset.
    '''
    n = points.length

    if not mean_zero:
        # Precompute K_sum in n^2 time.
        K_mean = compute_K_mean(kernel, points, points, jnp.ones((n,)) / n)
    else:
        K_mean = jnp.zeros((n,))

    def select_best_coreset(coresets):
        # Select the best coreset.
        if len(coresets) > 1:
            cur_mmd = np.inf
            for next_coreset in coresets:
                next_mmd = compute_mmd(kernel,
                                       points.subset(next_coreset),
                                       mode='mean-zero')
                if not mean_zero:
                    # Adjust the MMD to be against all points.
                    next_mmd += -2 * K_mean[next_coreset].mean()
                if next_mmd < cur_mmd:
                    coreset = next_coreset if inplace else np.copy(next_coreset)
                    cur_mmd = next_mmd
        else:
            coreset = coresets[0] if inplace else np.copy(coresets[0])
        return coreset

    candidates = [select_best_coreset(coresets)]
    if baseline:
        m = coresets[0].shape[0]
        if mean_zero:
            baseline_coreset = stein_thin(
                kernel, points, m)[1]
        else:
            baseline_coreset = np.linspace(
                0, n-1, m, dtype=int)
        candidates.append(baseline_coreset)

    if random_swap_order:
        # Save rng state to use same random orderings for each candidate
        rng_state = rng_gen.bit_generator.state

    # Next use KT-swap to improve each candidate coreset.
    for i, coreset in enumerate(candidates):
        # Set RNG state
        if random_swap_order:
            rng_gen.bit_generator.state = rng_state
        # Precompute sufficient statistic for identifying
        # best swap
        suff_stat = swap_sufficient_stat(kernel, points, coreset, K_mean)
        m = coreset.shape[0]
        for r in range(num_repeat):
            if random_swap_order:
                order = rng_gen.permutation(m)
                coreset = coreset[order]
            coreset, suff_stat = kernel_swap_impl(
                kernel, points, coreset, suff_stat)
            coreset = coreset.sort() 
        candidates[i] = coreset

    return select_best_coreset(candidates)


@partial(jax.jit, static_argnames=('kernel',))
def swap_sufficient_stat(kernel, points, coreset, K_mean):
    def loop_body(l, args):
        E, = args
        j = coreset[l]
        E = E + kernel(points[j], points)
        return (E,)

    m = coreset.shape[0]
    n = points.length
    return (kernel(points, points) / 2 - m * K_mean + 
            jax.lax.fori_loop(0, m, loop_body, (jnp.zeros((n,)),))[0])


@partial(jax.jit, static_argnames=('kernel',))
def kernel_swap_impl(kernel, points, coreset, suff_stat):
    def loop_body2(i, args):
        coreset, suff_stat = args
        # Remove influence of current coreset point
        j = coreset[i]
        suff_stat = suff_stat - kernel(points[j], points)
        # suff_stat[k] is the relative change in MMD^2 after 
        # adding point k to coreset
        k = jnp.argmin(suff_stat)
        suff_stat += kernel(points[k], points)
        coreset = coreset.at[i].set(k)
        return (coreset, suff_stat)

    return jax.lax.fori_loop(0, coreset.shape[0], loop_body2, 
                             (coreset, suff_stat))

def get_swap_params(sigma_sqr, b_sqr, delta):
    b = jnp.sqrt(b_sqr)
    sigma = jnp.sqrt(sigma_sqr)
    a = jnp.maximum(b * sigma * jnp.sqrt(2 * jnp.log(2 / delta)), b_sqr)
    a = jnp.maximum(a, 1.e-10) # stability fix
    sigma_sqr = sigma_sqr + b_sqr * jnp.maximum(
        0,
        1 + (b_sqr-2*a)*sigma_sqr/(a**2))
    return a, sigma_sqr
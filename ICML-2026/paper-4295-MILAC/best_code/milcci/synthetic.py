# -*- coding: utf-8 -*-
"""
Synthetic data generator for MILCCI.

Generates Y = A @ Phi.T with multi-axis label structure and controlled
ground truth, allowing quantitative evaluation of decomposition quality.

Temporal traces (Phi) are sampled from Gaussian Processes so that
trials sharing the same value on a given axis have correlated traces
for the ensembles assigned to that axis.
"""
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def generate_synthetic_data(N=50, T=100, n_ensembles_each=[2, 2],
                            axis_values=[[0, 1, 2], [0, 1]],
                            noise_std=0.3, seed=42,
                            A_sparsity=0.6,
                            gp_length_scale=0.15,
                            gp_sigma=0.25,
                            trials_per_condition=3):
    """
    Generate synthetic multi-axis tensor data.

    The model is:
        Y[:, :, m] = A[:, :, m] @ Phi[:, :, m].T + noise

    where A varies across conditions (with shared structure per axis)
    and Phi are GP-sampled temporal traces with within-axis correlation.

    Parameters
    ----------
    N : int
        Number of neurons / features.
    T : int
        Number of time bins.
    n_ensembles_each : list of int
        Ensembles per axis.
    axis_values : list of list
        Possible values for each axis.
    noise_std : float
    seed : int
    A_sparsity : float
        Fraction of zeros in A (between 0 and 1).
    gp_length_scale : float
        RBF length scale for GP prior on Phi.
    gp_sigma : float
        Perturbation scale for within-axis-value trial variability.
        Smaller = more similar traces across trials sharing an axis value.
    trials_per_condition : int
        Number of trials per unique condition.

    Returns
    -------
    data : dict with keys:
        'Y'                 : np.ndarray (N, T, M)
        'A_true'            : np.ndarray (N, P, M)
        'Phi_true'          : np.ndarray (T, P, M)
        'labels'            : list of int
        'labels_tuples'     : list of tuples
        'numbers2tuples'    : dict {int: tuple}
        'class_names'       : list of str
        'n_ensembles_each'  : list of int
        'ensembles_names'   : np.ndarray of str
        'axis_values'       : list of list
    """
    rng = np.random.RandomState(seed)
    num_axes = len(n_ensembles_each)
    assert num_axes == len(axis_values), (
        'num_axes %d != len(axis_values) %d' % (num_axes, len(axis_values))
    )
    n_ensembles = sum(n_ensembles_each)
    class_names = ['axis_%d' % ax for ax in range(num_axes)]
    ensembles_names = np.repeat(class_names, n_ensembles_each)

    # all label combinations, repeated trials_per_condition times
    all_combos = list(product(*axis_values))
    n_unique = len(all_combos)
    assert n_unique > 0, 'no label combinations generated'

    labels_tuples = []
    for combo in all_combos:
        labels_tuples.extend([combo] * trials_per_condition)
    M = len(labels_tuples)

    numbers2tuples = {}
    tuples2numbers = {}
    label_counter = 0
    for tup in all_combos:
        numbers2tuples[label_counter] = tup
        tuples2numbers[tup] = label_counter
        label_counter += 1

    labels = [tuples2numbers[tup] for tup in labels_tuples]

    # cumulative ensemble indices
    cumsum = np.cumsum([0] + list(n_ensembles_each))

    # axis assignment: which axis does each ensemble belong to
    axis_assignments = []
    for ax in range(num_axes):
        axis_assignments.extend([ax] * n_ensembles_each[ax])
    assert len(axis_assignments) == n_ensembles, (
        'axis_assignments length %d != n_ensembles %d' % (len(axis_assignments), n_ensembles)
    )

    # ------------------------------------------------------------------
    # generate ground-truth A (per axis-value templates)
    # ------------------------------------------------------------------
    A_templates = {}
    for ax in range(num_axes):
        e1, e2 = cumsum[ax], cumsum[ax + 1]
        n_ens = e2 - e1
        for val in axis_values[ax]:
            A_block = rng.randn(N, n_ens) * 0.5
            mask = rng.rand(N, n_ens) < A_sparsity
            A_block[mask] = 0
            A_block = np.abs(A_block)
            A_templates[(ax, val)] = A_block

    A_true_full = np.zeros((N, n_ensembles, M))
    for m, tup in enumerate(labels_tuples):
        for ax in range(num_axes):
            e1, e2 = cumsum[ax], cumsum[ax + 1]
            A_true_full[:, e1:e2, m] = A_templates[(ax, tup[ax])]

    # normalize A columns
    for m in range(M):
        col_norms = np.sum(np.abs(A_true_full[:, :, m]), axis=0) + 1e-18
        A_true_full[:, :, m] = A_true_full[:, :, m] / col_norms.reshape(1, -1) * 5.0

    # ------------------------------------------------------------------
    # generate ground-truth Phi via Gaussian Processes
    # ------------------------------------------------------------------
    Phi_true = _generate_gp_traces(
        T, n_ensembles, M, labels_tuples, axis_assignments,
        axis_values, num_axes,
        length_scale=gp_length_scale, sigma=gp_sigma, seed=seed,
    )
    assert Phi_true.shape == (T, n_ensembles, M), (
        'Phi_true shape %s != (%d, %d, %d)' % (str(Phi_true.shape), T, n_ensembles, M)
    )

    # ------------------------------------------------------------------
    # generate Y = A Phi^T + noise
    # ------------------------------------------------------------------
    Y = np.zeros((N, T, M))
    for m in range(M):
        signal = A_true_full[:, :, m] @ Phi_true[:, :, m].T
        noise = rng.randn(N, T) * noise_std
        Y[:, :, m] = signal + noise

    # ------------------------------------------------------------------
    # sanity checks
    # ------------------------------------------------------------------
    assert Y.shape == (N, T, M), 'Y shape %s' % str(Y.shape)
    assert A_true_full.shape == (N, n_ensembles, M), 'A shape %s' % str(A_true_full.shape)
    assert np.all(np.isfinite(Y)), 'Y contains non-finite values'
    assert np.all(np.isfinite(A_true_full)), 'A contains non-finite values'
    assert np.all(np.isfinite(Phi_true)), 'Phi contains non-finite values'

    for m in range(M):
        reco = A_true_full[:, :, m] @ Phi_true[:, :, m].T
        signal_power = np.sum(reco ** 2)
        noise_power = np.sum((Y[:, :, m] - reco) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-18))
        assert snr > 0, 'SNR for trial %d is %.1f dB (too low)' % (m, snr)

    return {
        'Y': Y,
        'A_true': A_true_full,
        'Phi_true': Phi_true,
        'labels': labels,
        'labels_tuples': labels_tuples,
        'numbers2tuples': numbers2tuples,
        'class_names': class_names,
        'n_ensembles_each': n_ensembles_each,
        'ensembles_names': ensembles_names,
        'axis_values': axis_values,
        'axis_assignments': axis_assignments,
    }


def _generate_gp_traces(T, n_ensembles, M, labels_tuples, axis_assignments,
                         axis_values, num_axes,
                         length_scale=0.15, sigma=0.25, seed=42):
    """
    Generate GP-based temporal traces with within-axis-value correlation.

    For each (ensemble, axis_value) pair:
      1. Sample a mean trace from a GP prior
      2. For each trial with that axis value, sample a perturbation
         around the mean (controlled by sigma)

    This ensures trials sharing the same axis value have correlated
    traces for the ensembles assigned to that axis.

    Parameters
    ----------
    T : int
    n_ensembles : int
    M : int
    labels_tuples : list of tuples
    axis_assignments : list of int
    axis_values : list of list
    num_axes : int
    length_scale : float
    sigma : float
    seed : int

    Returns
    -------
    Phi : np.ndarray, shape (T, n_ensembles, M)
    """
    time_pts = np.linspace(0, 1, T)[:, None]
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=1e-6)
    K = kernel(time_pts)

    labels_arr = np.array(labels_tuples)  # (M, num_axes)
    assert labels_arr.shape == (M, num_axes), (
        'labels_arr shape %s != (%d, %d)' % (str(labels_arr.shape), M, num_axes)
    )

    # pre-sample: one GP mean + perturbations per (ensemble, axis_value)
    gp_samples = {}
    for ens_num in range(n_ensembles):
        ax = axis_assignments[ens_num]
        for val in axis_values[ax]:
            # how many trials have this value on this axis?
            n_matching = int(np.sum(labels_arr[:, ax] == val))
            assert n_matching > 0, (
                'no trials with axis %d value %s' % (ax, str(val))
            )

            # sample GP mean
            gp = GaussianProcessRegressor(
                kernel=kernel,
                random_state=seed + ax + ens_num ** 2 + int(val),
            )
            mean_trace = gp.sample_y(
                time_pts, n_samples=1,
                random_state=int(val) + ax + ens_num ** 2,
            ).flatten()
            assert mean_trace.shape == (T,), (
                'mean_trace shape %s' % str(mean_trace.shape)
            )

            # sample perturbations around the mean
            rng = np.random.default_rng(seed + int(val) + ax + ens_num ** 2)
            perturbed = rng.multivariate_normal(
                mean=mean_trace,
                cov=sigma ** 2 * K,
                size=n_matching,
            ).T  # (T, n_matching)
            assert perturbed.shape == (T, n_matching), (
                'perturbed shape %s != (%d, %d)' % (str(perturbed.shape), T, n_matching)
            )

            gp_samples[(ens_num, ax, val)] = perturbed

    # assemble traces
    Phi = np.zeros((T, n_ensembles, M))
    counter = {}
    for ens_num in range(n_ensembles):
        ax = axis_assignments[ens_num]
        for m in range(M):
            val = labels_arr[m, ax]
            key = (ens_num, ax, val)
            idx = counter.get(key, 0)
            Phi[:, ens_num, m] = gp_samples[key][:, idx]
            counter[key] = idx + 1

    # make non-negative (absolute value)
    Phi = np.abs(Phi)

    assert np.all(np.isfinite(Phi)), 'GP traces contain non-finite values'
    return Phi

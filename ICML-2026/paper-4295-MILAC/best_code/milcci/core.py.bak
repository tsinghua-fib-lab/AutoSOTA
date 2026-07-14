# -*- coding: utf-8 -*-
"""
Core MILCCI decomposition.

Given a tensor Y of shape (N_neurons, T_time, M_trials) and per-trial
multi-axis labels, MILCCI finds:

    Y[:, :, m]  ~  A[:, :, m]  @  Phi[:, :, m].T

where A varies across conditions (with similarity regularization along
each label axis) and Phi are temporal traces.
"""
import numpy as np

from .utils import (check_empty_list, make_labels_unique_order,
                    identity, check_if_labels_batches, lists2list)
from .solvers import solve_regularized, solve_ls_for_phi
from .regularization import create_basis_patterns, build_nu_matrices
from .phi_inference import infer_phi


def fit(data, labels, numbers2tuples,
        n_ensembles=8, n_ensembles_each=[], num_axes=0,
        nu=[], lambda_similarity=500, factor_A=5,
        func_normalize_A=np.sum, decor_A=5, num_repeats=6,
        cont_axis_list=[], value_nu_fixed=1.0,
        params_init_A={}, params_basis_pattern={},
        split_A=True, another_update_for_A=False,
        style_infer_phi='LS', params_update_phi={},
        seed=5, verbose=False, max_trials_each=25):
    """
    Run the MILCCI multi-axis tensor decomposition.

    Parameters
    ----------
    data : np.ndarray, shape (N, T, M)
        Observed tensor. N neurons/features, T time bins, M trials.
    labels : list or array of length M
        Integer label per trial (indexes into numbers2tuples).
    numbers2tuples : dict
        {label_int: (axis0_val, axis1_val, ...)} mapping.
    n_ensembles : int
        Total number of ensembles (components).
    n_ensembles_each : list of int
        Ensembles per axis.  Must sum to n_ensembles.
    num_axes : int
        Number of label axes (inferred from numbers2tuples if 0).
    nu : list or array of float
        Per-ensemble similarity strength.
    lambda_similarity : float
        Global similarity regularization weight.
    factor_A : float
        Normalization target for A columns.
    func_normalize_A : callable
        np.sum, np.max, or identity.
    decor_A : float
        Decorrelation penalty between ensembles.
    num_repeats : int
        Number of alternating update iterations.
    cont_axis_list : list of int
        Which axes are continuous (e.g. trial number).
    value_nu_fixed : float
        Nu value for ensembles whose axis label matches.
    params_init_A : dict
        DictionaryLearning init parameters.
    params_basis_pattern : dict
        Continuous-axis basis pattern parameters.
    split_A : bool
        If True, infer a separate A per condition on each axis.
    another_update_for_A : bool
        Extra refinement pass on A.
    style_infer_phi : str
        'LS' or 'dynamic_prior'.
    params_update_phi : dict
        Extra parameters for Phi update.
    seed : int
    verbose : bool
    max_trials_each : int
        Max trials per condition used for initialization.

    Returns
    -------
    result : dict with keys
        'Phi'   : np.ndarray (T, P, M) -- temporal traces
        'A'     : np.ndarray (N, P, K) -- spatial components per unique label
        'A_full': np.ndarray (N, P, M) -- A expanded to all trials
        'params': dict of run parameters
    """
    # ----------------------------------------------------------------
    # 0. Validate inputs
    # ----------------------------------------------------------------
    assert data.ndim == 3, 'data must be 3D (N, T, M), got ndim=%d' % data.ndim
    N, T, M = data.shape
    n_trials = M
    assert len(labels) == M, (
        'len(labels) %d != data.shape[2] %d' % (len(labels), M)
    )
    assert len(numbers2tuples) > 0, 'numbers2tuples must not be empty'

    # infer num_axes
    tuple_lens = np.array([len(v) for v in numbers2tuples.values()])
    assert (tuple_lens == tuple_lens[0]).all(), (
        'all tuples in numbers2tuples must have the same length, got %s' % str(tuple_lens)
    )
    if num_axes == 0:
        num_axes = tuple_lens[0]
    assert num_axes == tuple_lens[0], (
        'num_axes %d != tuple length %d' % (num_axes, tuple_lens[0])
    )

    # nu
    if check_empty_list(nu) or len(nu) == 0:
        nu = np.ones(n_ensembles) * 0.01
    nu = np.array(nu, dtype=float)
    assert len(nu) == n_ensembles, (
        'len(nu) %d != n_ensembles %d' % (len(nu), n_ensembles)
    )
    assert (nu >= 0).all(), 'all nu values must be >= 0'

    # n_ensembles_each
    if check_empty_list(n_ensembles_each) or len(n_ensembles_each) == 0:
        n_ensembles_each = [
            len(seg) for seg in np.array_split(np.arange(n_ensembles), num_axes)
        ]
    if not isinstance(n_ensembles_each, list):
        n_ensembles_each = list(n_ensembles_each)
    if len(n_ensembles_each) == 1 and num_axes > 1:
        n_ensembles_each = n_ensembles_each * num_axes
    assert sum(n_ensembles_each) == n_ensembles, (
        'sum(n_ensembles_each)=%d != n_ensembles=%d' % (sum(n_ensembles_each), n_ensembles)
    )

    # axis-to-ensemble mapping
    cumsum = np.cumsum([0] + n_ensembles_each)
    axes2ensembles = {
        ax: np.arange(cumsum[ax], cumsum[ax + 1])
        for ax in range(num_axes)
    }

    discrete_axis_list = np.setdiff1d(np.arange(num_axes), np.array(cont_axis_list))
    labels_unique_order = make_labels_unique_order(labels)
    num_unique_conditions = len(labels_unique_order)

    if verbose:
        print('N=%d, T=%d, M=%d, P=%d, num_axes=%d, num_unique=%d'
              % (N, T, M, n_ensembles, num_axes, num_unique_conditions))

    # ----------------------------------------------------------------
    # 1. Build regularization structures
    # ----------------------------------------------------------------
    # basis patterns for continuous axes
    label_distance_to_basis_pattern_values = {}
    if len(cont_axis_list) > 0:
        wmin = params_basis_pattern.get('weight_min', np.mean(nu))
        wmax = params_basis_pattern.get('weight_max', np.mean(nu))
        if isinstance(wmin, str) and wmin == 'nu':
            wmin = np.mean(nu)
        if isinstance(wmax, str) and wmax == 'nu':
            wmax = np.mean(nu)
        params_basis_pattern['weight_min'] = wmin
        params_basis_pattern['weight_max'] = wmax

        _, params_basis_pattern, label_distance_to_basis_pattern_values = \
            create_basis_patterns(
                labels, numbers2tuples,
                cont_labels=np.vstack([numbers2tuples[lab] for lab in labels]),
                cont_axis_list=cont_axis_list,
                params_basis_pattern=params_basis_pattern,
                value_nu_fixed=value_nu_fixed,
            )

    # nu matrices
    nu_full_each_axes_dict = build_nu_matrices(
        labels, numbers2tuples, n_ensembles, n_ensembles_each,
        nu, cont_axis_list, discrete_axis_list,
        value_nu_fixed, params_basis_pattern,
        label_distance_to_basis_pattern_values,
    )

    # ----------------------------------------------------------------
    # 2. Initialize A and Phi
    # ----------------------------------------------------------------
    # Condition-aware initialization: for each axis, find directions
    # that maximize between-group variance for that axis's values.
    A_init = _condition_aware_init(
        data, labels, numbers2tuples, n_ensembles, n_ensembles_each,
        num_axes, labels_unique_order, seed, verbose,
    )
    assert A_init.shape == (N, n_ensembles), (
        'A_init shape %s != (%d, %d)' % (str(A_init.shape), N, n_ensembles)
    )

    data_hstack_all = np.hstack([data[:, :, j] for j in range(n_trials)])
    Phi_flat = solve_ls_for_phi(data_hstack_all, A_init)  # (P, T*M)

    # normalize A
    if func_normalize_A is not identity:
        sums = func_normalize_A(np.abs(A_init), 0)
        sums_safe = sums + 1e-18
        A_init = A_init * factor_A / sums_safe.reshape(1, -1)
        Phi_flat = Phi_flat / factor_A * sums_safe.reshape(-1, 1)

    # reshape Phi to 3D
    edges = np.linspace(0, Phi_flat.shape[1], n_trials + 1).astype(int)
    Phi_3d = np.dstack([Phi_flat[:, e1:e2].T for e1, e2 in zip(edges[:-1], edges[1:])])
    # Phi_3d: (T, P, M)
    assert Phi_3d.shape == (T, n_ensembles, M), (
        'Phi_3d shape %s != (%d, %d, %d)' % (str(Phi_3d.shape), T, n_ensembles, M)
    )

    full_A_hat = A_init.copy()  # shared init: (N, P)

    if verbose:
        print('Initialization done. A shape %s, Phi shape %s'
              % (str(full_A_hat.shape), str(Phi_3d.shape)))

    # ----------------------------------------------------------------
    # 3. Solver configuration
    # ----------------------------------------------------------------
    ensemble_positive = params_init_A.get('ensemble_positive', True)
    solver_name = 'nnls' if ensemble_positive else 'inv'
    solver_params = {'solver': solver_name, 'l1': 0.0, 'seed': seed}

    phi_solver_params = {**{'solver': 'nnls', 'l1': 0.0, 'seed': seed + 4}, **params_update_phi}

    # ----------------------------------------------------------------
    # 4. Alternating A and Phi updates
    # ----------------------------------------------------------------
    additional_outputs = {'Q': None}
    Phi_current = Phi_3d.copy()
    A_individual = None  # will be set in first iteration

    for outer_rep in range(num_repeats):
        if verbose:
            print('--- Outer iteration %d/%d ---' % (outer_rep + 1, num_repeats))

        # 4a. Update A given current Phi
        A_individual = _infer_A_split(
            data, labels, Phi_current, full_A_hat,
            labels_unique_order, num_unique_conditions, n_ensembles,
            axes2ensembles, numbers2tuples, nu, nu_full_each_axes_dict,
            lambda_similarity, decor_A, factor_A, func_normalize_A,
            solver_name, seed, 1, another_update_for_A,
            cont_axis_list, params_basis_pattern,
            verbose=False,
            A_tensor_init=A_individual,  # carry forward from previous iter
        )

        # 4b. Update Phi given current A
        Phi_current = _update_phi_all_conditions(
            data, labels, labels_unique_order, A_individual,
            Phi_current, n_ensembles, style_infer_phi, phi_solver_params,
            additional_outputs,
        )

        if verbose:
            A_full_tmp = _make_A_full(A_individual, labels, labels_unique_order)
            r2_tmp = _sanity_check_reconstruction(data, A_full_tmp, Phi_current, verbose=False)
            print('  R^2 = %.4f' % r2_tmp)

    assert A_individual.shape == (N, n_ensembles, num_unique_conditions), (
        'A_individual shape %s != (%d, %d, %d)'
        % (str(A_individual.shape), N, n_ensembles, num_unique_conditions)
    )
    assert Phi_current.shape == (T, n_ensembles, M), (
        'Phi shape %s != (%d, %d, %d)' % (str(Phi_current.shape), T, n_ensembles, M)
    )
    Phi_updated = Phi_current

    # ----------------------------------------------------------------
    # 5. Build full A (expanded to per-trial)
    # ----------------------------------------------------------------
    A_full = _make_A_full(A_individual, labels, labels_unique_order)
    assert A_full.shape == (N, n_ensembles, M), (
        'A_full shape %s != (%d, %d, %d)' % (str(A_full.shape), N, n_ensembles, M)
    )

    # ----------------------------------------------------------------
    # 6. Sanity checks
    # ----------------------------------------------------------------
    _sanity_check_reconstruction(data, A_full, Phi_updated, verbose)

    params_save = {
        'n_ensembles': n_ensembles, 'n_ensembles_each': n_ensembles_each,
        'num_axes': num_axes, 'nu': nu, 'lambda_similarity': lambda_similarity,
        'factor_A': factor_A, 'decor_A': decor_A, 'num_repeats': num_repeats,
        'cont_axis_list': cont_axis_list, 'split_A': split_A,
        'style_infer_phi': style_infer_phi, 'seed': seed,
        'labels_unique_order': labels_unique_order,
        'numbers2tuples': numbers2tuples,
    }

    return {
        'Phi': Phi_updated,
        'A': A_individual,
        'A_full': A_full,
        'params': params_save,
    }


# ==================================================================
#  Private helpers
# ==================================================================

def _make_A_full(A_individual, labels, labels_unique_order):
    """Expand A from (N, P, K_unique) to (N, P, M) by label mapping."""
    labels_unique_list = list(labels_unique_order)
    slices = []
    for lab in labels:
        idx = labels_unique_list.index(lab)
        slices.append(A_individual[:, :, idx])
    return np.dstack(slices)


def _condition_aware_init(data, labels, numbers2tuples, n_ensembles,
                          n_ensembles_each, num_axes, labels_unique_order,
                          seed, verbose):
    """
    Initialize A using per-axis between-group variance.

    For each axis, compute group means across that axis's values,
    then use SVD on the between-group contrast matrix to find
    directions that capture axis-specific variation.
    """
    N, T, M = data.shape
    rng = np.random.RandomState(seed)
    cumsum = np.cumsum([0] + list(n_ensembles_each))
    labels_arr = np.array([numbers2tuples[lab] for lab in labels])

    A_init = np.zeros((N, n_ensembles))

    for ax in range(num_axes):
        e1, e2 = cumsum[ax], cumsum[ax + 1]
        n_ens = e2 - e1
        axis_vals = np.unique(labels_arr[:, ax])

        # compute mean data per axis-value
        group_means = []
        for val in axis_vals:
            trial_idx = np.where(labels_arr[:, ax] == val)[0]
            # mean over trials and time: (N,)
            mean_profile = np.mean(data[:, :, trial_idx], axis=(1, 2))
            group_means.append(mean_profile)
        group_means = np.array(group_means)  # (n_vals, N)

        # also build a contrast matrix: each group's deviation from grand mean
        grand_mean = group_means.mean(0)
        contrasts = group_means - grand_mean.reshape(1, -1)  # (n_vals, N)

        # also stack time-resolved group means for richer information
        group_time_means = []
        for val in axis_vals:
            trial_idx = np.where(labels_arr[:, ax] == val)[0]
            mean_time = np.mean(data[:, :, trial_idx], axis=2)  # (N, T)
            group_time_means.append(mean_time)
        # stack: (N, T * n_vals)
        stacked = np.hstack(group_time_means)

        # SVD on the stacked group means
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        # take top n_ens components
        A_block = np.abs(U[:, :n_ens]) * S[:n_ens].reshape(1, -1)

        # normalize columns
        col_norms = np.sum(np.abs(A_block), axis=0) + 1e-18
        A_block = A_block / col_norms.reshape(1, -1)

        A_init[:, e1:e2] = A_block

    # add small noise to break symmetry
    A_init += rng.randn(N, n_ensembles) * 0.01
    A_init = np.abs(A_init)

    if verbose:
        print('Condition-aware init: A shape %s' % str(A_init.shape))

    return A_init


def _sanity_check_reconstruction(data, A_full, Phi, verbose=False):
    """Check that A @ Phi.T gives a finite reconstruction."""
    N, T, M = data.shape
    # spot-check a few trials
    trials_to_check = np.linspace(0, M - 1, min(5, M)).astype(int)
    for trial in trials_to_check:
        reco = A_full[:, :, trial] @ Phi[:, :, trial].T
        assert reco.shape == (N, T), (
            'reco shape %s != (%d, %d)' % (str(reco.shape), N, T)
        )
        assert np.all(np.isfinite(reco)), (
            'non-finite values in reconstruction for trial %d' % trial
        )
    # global R^2
    reco_all = np.dstack([
        A_full[:, :, m] @ Phi[:, :, m].T for m in range(M)
    ])
    ss_res = np.sum((data - reco_all) ** 2)
    ss_tot = np.sum((data - data.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-18)
    if verbose:
        print('Global R^2 = %.4f' % r2)
    if r2 < -1.0:
        import warnings
        warnings.warn(
            'Reconstruction R^2 = %.4f is very poor. '
            'Consider adjusting hyperparameters.' % r2
        )
    return r2


def _infer_A_split(data, labels, Phi_3d, full_A_hat,
                   labels_unique_order, num_unique_conditions, n_ensembles,
                   axes2ensembles, numbers2tuples, nu, nu_full_each_axes_dict,
                   lambda_similarity, decor_A, factor_A, func_normalize_A,
                   solver_name, seed, num_repeats, another_update_for_A,
                   cont_axis_list, params_basis_pattern, verbose,
                   A_tensor_init=None):
    """
    Infer per-condition A with multi-axis similarity regularization.

    This is the core of MILCCI: for each axis, the ensembles assigned
    to that axis share A across all trials with the same axis-value,
    regularized to be similar to trials with nearby axis-values.
    """
    N = data.shape[0]
    n_trials = data.shape[2]
    num_axes = len(list(numbers2tuples.values())[0])
    axes = np.arange(num_axes)

    # initialize: use previous A if available, else replicate shared A
    if A_tensor_init is not None:
        A_tensor = A_tensor_init.copy()
        assert A_tensor.shape == (N, n_ensembles, num_unique_conditions), (
            'A_tensor_init shape %s' % str(A_tensor.shape)
        )
    else:
        A_tensor = np.repeat(np.expand_dims(full_A_hat, 2), num_unique_conditions, axis=2)
        assert A_tensor.shape == (N, n_ensembles, num_unique_conditions), (
            'A_tensor init shape %s' % str(A_tensor.shape)
        )

    # first pass: per-axis A update
    axis_label_visited = []
    for c, label in enumerate(labels_unique_order):
        labels_loc = np.where(labels == label)[0]
        for ax in axes:
            label_axis = numbers2tuples[label][ax]

            # skip if we already updated this axis-value
            key = (ax, label_axis)
            if key in axis_label_visited:
                continue
            axis_label_visited.append(key)

            A_fixed_indices = axes2ensembles[ax]
            A_changing_indices = np.setdiff1d(np.arange(n_ensembles), A_fixed_indices)
            n_fixed = len(A_fixed_indices)

            # which conditions share this axis-value?
            cond_indices = np.array([
                li for li, lab in enumerate(labels_unique_order)
                if numbers2tuples[lab][ax] == label_axis
            ])
            # which trials have this axis-value?
            trial_indices = np.array([
                ti for ti, lab in enumerate(labels)
                if numbers2tuples[lab][ax] == label_axis
            ])
            # conditions that differ
            other_cond_indices = np.setdiff1d(np.arange(num_unique_conditions), cond_indices)

            # build system: right @ A_fixed = left
            # iterate over ALL trials with this axis-value
            right_blocks = []
            left_blocks = []
            labels_unique_list = list(labels_unique_order)
            for ti in trial_indices:
                # find which unique condition this trial belongs to
                ci = labels_unique_list.index(labels[ti])
                phi_ti = Phi_3d[:, A_fixed_indices, ti]  # (T, n_fixed)
                right_blocks.append(phi_ti)
                # subtract contribution of changing ensembles
                extra = (A_tensor[:, A_changing_indices, ci]
                         @ Phi_3d[:, A_changing_indices, ti].T)  # (N, T)
                residual = data[:, :, ti].T - extra.T  # (T, N)
                left_blocks.append(residual)

            # similarity regularization
            if len(other_cond_indices) > 0:
                nu_axis = nu[A_fixed_indices]
                right_blocks.append(
                    lambda_similarity * np.eye(n_fixed) * nu_axis.reshape(1, -1)
                )
                A_others_mean = A_tensor[:, A_fixed_indices, :][:, :, other_cond_indices].mean(2)
                left_blocks.append(
                    lambda_similarity * (A_others_mean * nu_axis.reshape(1, -1)).T
                )

            right = np.vstack(right_blocks)
            left = np.vstack(left_blocks)

            # decorrelation
            if decor_A > 0:
                right = np.vstack([
                    right,
                    decor_A * (np.ones((n_fixed, n_fixed)) - np.eye(n_fixed))
                ])
                left = np.vstack([left, np.zeros((n_fixed, N))])

            # remove zero rows
            nonzero = np.where(
                (np.abs(left).sum(1) > 0) | (np.abs(right).sum(1) > 0)
            )[0]
            if len(nonzero) < right.shape[0]:
                right = right[nonzero]
                left = left[nonzero]

            addi = solve_regularized(right, left, solver=solver_name, seed=seed).T
            assert addi.shape == (N, n_fixed), (
                'addi shape %s != (%d, %d)' % (str(addi.shape), N, n_fixed)
            )

            for ci in cond_indices:
                A_tensor[:, A_fixed_indices, ci] = addi

    # normalize A
    if func_normalize_A is not identity:
        sums = func_normalize_A(np.abs(A_tensor), axis=0)
        sums_safe = (np.expand_dims(sums, 0) + 1e-18) / factor_A
        A_tensor = A_tensor / sums_safe

    # second pass: full condition-level refinement
    if another_update_for_A:
        cond_array = np.arange(num_unique_conditions)
        for rep in range(num_repeats):
            for c, label in enumerate(labels_unique_order):
                cur_nu_mat = nu_full_each_axes_dict[label]
                non_c = cond_array[cond_array != c]
                cur_nu_non_c = cur_nu_mat[:, non_c]

                A_others = A_tensor[:, :, non_c]
                A_others_vstack = np.vstack([
                    A_others[:, :, layer].T
                    for layer in range(len(non_c))
                ])
                nus_list = [cur_nu_non_c[:, col].reshape(-1, 1) for col in range(len(non_c))]
                nus_vstack = np.vstack(nus_list)
                left_nus = np.vstack([np.diag(n.flatten()) for n in nus_list])

                labels_loc = np.where(labels == label)[0]
                phi_label = Phi_3d[:, :, labels_loc]
                phi_label_2d = np.vstack([
                    phi_label[:, :, tr] for tr in range(phi_label.shape[2])
                ])
                data_label = data[:, :, labels_loc]
                data_label_2d = np.hstack([
                    data_label[:, :, tr] for tr in range(data_label.shape[2])
                ])

                left = np.vstack([phi_label_2d, lambda_similarity * left_nus])
                right = np.vstack([data_label_2d.T, lambda_similarity * A_others_vstack * nus_vstack])

                nonzero = np.where(
                    (np.abs(left).sum(1) > 0) | (np.abs(right).sum(1) > 0)
                )[0]
                left = left[nonzero]
                right = right[nonzero]

                addi = solve_regularized(left, right, solver=solver_name, seed=seed)
                A_tensor[:, :, c] = addi.T

    return A_tensor


def _update_phi_all_conditions(data, labels, labels_unique_order, A_individual,
                                Phi_3d, n_ensembles, style_infer_phi,
                                solver_params, additional_outputs):
    """Update Phi for all conditions given refined A."""
    N, T, M = data.shape
    Phi_new_blocks = []
    labels_check = []

    for c, label in enumerate(labels_unique_order):
        assert check_if_labels_batches(labels), (
            'labels must be sorted in contiguous batches for Phi update'
        )
        cur_A = A_individual[:, :, c]  # (N, P)
        where_label = np.where(labels == label)[0]
        labels_check.extend([label] * len(where_label))

        Phi_label, additional_outputs_new = infer_phi(
            cur_A, data, trial_indices=where_label,
            style=style_infer_phi,
            solver_params=solver_params,
            Phi_init=Phi_3d,
            Q_init=additional_outputs.get('Q'),
        )
        additional_outputs.update(additional_outputs_new)
        Phi_new_blocks.append(Phi_label)

    assert (np.array(labels_check) == np.array(labels)).all(), (
        'label order mismatch after Phi update'
    )
    Phi_updated = np.dstack(Phi_new_blocks)
    return Phi_updated


def reconstruct(A_full, Phi):
    """
    Compute reconstruction Y_hat[:,:,m] = A_full[:,:,m] @ Phi[:,:,m].T

    Parameters
    ----------
    A_full : np.ndarray, shape (N, P, M)
    Phi : np.ndarray, shape (T, P, M)

    Returns
    -------
    Y_hat : np.ndarray, shape (N, T, M)
    """
    M = A_full.shape[2]
    assert Phi.shape[2] == M, 'Phi trials %d != A trials %d' % (Phi.shape[2], M)
    Y_hat = np.dstack([A_full[:, :, m] @ Phi[:, :, m].T for m in range(M)])
    assert Y_hat.ndim == 3, 'Y_hat must be 3D'
    return Y_hat

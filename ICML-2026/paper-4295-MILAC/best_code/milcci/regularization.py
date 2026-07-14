# -*- coding: utf-8 -*-
"""
Regularization utilities for MILCCI.

Handles construction of:
- basis patterns for continuous-axis temporal regularization
- per-condition nu matrices that encode which ensembles should be
  similar across which condition pairs
"""
import numpy as np
from .utils import check_empty_list, make_labels_unique_order


def create_basis_patterns(labels, numbers2tuples, cont_labels=[],
                          cont_axis_list=[1], params_basis_pattern={},
                          value_nu_fixed=1.0):
    """
    Build a windowed weight template for continuous-variable regularization.

    The returned *label_distance_to_basis_pattern_values* maps integer
    label-distance to regularization weight, so that nearby trials are
    pulled toward each other more strongly.

    Parameters
    ----------
    labels : array-like
        Numeric label per trial.
    numbers2tuples : dict
        {label_number: tuple_of_axis_values}.
    cont_labels : array-like, optional
        Explicit continuous values per trial.
    cont_axis_list : list of int
        Which tuple axes are continuous.
    params_basis_pattern : dict
        'wind_size', 'weight_func', 'weight_min', 'weight_max',
        'one_or_two_sides'.
    value_nu_fixed : float
        Fixed reference nu (must be > weight_min).

    Returns
    -------
    basis_pattern : np.ndarray
    params_basis_pattern : dict  (updated)
    label_distance_to_basis_pattern_values : dict
    """
    defaults = {
        'wind_size': 5,
        'weight_func': 'linear',
        'weight_min': 0,
        'weight_max': 0,
        'one_or_two_sides': -1,
    }
    params_basis_pattern = {**defaults, **params_basis_pattern}

    if check_empty_list(cont_labels):
        cont_labels = np.vstack([numbers2tuples[lab] for lab in labels])
    assert len(cont_labels) == len(labels), (
        'cont_labels length %d != labels length %d' % (len(cont_labels), len(labels))
    )

    if cont_labels.ndim == 1 or max(cont_labels.shape) == cont_labels.size:
        cont_labels = cont_labels.reshape(-1, 1)
        cont_axis_list = [0]

    wf = params_basis_pattern['weight_func']
    assert wf in ['linear', 'log', 'exp'], (
        'weight_func must be linear/log/exp, got %s' % wf
    )

    wmin = params_basis_pattern['weight_min']
    wmax = params_basis_pattern['weight_max']
    assert not (wmin == 0 and wmax == 0), (
        'weight_min and weight_max cannot both be 0'
    )
    if wmin == 0:
        wmin = wmax / params_basis_pattern['wind_size']
        params_basis_pattern['weight_min'] = wmin
    if wmax == 0:
        wmax = wmin * params_basis_pattern['wind_size']
        params_basis_pattern['weight_max'] = wmax

    ws = params_basis_pattern['wind_size']
    if isinstance(ws, str):
        assert ws == 'all', "wind_size as string must be 'all', got %s" % ws
        ws = int(np.floor((len(labels) - 1) / 2))
        params_basis_pattern['wind_size'] = ws
    elif ws * 2 + 1 > len(labels):
        ws = int(np.floor((len(labels) - 1) / 2))
        params_basis_pattern['wind_size'] = ws

    params_basis_pattern['wind_size_full_len'] = ws * 2 + 1

    assert value_nu_fixed > wmin, (
        'value_nu_fixed %.4f must be > weight_min %.4f' % (value_nu_fixed, wmin)
    )

    half = np.linspace(wmin, wmax, ws)

    if wf != 'linear':
        if wf == 'exp':
            half = np.exp(half)
        elif wf == 'log':
            half = np.log(half + 1e-12)
        half = (half - half.min()) / (half.max() + 1e-18) * wmax

    sides = params_basis_pattern['one_or_two_sides']
    if sides == 2:
        basis_pattern = np.hstack([half, np.array([0]), half[::-1]])
    elif sides == 1:
        basis_pattern = np.hstack([np.zeros(len(half)), np.array([0]), half[::-1]])
    elif sides == -1:
        basis_pattern = np.hstack([half, np.array([0]), np.zeros(len(half))])
    else:
        raise ValueError('one_or_two_sides must be -1, 1, or 2, got %d' % sides)

    arange = np.linspace(-ws, ws, params_basis_pattern['wind_size_full_len']).astype(int)
    assert len(arange) == len(basis_pattern), (
        'arange length %d != basis_pattern length %d' % (len(arange), len(basis_pattern))
    )

    label_distance_to_basis_pattern_values = {
        d: v for d, v in zip(arange, basis_pattern)
    }
    return basis_pattern, params_basis_pattern, label_distance_to_basis_pattern_values


def build_nu_matrices(labels, numbers2tuples, n_ensembles, n_ensembles_each,
                      nu, cont_axis_list, discrete_axis_list,
                      value_nu_fixed, params_basis_pattern,
                      label_distance_to_basis_pattern_values={}):
    """
    Build per-condition nu matrices that encode pairwise ensemble similarity.

    Returns
    -------
    nu_full_each_axes_dict : dict
        {label: np.ndarray of shape (n_ensembles, n_unique_labels)}
        Entry [j, u] says how strongly ensemble j should be similar between
        the current label and label u.
    """
    labels_unique_order = make_labels_unique_order(labels)
    num_unique = len(labels_unique_order)
    num_axes = len(list(numbers2tuples.values())[0])

    n_ensembles_each_cumsum = np.cumsum(np.array([0] + list(n_ensembles_each)))

    # per-axis nu: column 0 = learned nu, column 1 = fixed value
    nu_each_axes_2d = [
        np.hstack([
            nu[e1:e2].reshape(-1, 1),
            value_nu_fixed * np.ones(e2 - e1).reshape(-1, 1)
        ])
        for e1, e2 in zip(n_ensembles_each_cumsum[:-1], n_ensembles_each_cumsum[1:])
    ]

    nu_full_each_axes_dict = {}
    for lc, label in enumerate(labels_unique_order):
        cur_mat = np.zeros((n_ensembles, num_unique))
        for lc2, label2 in enumerate(labels_unique_order):
            if lc2 == lc:
                # diagonal = 0 (no self-regularization)
                pass
            else:
                tup1 = numbers2tuples[label]
                tup2 = numbers2tuples[label2]
                parts = []
                for ax in range(num_axes):
                    if ax in discrete_axis_list:
                        # same value on this axis -> strong coupling (col 1)
                        ind = 1 if tup1[ax] == tup2[ax] else 0
                        parts.append(nu_each_axes_2d[ax][:, ind].reshape(-1, 1))
                    else:
                        # continuous axis
                        diff = tup1[ax] - tup2[ax]
                        ws = params_basis_pattern['wind_size']
                        if np.abs(diff) <= ws and diff in label_distance_to_basis_pattern_values:
                            val = label_distance_to_basis_pattern_values[diff]
                            parts.append(
                                val * np.ones((n_ensembles_each[ax], 1))
                            )
                        else:
                            parts.append(np.zeros((n_ensembles_each[ax], 1)))
                cur_mat[:, lc2] = np.vstack(parts).flatten()
        nu_full_each_axes_dict[label] = cur_mat

    return nu_full_each_axes_dict

# -*- coding: utf-8 -*-
"""
Utility functions for MILCCI.
"""
import numpy as np
import warnings


def check_empty_list(obj):
    """Return True if obj is an empty list."""
    return isinstance(obj, list) and len(obj) == 0


def make_labels_unique_order(labels, make_array=True):
    """
    Return unique labels preserving first-appearance order.

    Parameters
    ----------
    labels : list or array-like
    make_array : bool
        If True return np.array, else return list.

    Returns
    -------
    np.array or list
    """
    visited = []
    for lab in labels:
        if lab not in visited:
            visited.append(lab)
    if make_array:
        return np.array(visited)
    return visited


def find_indices_in_list(lst, element):
    """Return all indices where *element* appears in *lst*."""
    return [i for i, el in enumerate(lst) if el == element]


def lists2list(xss):
    """Flatten a list of lists into a single list."""
    return [x for xs in xss for x in xs]


def check_if_labels_batches(labels):
    """
    Return True if labels are arranged in contiguous batches
    (no label reappears after a different one has started).

    Example
    -------
    >>> check_if_labels_batches([0, 0, 1, 1, 2])
    True
    >>> check_if_labels_batches([0, 1, 0])
    False
    """
    visited = []
    for l1, l2 in zip(labels[:-1], labels[1:]):
        if l1 != l2:
            visited.append(l1)
        if l2 in visited:
            return False
    return True


def spec_corr(v1, v2, to_abs=True):
    """Pearson correlation between two flat vectors."""
    corr = np.corrcoef(v1.ravel(), v2.ravel())
    if to_abs:
        return np.abs(corr[0, 1])
    return corr[0, 1]


def identity(mat):
    """Identity function, returns input unchanged."""
    return mat


def normalize_A_columns(full_A, normalize_A_style='avg', epsilon=1e-9):
    """
    Normalize columns of A across the neuron axis (axis 0).

    Parameters
    ----------
    full_A : np.ndarray
        Shape (N, P) or (N, P, K).
    normalize_A_style : str
        'avg' or 'max'.
    epsilon : float

    Returns
    -------
    full_A_normalized : np.ndarray
    normalize_values : np.ndarray
    """
    if normalize_A_style == 'avg':
        norms = np.mean(np.abs(full_A), axis=0) + epsilon
    elif normalize_A_style == 'max':
        norms = np.max(np.abs(full_A), axis=0) + epsilon
    else:
        raise ValueError('Unknown normalize_A_style: %s' % normalize_A_style)

    full_A_normalized = full_A / norms[np.newaxis, ...] if full_A.ndim == 2 else full_A / norms[np.newaxis, :]
    return full_A_normalized, norms

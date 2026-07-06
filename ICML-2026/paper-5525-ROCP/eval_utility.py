"""
Evaluation utilities for conformal decision experiments.
"""

from typing import Iterable, List, Mapping, Sequence, Set, Dict, Any
import numpy as np


def _action_to_index(actions: Sequence[object]) -> Mapping[object, int]:
    return {a: i for i, a in enumerate(actions)}


def averaged_realized_worst_case_risk(
    pred_sets: Sequence[Set[int]],
    actions_pred: Sequence[object],
    actions: Sequence[object],
    loss_matrix: Sequence[Sequence[float]],
    alpha: float,
) -> float:
    """
    Average of L_S(a_method) over test points, where
      L_S(a) = ell_in + alpha * (ell_out - ell_in)_+
      ell_in  = max_{y in S} loss[y,a]
      ell_out = max_{y not in S} loss[y,a]

    Fallback for empty S: define L_∅(a) = max_{y in Y} loss[y,a],
    so that min_a L_∅(a) = min_a max_y loss[y,a].

    Args:
        pred_sets: iterable of prediction sets C(x).
        actions_pred: actions chosen by a method for each C(x).
        actions: action list (used to map action objects to indices).
        loss_matrix: array-like [num_labels, num_actions].
        alpha: miscoverage level in (0,1).
    """
    loss_mat = np.asarray(loss_matrix, dtype=float)
    num_labels, num_actions = loss_mat.shape
    total = 0.0
    n = len(pred_sets)
    if len(actions_pred) != n:
        raise ValueError("pred_sets and actions_pred must have the same length.")

    a2i = _action_to_index(actions)

    def to_idx(a: object) -> int:
        if a in a2i:
            return a2i[a]
        if isinstance(a, (int, np.integer)):
            ai = int(a)
            if 0 <= ai < len(actions):
                return ai
        raise ValueError("Unrecognized action {!r}; must be in actions or a valid index.".format(a))

    for S, a in zip(pred_sets, actions_pred):
        a_idx = to_idx(a)
        subset = np.fromiter(S, dtype=int)
        comp = np.setdiff1d(np.arange(num_labels), subset, assume_unique=True)

        if subset.size:
            ell_in = float(np.max(loss_mat[subset, a_idx]))
        else:
            # fallback for empty set: max over all labels
            ell_in = float(np.max(loss_mat[:, a_idx]))
        ell_out = float(np.max(loss_mat[comp, a_idx])) if comp.size else 0.0
        total += ell_in + alpha * max(ell_out - ell_in, 0.0)
    return total / max(n, 1)


def averaged_realized_loss(
    actions_pred: Sequence[object],
    true_labels: Sequence[int],
    actions: Sequence[object],
    loss_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Average realized loss: mean of loss(y_true, a_pred).
    """
    loss_mat = np.asarray(loss_matrix, dtype=float)
    a2i = _action_to_index(actions)
    total = 0.0
    n = len(true_labels)
    for a, y in zip(actions_pred, true_labels):
        a_idx = a2i[a] if a in a2i else int(a)
        total += float(loss_mat[int(y), a_idx])
    return total / max(n, 1)


def critical_mistake_rates(
    actions_pred: Sequence[object],
    true_labels: Sequence[int],
    actions: Sequence[object],
    critical_labels: Iterable[int],
    loss_matrix: Sequence[Sequence[float]],
) -> Dict[int, float]:
    """
    Fraction per critical label where action equals the worst (max-loss) action
    for that label: a_idx == argmax_a loss[y, a].
    Returns dict {label: rate}.
    """
    a2i = _action_to_index(actions)
    loss_mat = np.asarray(loss_matrix, dtype=float)
    if loss_mat.shape[1] != len(actions):
        raise ValueError("loss_matrix second dimension must match actions.")

    def to_idx(a: object) -> int:
        # direct match in action list
        if a in a2i:
            return a2i[a]
        # integer index fallback
        if isinstance(a, (int, np.integer)):
            ai = int(a)
            if 0 <= ai < len(actions):
                return ai
        raise ValueError(f"Action {a!r} not recognized; must be in actions or a valid index.")

    crit_set = set(int(c) for c in critical_labels)
    counts = {c: 0 for c in crit_set}
    mistakes = {c: 0 for c in crit_set}
    for a, y in zip(actions_pred, true_labels):
        y_int = int(y)
        if y_int not in crit_set:
            continue
        counts[y_int] += 1
        a_idx = to_idx(a)
        worst_idx = int(np.argmax(loss_mat[y_int]))
        if a_idx == worst_idx:
            mistakes[y_int] += 1
    return {c: (mistakes[c] / counts[c]) if counts[c] else 0.0 for c in crit_set}


def critical_bad_action_rates(
    actions_pred: Sequence[object],
    true_labels: Sequence[int],
    actions: Sequence[object],
    critical_labels: Iterable[int],
    loss_matrix: Sequence[Sequence[float]],
    loss_threshold: float,
) -> Dict[int, float]:
    """
    Fraction per critical label where the chosen action is "bad" according to a
    loss threshold:

        bad(y, a) := 1{ loss[y, a] >= loss_threshold }.

    Returns dict {label: rate}.
    """
    a2i = _action_to_index(actions)
    loss_mat = np.asarray(loss_matrix, dtype=float)
    if loss_mat.shape[1] != len(actions):
        raise ValueError("loss_matrix second dimension must match actions.")

    def to_idx(a: object) -> int:
        # direct match in action list
        if a in a2i:
            return a2i[a]
        # integer index fallback
        if isinstance(a, (int, np.integer)):
            ai = int(a)
            if 0 <= ai < len(actions):
                return ai
        raise ValueError(f"Action {a!r} not recognized; must be in actions or a valid index.")

    crit_set = set(int(c) for c in critical_labels)
    counts = {c: 0 for c in crit_set}
    bad = {c: 0 for c in crit_set}
    th = float(loss_threshold)

    for a, y in zip(actions_pred, true_labels):
        y_int = int(y)
        if y_int not in crit_set:
            continue
        counts[y_int] += 1
        a_idx = to_idx(a)
        if float(loss_mat[y_int, a_idx]) >= th:
            bad[y_int] += 1

    return {c: (bad[c] / counts[c]) if counts[c] else 0.0 for c in crit_set}


def averaged_miscoverage(pred_sets: Sequence[Set[int]], true_labels: Sequence[int]) -> float:
    """
    Average non-coverage rate: mean of 1{y_true not in C(x)}.
    """
    misses = 0
    n = len(true_labels)
    for S, y in zip(pred_sets, true_labels):
        if int(y) not in S:
            misses += 1
    return misses / max(n, 1)


__all__ = [
    "averaged_realized_worst_case_risk",
    "averaged_realized_loss",
    "critical_mistake_rates",
    "critical_bad_action_rates",
    "averaged_miscoverage",
]

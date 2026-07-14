# -*- coding: utf-8 -*-
"""
Phi (temporal trace) inference for MILCCI.

Two modes:
  - 'LS':  standard least-squares  phi_t = pinv(A) @ y_t
  - 'dynamic_prior':  LDS-regularized  phi_{t+1} approx Q phi_t
"""
import numpy as np
from .solvers import solve_regularized


def infer_phi(A, data, trial_indices=[], style='LS',
              solver_params={}, Phi_init=None, Q_init=None,
              gamma3=1.0, gamma4=0.1, gamma5=0.01, n_inner_iters=5):
    """
    Infer temporal traces Phi for a set of trials.

    Parameters
    ----------
    A : np.ndarray, shape (N, P) or (N, P, M)
        Spatial components.  2-D means shared A across trials.
    data : np.ndarray, shape (N, T, M)
        Observations.
    trial_indices : array-like
        Which trials to process.
    style : str
        'LS' or 'dynamic_prior'.
    solver_params : dict
        Passed to solve_regularized.
    Phi_init : np.ndarray or None
        Initial Phi, shape (T, P, M).  Used by dynamic_prior.
    Q_init : np.ndarray or None
        Initial transition matrix, shape (P, P, M).
    gamma3, gamma4, gamma5 : float
        LDS prior hyperparameters.
    n_inner_iters : int

    Returns
    -------
    Phi : np.ndarray, shape (T, P, n_trials_selected)
    additional_outputs : dict
        May contain 'Q'.
    """
    assert style in ['LS', 'dynamic_prior'], (
        "style must be 'LS' or 'dynamic_prior', got '%s'" % style
    )

    if len(trial_indices) == 0:
        trial_indices = np.arange(data.shape[2])
    trial_indices = np.array(trial_indices)

    N, T, M_total = data.shape
    P = A.shape[1] if A.ndim == 2 else A.shape[1]
    n_trials = len(trial_indices)
    additional_outputs = {}

    if style == 'LS':
        # A is either (N, P) shared, or (N, P, M_total) per-trial
        phi_list = []
        for trial in trial_indices:
            A_trial = A if A.ndim == 2 else A[:, :, trial]
            phi_trial = np.vstack([
                solve_regularized(A_trial, data[:, t, trial].flatten(),
                                  **solver_params).reshape(1, -1)
                for t in range(T)
            ])
            assert phi_trial.shape == (T, P), (
                'phi_trial shape %s != (%d, %d)' % (str(phi_trial.shape), T, P)
            )
            phi_list.append(phi_trial)
        Phi = np.dstack(phi_list)
        assert Phi.shape == (T, P, n_trials), (
            'Phi shape %s != (%d, %d, %d)' % (str(Phi.shape), T, P, n_trials)
        )

    elif style == 'dynamic_prior':
        A_2d = A if A.ndim == 2 else None
        assert A_2d is not None or A.ndim == 3, 'A must be 2D or 3D'

        phi_list = []
        Q_list = []
        for i, trial in enumerate(trial_indices):
            A_t = A_2d if A_2d is not None else A[:, :, trial]
            assert A_t.ndim == 2, 'A for LDS must be 2D per trial'

            Phi_init_trial = None
            if Phi_init is not None:
                if Phi_init.shape[2] == M_total:
                    Phi_init_trial = Phi_init[:, :, trial].T  # -> (P, T)
                else:
                    Phi_init_trial = Phi_init[:, :, i].T

            Q_init_trial = None
            if Q_init is not None:
                Q_init_trial = Q_init[:, :, trial]

            Phi_t, Q_t = _update_phi_under_dynamic_prior(
                data[:, :, trial], A_t,
                gamma3=gamma3, gamma4=gamma4, gamma5=gamma5,
                n_inner_iters=n_inner_iters,
                Phi_init=Phi_init_trial, Q_init=Q_init_trial
            )
            assert Phi_t.shape == (P, T), (
                'Phi_t shape %s != (%d, %d)' % (str(Phi_t.shape), P, T)
            )
            phi_list.append(Phi_t.T)  # -> (T, P)
            Q_list.append(Q_t)

        Phi = np.dstack(phi_list)
        assert Phi.shape == (T, P, n_trials), (
            'Phi shape %s != (%d, %d, %d)' % (str(Phi.shape), T, P, n_trials)
        )

        # store Q in full-trial space
        if Q_init is None:
            Q_init = np.zeros((P, P, M_total))
        Q_init[:, :, trial_indices] = np.dstack(Q_list)
        assert (np.dstack(Q_list).sum(0).sum(0) != 0).all(), (
            'some Q transitions are all zeros'
        )
        additional_outputs['Q'] = Q_init

    return Phi, additional_outputs


# ------------------------------------------------------------------ #
#  LDS prior internals                                                #
# ------------------------------------------------------------------ #

def _update_phi_under_dynamic_prior(Y, A, gamma3=1.0, gamma4=0.1,
                                     gamma5=0.01, n_inner_iters=5,
                                     Phi_init=None, Q_init=None):
    """
    Alternating update of Phi and Q under a linear dynamical system prior.

        Phi_{t+1} ~ Q Phi_t
        Y_t = A Phi_t + noise

    Parameters
    ----------
    Y : np.ndarray, shape (N, T)
    A : np.ndarray, shape (N, P)
    gamma3 : float  -- LDS smoothness weight
    gamma4 : float  -- cross-correlation penalty weight
    gamma5 : float  -- Q regularization weight
    n_inner_iters : int
    Phi_init : np.ndarray or None, shape (P, T)
    Q_init : np.ndarray or None, shape (P, P)

    Returns
    -------
    Phi : np.ndarray, shape (P, T)
    Q : np.ndarray, shape (P, P)
    """
    assert Y.ndim == 2 and A.ndim == 2, 'Y and A must be 2D'
    N, T = Y.shape
    P = A.shape[1]
    assert A.shape[0] == N, 'A rows %d != Y rows %d' % (A.shape[0], N)

    # initialize
    if Phi_init is not None:
        assert Phi_init.shape == (P, T), (
            'Phi_init shape %s != (%d, %d)' % (str(Phi_init.shape), P, T)
        )
        Phi = Phi_init.copy()
    else:
        Phi = np.linalg.pinv(A) @ Y
        assert Phi.shape == (P, T), (
            'Phi init shape %s != (%d, %d)' % (str(Phi.shape), P, T)
        )

    if Q_init is not None:
        assert Q_init.shape == (P, P), (
            'Q_init shape %s != (%d, %d)' % (str(Q_init.shape), P, P)
        )
        Q = Q_init.copy()
    else:
        Q = np.eye(P) * 0.9

    AtA = A.T @ A
    AtY = A.T @ Y

    # cross-correlation mask
    C = np.corrcoef(Phi)
    if np.any(np.isnan(C)):
        C = np.zeros((P, P))
    mask = np.ones((P, P)) - np.eye(P)

    for iteration in range(n_inner_iters):
        # --- update Phi given Q ---
        Phi = _update_phi_given_Q(Y, A, Phi, Q, gamma3, gamma4, AtA, AtY, C, mask)
        # --- update Q given Phi ---
        Q = _estimate_transition_matrix(Phi, gamma5)
        # refresh cross-corr
        C = np.corrcoef(Phi)
        if np.any(np.isnan(C)):
            C = np.zeros((P, P))

    assert Phi.shape == (P, T), 'final Phi shape %s != (%d, %d)' % (str(Phi.shape), P, T)
    assert Q.shape == (P, P), 'final Q shape %s != (%d, %d)' % (str(Q.shape), P, P)
    return Phi, Q


def _update_phi_given_Q(Y, A, Phi, Q, gamma3, gamma4, AtA, AtY, C, mask):
    """One sweep of Phi updates given fixed Q."""
    P, T = Phi.shape
    for t in range(T):
        # data fidelity gradient
        grad = AtA @ Phi[:, t] - AtY[:, t]

        # LDS prior
        if t > 0:
            grad += gamma3 * (Phi[:, t] - Q @ Phi[:, t - 1])
        if t < T - 1:
            grad += gamma3 * Q.T @ (Q @ Phi[:, t] - Phi[:, t + 1])

        # cross-correlation penalty
        D = np.zeros(P)
        for p in range(P):
            D[p] = np.sum((C[p, :] * mask[p, :]) * Phi[:, t])
        grad += gamma4 * D

        # step size (simple diagonal scaling)
        diag = np.diag(AtA) + gamma3 * (1.0 + np.sum(Q ** 2, axis=0)) + gamma4 + 1e-8
        Phi[:, t] -= grad / diag

    return Phi


def _estimate_transition_matrix(Phi, gamma5):
    """Estimate Q from  Phi_{t+1} ~ Q Phi_t  with L2 regularization."""
    P, T = Phi.shape
    assert T > 1, 'need at least 2 time points to estimate Q'

    Phi_prev = Phi[:, :-1]  # (P, T-1)
    Phi_next = Phi[:, 1:]   # (P, T-1)

    reg = gamma5 * np.eye(P)
    Q = Phi_next @ Phi_prev.T @ np.linalg.inv(Phi_prev @ Phi_prev.T + reg)

    assert Q.shape == (P, P), 'Q shape %s != (%d, %d)' % (str(Q.shape), P, P)
    return Q

import numpy as np
from scipy.optimize import linear_sum_assignment
from multiviewica import multiviewica
from shica import shica_j, shica_ml
from .utils import find_order, _estimate_causal_order


def ica_limvam(
    X,
    shared_causal_ordering=True,
    ica_algo="shica_ml",
    max_iter=3000,
    tol=1e-8,
    random_state=None,
    return_full=False,
):
    """
    Assume the model xi = Bi xi + Di s + ni, where Bi are DAG matrices that can
    share the same causal ordering, Di are diagonal scaling matrices, s are shared
    sources, and ni are Gaussian disturbances.
    ICA-LiMVAM estimates matrices Bi, and then identifies the causal ordering(s).

    Parameters
    ----------
    X : ndarray, shape (n_views, n_components, n_samples)
        Training data, where ``n_views`` is the number of views, 
        ``n_components`` is the number of components, and ``n_samples`` is
        the number of samples.

    shared_causal_ordering : bool (default=True)
        Whether we estimate a causal order common to all views, or
        one causal order per view.

    ica_algo : string, optional (default="shica_ml")
        The multiview ICA algorithm used in the first step.
        It can be either ``shica_ml``, ``shica_j``, or ``multiviewica``.
        Here are the default parameters of:
            ``shica_ml``: max_iter=3000; tol=1e-8
            ``shica_j``: max_iter=10000; tol=1e-5
            ``multiviewica``: max_iter=1000; tol=1e-3.

    max_iter : int, optional (default=3000)
        The maximum number of iterations of the multiview ICA algorithm.

    tol : float, optional (default=1e-8)
        The tolerance parameter of the multiview ICA algorithm.

    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.

    new_find_order_function : bool (default=False)
        Whether we use a new heuristic function to find P and T from B,
        or the one implemented in the original LiNGAM.

    return_full : bool (default=False)
        Whether we return all parameters or only the most important ones.

    Returns
    -------
    B : ndarray, shape (n_views, n_components, n_components)
        Causal effect matrices.

    T : ndarray, shape (n_views, n_components, n_components)
        Causal effects represented by matrices T[i] as close as possible to
        strictly lower triangular.

    P : ndarray, shape (n_components, n_components) or (n_views, n_components, n_components)
        Causal order(s) represented by a permutation matrix or multiple permutation 
        matrices, depending on ``shared_causal_ordering``.

    S_avg: ndarray, shape (n_components, n_samples)
        Source estimates. Only returned if return_full==True.

    W: ndarray, shape (n_views, n_components, n_components)
        Unmixing matrices found by the multiview ICA algorithm. 
        Only returned if return_full==True.
    
    D: ndarray, shape (n_views, n_components)
        Scaling matrices. Only returned if return_full==True.
    
    Sigma: ndarray, shape (n_views, n_components)
        Variances of the disturbances ni. Only returned if return_full==True.
    """
    m, p, n = X.shape
    
    # Step 1: use a multiview ICA algorithm
    if ica_algo == "shica_ml":
        W, Sigma, S_avg = shica_ml(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "shica_j":
        W, Sigma, S_avg = shica_j(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "multiviewica":
        _, W, S_avg = multiviewica(
            X, max_iter=max_iter, tol=tol, random_state=random_state)
        Sigma = np.zeros((m, p))
    else:
        raise ValueError(
            "ica_algo should be either 'shica_ml', 'shica_j', or 'multiviewica'")

    # Step 2: find permutation Q
    W_inv = 1 / np.sum([np.abs(Wi.T) for Wi in W], axis=0)  # shape (p, p)
    _, index = linear_sum_assignment(W_inv)
    QW = np.array([Wi[index] for Wi in W])

    # Step 3: scaling
    D = np.array([np.diag(Wi) for Wi in QW])                # shape (m, p)
    DQW = QW / D[:, :, np.newaxis]
    
    if ica_algo in ["shica_ml", "shica_j"]:
        Sigma = D**2 * Sigma

    # Step 4: causal effects
    B = np.array([np.eye(p)] * m) - DQW  # B is not lower triangular

    # Step 5: estimate causal order(s) with a simple method
    if shared_causal_ordering:
        B_avg = np.mean(np.abs(B), axis=0)
        order = _estimate_causal_order(B_avg)
        if order is None:
            # backup solution when B_avg cannot be permuted to strictly lower triangular
            print("The order was not found. Using backup function.")
            order = find_order(B_avg)
        P = np.eye(p)[order]
        T = P @ B @ P.T
    else:
        P = np.zeros((m, p, p))
        for i in range(m):
            order = _estimate_causal_order(np.abs(B[i]))
            if order is None:
                # backup solution when B cannot be permuted to strictly lower triangular
                print("The order was not found. Using backup function.")
                order = find_order(np.abs(B[i]))
            P[i] = np.eye(p)[order]
        T = np.array([Pi @ Bi @ Pi.T for Pi, Bi in zip(P, B)])

    if return_full:
        return B, T, P, S_avg, W, D, Sigma
    return B, T, P

"""
Calculate circular moments and related quantities

"""

__date__ = "February 2024 - September 2025"

import jax
from jax import jacfwd, jit
from jax.lax import scan
import jax.numpy as jnp
from tqdm import tqdm


@jit
def get_stats(x):
    """
    Compute pairwise trigonometric statistics for an array of angles.

    For each pair of indices (i, j), we construct an "angle matrix"
    A with the following entries:

        A[i, j] = x[i] + x[j],   if i < j   (upper triangle)
        A[i, i] = x[i],          if i = j   (diagonal)
        A[i, j] = x[j] - x[i],   if i > j   (lower triangle)

    Note in particular: the **strictly lower triangular part** is
    defined as x[j] - x[i] (not x[i] - x[j]).

    The return value stacks cos(A) and sin(A) along the last axis.

    Parameters
    ----------
    x : array, shape (d,)
        Input vector of angles.

    Returns
    -------
    stats : array, shape (d, d, 2)
        For each pair (i, j):
          stats[i, j, 0] = cos(A[i, j])
          stats[i, j, 1] = sin(A[i, j])
    """
    x1 = jnp.expand_dims(x, axis=0)  # [1, d]
    x2 = jnp.expand_dims(x, axis=1)  # [d, 1]
    diff_mat = x1 - x2  # [d, d] for differences (for lower triangle)
    sum_mat = x1 + x2   # [d, d] for sums (for upper triangle)
    
    # For i < j (upper triangle): use sum_mat.
    # For i > j (lower triangle): use diff_mat.
    # For the diagonal (i == j): we want x, and note sum_mat diagonal is 2*x.
    upper = jnp.triu(sum_mat, k=1)
    lower = jnp.tril(diff_mat, k=-1)
    diag = jnp.diag(x)  # directly place single-angle stats

    # Combine the parts into one matrix.
    angles = upper + lower + diag

    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=2)


stats_jac = jacfwd(get_stats)
"""Jacobian of get_stats"""

flat_stats = lambda x: get_stats(x).flatten()
"""Flat version of get_stats: [2 d^2]"""


def _angles_from_x(x):
    x1 = x[None, :]         # [1,d]
    x2 = x[:, None]         # [d,1]
    sum_mat  = x1 + x2
    diff_mat = x1 - x2
    upper = jnp.triu(sum_mat,  k=1)   # i<j: x_i + x_j
    lower = jnp.tril(diff_mat, k=-1)  # i>j: x_j - x_i
    diag  = jnp.diag(x)                # i=j: x_i
    return upper + lower + diag        # [d,d]


def get_average_stats_with_loader(loader, n_iter=1000):
    """
    Calculate the average stats using a loader.

    Parameters
    ----------

    Returns
    -------
    stats : [d,d,2]
    """
    batch_stats_func = jax.vmap(get_stats)
    get_batch_stats = jit(lambda x: jnp.mean(batch_stats_func(x), axis=0))
    stats = 0.0
    for i in tqdm(range(1, n_iter + 1), desc="Collecting stats"):
        xb = next(loader)
        xb = xb.reshape(len(xb), -1)
        batch_stats = get_batch_stats(xb) # [d,d,2]
        stats = stats + batch_stats / n_iter
    return stats


def get_average_Gamma_with_loader(loader, n_iter=1000):
    """
    Calculate the average stats using a loader.

    Parameters
    ----------

    Returns
    -------
    Gamma : [2d^2, 2d^2]
    """
    Gamma = 0.0
    for i in tqdm(range(1, n_iter + 1), desc="Collecting Gamma"):
        xb = next(loader)
        xb = xb.reshape(len(xb), -1)
        batch_Gamma = get_Gamma_hat(xb)
        Gamma = Gamma + batch_Gamma / n_iter
    return Gamma 


@jit
def get_S(x):
    """Get the S vector used for score matching."""
    d = len(x)
    coeff = (2 * jnp.ones((d, d)) - jnp.eye(d)).reshape(d, d, 1)
    return (coeff * get_stats(x)).flatten()


@jit
def get_cross_stats(x, y):
    """
    Get the cross statistics for two arrays of angles.

    Parameters
    ----------
    x : [c]
    y : [d]

    Returns
    -------
    stats: [c,d,4]
    """
    x1 = x.reshape(-1, 1)
    x2 = y.reshape(1, -1)
    diff_mat = x1 - x2
    sum_mat = x1 + x2
    return jnp.stack(
        [
            jnp.cos(sum_mat),
            jnp.sin(sum_mat),
            jnp.cos(diff_mat),
            jnp.sin(diff_mat),
        ],
        axis=2,
    )


cross_stats_jac = jacfwd(get_cross_stats, argnums=(0, 1))
"""Jacobian of get_cross_stats"""


def get_H_hat(X, weights=None):
    """
    Get the empirical H vector for fitting TG models.

    Parameters
    ----------
    X : (n,d)
    weights : (n,), optional

    Returns
    -------
    H_hat : [2d^2]
    """
    n, d = X.shape
    coeff = (2 * jnp.ones((d, d)) - jnp.eye(d)).reshape(d, d, 1)

    def _step(carry, inputs):
        H_sum = carry  # [2d^2]
        x, w = inputs
        val = w * coeff * get_stats(x)  # [d,d,2]
        H_sum = H_sum + val.flatten()  # [2d^2]
        return H_sum, None

    if weights is None:
        weights = jnp.ones(n) / n
    else:
        weights = weights / weights.sum()
    H_sum = jnp.zeros(2 * d**2)
    H_sum, _ = scan(_step, H_sum, (X, weights))
    return H_sum


def get_Gamma_hat(X, weights=None):
    """
    Get the empirical Gamma matrix for fitting TG models.

    Parameters
    ----------
    X : (n,d)
    weights : (n,), optional

    Returns
    -------
    Gamma_hat : [2d^2]
    """
    n, d = X.shape

    def _step(carry, inputs):
        x, w = inputs
        Gamma_sum = carry  # [2d^2, 2d^2]
        jac = stats_jac(x)  # [d,d,2,d]
        jac = jac.reshape(-1, d)  # [2d^2,d]
        Gamma_sum = Gamma_sum + w * jac @ jac.T
        return Gamma_sum, None

    if weights is None:
        weights = jnp.ones(n) / n
    else:
        weights = weights / weights.sum()
    Gamma_sum = jnp.zeros((2 * d**2, 2 * d**2))
    Gamma_sum, _ = scan(_step, Gamma_sum, (X, weights))
    return Gamma_sum


def get_cross_H_hat(X, Y):
    """
    Get the empirical H vector for fitting TG models.

    Parameters
    ----------
    X : [n,c]
    Y : [n,d]

    Returns
    -------
    H_hat : [2cd]
    """
    assert X.shape[0] == Y.shape[0]
    n, c = X.shape
    d = Y.shape[1]

    def _step(carry, inp):
        x, y = inp
        H_sum = carry  # [4cd]
        val = 2 * get_cross_stats(x, y)  # [c,d,4]
        H_sum = H_sum + val.flatten()  # [4cd]
        return H_sum, None

    H_sum = jnp.zeros(4 * c * d)
    H_sum, _ = scan(_step, H_sum, (X, Y))
    return H_sum / n


def get_Gamma_hat_matvec_func(X, l2_reg=0.0):
    n, d = X.shape

    def _step(carry, x):
        prod, vec = carry
        jac = stats_jac(x)  # [d,d,2,d]
        jac = jac.reshape(-1, d)  # [2d^2,d]
        prod = prod + jac @ jac.T @ vec / n
        return (prod, vec), None

    def matvec_func(vec):
        temp = jnp.zeros(2 * d**2)
        return scan(_step, (temp, vec), X)[0][0] + l2_reg * vec

    return jit(matvec_func)


def get_D_hat(X):
    """
    Get the empirical D matrix for fitting TG models.

    Parameters
    ----------
    X : [n,d]

    Returns
    -------
    D_hat : [2d^2, d]
    """
    n, d = X.shape

    def _step(carry, x):
        jac_sum = carry
        jac = stats_jac(x)  # [d,d,2,d]
        jac = jac.reshape(-1, d)  # [2d^2,d]
        jac_sum = jac_sum + jac
        return jac_sum, None

    D_hat = jnp.zeros((2 * d**2, d))
    D_hat, _ = scan(_step, D_hat, X)
    return D_hat / n


def get_cross_D_hat(X, Y):
    """
    Get the empirical D matrix for fitting TG models.

    Parameters
    ----------
    X : [n,c]
    Y : [n,d]

    Returns
    -------
    D_hat : [2cd, c+d]
    """
    assert X.shape[0] == Y.shape[0]
    n, c = X.shape
    d = Y.shape[1]

    def _step(carry, inp):
        x, y = inp
        jac_sum = carry
        jac = cross_stats_jac(x, y)
        jac = jnp.concatenate(jac, -1).reshape(-1, c + d)  # [4cd, c+d]
        jac_sum = jac_sum + jac
        return jac_sum, None

    D_hat = jnp.zeros((4 * c * d, c + d))
    D_hat, _ = scan(_step, D_hat, (X, Y))
    return D_hat / n


def solve_tg_exact(X, weights=None, reg=None):
    """
    Solve the exact system by explictly forming the Gamma matrix and solving.

    Parameters
    ----------
    X : (n,d)
    weights : (n,), optional
    reg : float, optional
        Add reg * I to the estimated Gamma matrix

    Returns
    -------
    phi : (d,d,2)
    
    """
    assert X.ndim == 2
    d = X.shape[1]
    Gamma_hat = get_Gamma_hat(X, weights=weights)
    if reg is not None:
        Gamma_hat = Gamma_hat + reg * jnp.eye(Gamma_hat.shape[0])
    H_hat = get_H_hat(X, weights=weights)
    return jnp.linalg.solve(Gamma_hat, H_hat).reshape(d,d,2)

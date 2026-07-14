"""
Following the torus graph reference implementation closely with JAX.

Reference code: https://github.com/natalieklein/torus-graphs
"""

__date__ = "May 2023"

import jax.numpy as jnp


def pairwise(x):
    (d,) = x.shape
    num_pairs = (d * (d - 1)) // 2
    Xdif = jnp.zeros(num_pairs)
    Xsum = jnp.zeros(num_pairs)

    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            Xdif = Xdif.at[idx].set((x[i] - x[j]) % (2 * jnp.pi))
            Xsum = Xsum.at[idx].set((x[i] + x[j]) % (2 * jnp.pi))
            idx += 1
    return Xdif, Xsum


def sufficient_statistics(x):
    Xdif, Xsum = pairwise(x)
    Sc = jnp.cos(x)
    Ss = jnp.sin(x)
    Salpha = jnp.cos(Xdif)
    Sbeta = jnp.sin(Xdif)
    Sgamma = jnp.cos(Xsum)
    Sdelta = jnp.sin(Xsum)
    return Sc, Ss, Salpha, Sbeta, Sgamma, Sdelta


def estimate_params(X, reg=0):
    """
    Parameters
    ----------
    X : [d,n]
    reg : float, optional

    Returns
    -------
    phi_hat : [2 d^2]
    """
    d, n = X.shape
    num_pairs = (d * (d - 1)) // 2
    num_param = 2 * d**2
    Gamma_sum = jnp.zeros((num_param, num_param))
    H_sum = jnp.zeros(num_param)

    arr = jnp.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=int)

    for m in range(n):
        Sc, Ss, Salpha, Sbeta, Sgamma, Sdelta = sufficient_statistics(
            X[:, m]
        )  # [d], [d], [num_pairs], [num_pairs], [num_pairs], [num_pairs]

        Dx_Sc = -jnp.diag(Ss)
        Dx_Ss = jnp.diag(Sc)
        Dx_Salpha = jnp.zeros((num_pairs, d))
        Dx_Sbeta = jnp.zeros((num_pairs, d))
        Dx_Sgamma = jnp.zeros((num_pairs, d))
        Dx_Sdelta = jnp.zeros((num_pairs, d))

        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                col_idx = jnp.array([i, j], dtype=int)
                Dx_Salpha = Dx_Salpha.at[idx, col_idx].set(arr[0] * Sbeta[idx])
                Dx_Sbeta = Dx_Sbeta.at[idx, col_idx].set(arr[1] * Salpha[idx])
                Dx_Sgamma = Dx_Sgamma.at[idx, col_idx].set(arr[2] * Sdelta[idx])
                Dx_Sdelta = Dx_Sdelta.at[idx, col_idx].set(arr[3] * Sgamma[idx])
                idx += 1
        Dx_temp = jnp.concatenate(
            [Dx_Sc, Dx_Ss, Dx_Salpha, Dx_Sbeta, Dx_Sgamma, Dx_Sdelta], axis=0
        )
        H = jnp.concatenate([Sc, Ss, 2 * Salpha, 2 * Sbeta, 2 * Sgamma, 2 * Sdelta])
        H_sum = H_sum + H
        Gamma_sum = Gamma_sum + Dx_temp @ Dx_temp.T

    H_hat = H_sum / n
    Gamma_hat = Gamma_sum / n + reg * jnp.eye(Gamma_sum.shape[0])
    phi_hat = jnp.linalg.solve(Gamma_hat, H_hat)
    return phi_hat


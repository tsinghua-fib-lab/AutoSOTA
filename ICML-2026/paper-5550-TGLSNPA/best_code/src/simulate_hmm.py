"""
Sample from a random HMM

"""
__date__ = "April 2025"

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple

from src.sample import sample_torus_graph

batched_sample = jax.vmap(sample_torus_graph, in_axes=(None, None, 0), out_axes=0)


class HMMPriorParams(NamedTuple):
    alpha_self: float = 10.0
    alpha_other: float = 1.0
    phi_prec_tril: float = 100.0
    phi_prec_diag: float = 1.0
    phi_prec_triu: float = 100.0


class HMMParams(NamedTuple):
    phis : jnp.ndarray
    pi : jnp.ndarray
    trans_mat : jnp.ndarray


def sample_sticky_transition_matrix(key, K, alpha_self=10.0, alpha_other=1.0):
    """
    Sample a (K, K) transition matrix in JAX with a self-transition bias.

    Args:
        key: jax.random.PRNGKey
        K (int): Number of HMM states.
        alpha_self (float): Dirichlet concentration for self-transition.
        alpha_other (float): Dirichlet concentration for other transitions.

    Returns:
        A (jnp.ndarray): Transition matrix of shape (K, K), rows sum to 1.
    """
    def sample_row(key, k):
        alpha = jnp.full((K,), alpha_other).at[k].set(alpha_self)
        return jax.random.dirichlet(key, alpha)

    keys = jax.random.split(key, K)
    A = jax.vmap(sample_row)(keys, jnp.arange(K))
    return A


def sample_phis(key, K, d, phi_prec_tril=2.0, phi_prec_diag=1.0, phi_prec_triu=2.0):
    ones = jnp.ones((d,d))
    factor = ( \
        phi_prec_tril**-0.5 * jnp.tril(ones, k=-1) +
        phi_prec_diag**-0.5 * jnp.diag(ones) +
        phi_prec_triu**-0.5 * jnp.triu(ones, k=1)
    )
    return factor.reshape(1,d,d,1) * jr.normal(key, (K,d,d,2))


def sample_hmm(key, T, log_trans_mat, phis, burn_in=100):
    """
    key           : jax.random.PRNGKey
    T             : int, desired sequence length
    log_trans_mat : [K,K] array of log P(z_t = j | z_{t-1} = i)
    phis          : state emission params, shape (K, d, d, 2) for batched_sample
    burn_in       : int, prefix to discard from the continuous sampler
    Returns:
      observations : [T, d]  continuous observations
      z            : [T]     discrete state trajectory
    """
    assert log_trans_mat.ndim == 2
    assert phis.ndim == 4
    assert len(log_trans_mat) == len(phis)
    K = log_trans_mat.shape[0]

    # 1. Sample all state‐conditional emissions up front:
    subkey, key = jr.split(key)
    X_full = batched_sample(subkey, 2*T + burn_in, phis)  # => [K, 2T+burn_in, d]
    X = X_full[:, burn_in:, :]  [:, ::2, :]               # => [K, T, d]
    subkey, key = jr.split(key)
    X = X[:, jr.permutation(subkey, X.shape[1]), :]  # [K, T, d]

    # 2. Pick a uniform random starting state z0
    subkey, key = jr.split(key)
    z0 = jr.randint(subkey, shape=(), minval=0, maxval=K)

    # 3. Roll out the Markov chain for T−1 steps
    def step(prev_state, rng):
        # sample z_t ~ Categorical(log_probs=log_trans_mat[prev_state])
        z_t = jr.categorical(rng, log_trans_mat[prev_state])
        return z_t, z_t

    # generate T−1 fresh keys
    subkeys = jr.split(key, T - 1)
    _, z_tail = jax.lax.scan(step, z0, subkeys)  # z_tail: [T-1]
    z = jnp.concatenate([jnp.expand_dims(z0, 0), z_tail], axis=0)  # [T]

    # 4. Index into X to pick the actual observations
    t_idx = jnp.arange(T)
    observations = X[z, t_idx, :]  # => [T, d]

    return observations, z



if __name__ == '__main__':
    pass


###
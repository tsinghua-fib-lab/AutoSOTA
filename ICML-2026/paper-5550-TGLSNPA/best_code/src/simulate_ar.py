"""
Sample from an AR-TG

"""
__date__ = "May - July 2025"


import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.stats import vonmises
from typing import NamedTuple

from src.von_mises import vm_entropy, vm_log_pdf


class ARTGPriorParams(NamedTuple):
    d : int = 2
    variance_initial_self : float =  1.0
    variance_initial_other : float = 0.5
    variance_slope : float = 0.2
    allow_12 : bool = True # Allow T_{1 \rightarrow 2} > 0 (only used if d=2)
    allow_21 : bool = True # Allow T_{2 \rightarrow 1} > 0 (only used if d=2)
    

class ARTGParams(NamedTuple):
    W : jnp.ndarray # Shape: (d_out,d_in,L,2)


def univariate_to_bivariate(W_in):
    """
    W_in: (2_out, L, 2)
    
    W_out: (2_out, 2_in, L, 2)
    """
    z = 0 * W_in[0] # [L,2]
    W_out_1 = jnp.stack([W_in[0], z], 0) # (2_in, L, 2)
    W_out_2 = jnp.stack([z, W_in[1]], 0) # (2_in, L, 2)
    W_out = jnp.stack([W_out_1, W_out_2], 0) # (2_out, 2_in, L, 2)
    return W_out


def sample_artg_params(key, prior, L, W_std=None):
    """
    Sample AR-TG parameters

    Parameters
    ----------
    key : jax.random.PRNGKey
    prior : ARTGPriorParams
    L : int

    Returns
    -------
    params : ARTGParams
    """
    d = getattr(prior, "d", 2)
    if W_std is None:
        W_std = get_W_std(prior, L)
    W = W_std * jr.normal(key, (d, d, L, 2))
    return ARTGParams(W=W)


def get_W_std(prior, L):
    d = getattr(prior, "d", 2)
    vs, vo, m = prior.variance_initial_self, prior.variance_initial_other, prior.variance_slope

    var_self = vs * jnp.exp(-m * jnp.arange(L)[::-1])      # (L,)
    var_other = vo * jnp.exp(-m * jnp.arange(L)[::-1])     # (L,)

    var = jnp.full((d, d, L), var_other)                   # (d,d,L)
    var = var.at[jnp.arange(d), jnp.arange(d)].set(var_self)

    if d == 2:
        if not prior.allow_21:
            var = var.at[0, 1].set(0.0)
        if not prior.allow_12:
            var = var.at[1, 0].set(0.0)

    var = jnp.stack([var, var], -1)  # (d,d,L,2)
    return jnp.sqrt(var)


def get_W_log_prob(W, prior, W_std=None):
    L = W.shape[2]
    if W_std is None:
        W_std = get_W_std(prior, L)
    W_z = W / W_std
    return -0.5 * jnp.sum((jnp.log(2 * jnp.pi) + W_z**2))


def sample_artg(key, T, params):
    """
    Sample from an AR-TG model.

    Returns a JAX array of shape (d, T).
    """
    W = params.W  # (d, d, L, 2)
    L = W.shape[2]
    d = W.shape[0]

    # initialize history with uniform angles in [0, 2π)
    rng, subkey = jr.split(key)
    history = jr.uniform(subkey, shape=(d, L), minval=0.0, maxval=2*jnp.pi)

    samples = []
    for _ in range(T):
        # advance JAX RNG
        rng, step_key = jr.split(rng)

        # build (cos, sin) features from the last L angles
        features = jnp.stack([jnp.cos(history), jnp.sin(history)], axis=-1)  # (d, L, 2)

        # linear map → a, b parameters for each of the 2 dimensions
        pred = jnp.sum(W * features, axis=(1,2))  # (d, 2)
        a, b = np.array(pred[:, 0]), np.array(pred[:, 1])

        # convert to von Mises parameters
        loc   = np.angle(a + 1j * b)              # mean direction
        kappa = np.sqrt(a**2 + b**2)              # concentration

        # seed Numpy RNG from JAX
        seed = int(jr.randint(step_key, (), 0, 2**30))
        rng_np = np.random.default_rng(seed)

        # draw a pair of independent von Mises angles
        sample_np = vonmises.rvs(kappa, loc=loc, random_state=rng_np)
        sample    = jnp.asarray(sample_np)       # back to JAX

        # append and update history
        samples.append(sample)
        history = jnp.concatenate([history[:, 1:], sample[:, None]], axis=1)

    return jnp.stack(samples, axis=1)  # shape (d, T)


def get_artg_loglike(samples, params):
    """
    Calculate the log likelihood of the samples under the model.

    Note that this is only the log likelihood of the T-L last samples!

    Parameters
    ----------
    samples : jnp.ndarray
        Shape: (d,T)
    params : ARTGParams

    Returns
    -------
    log_like : jnp.ndarray
        Shape: (d,)
    """
    W = params.W  # (d, d, L, 2)
    L = W.shape[2]
    T = samples.shape[1]
    d = W.shape[0]

    def _step(carry, x):
        history, ll = carry

        # build (cos, sin) features from the last L angles
        features = jnp.stack([jnp.cos(history), jnp.sin(history)], axis=-1)  # (d, L, 2)

        # linear map → a, b parameters for each of the 2 dimensions
        pred = jnp.sum(W * features, axis=(1,2))  # (d, 2)

        # Calculate log likelihood.
        ll = ll + vm_log_pdf(pred[:, 0], pred[:, 1], x)

        # Update history and return.
        history = jnp.concatenate([history[:, 1:], x[:, None]], axis=1)
        return (history, ll), None

    history = samples[:,:L]
    ll = jnp.zeros(d)

    (_, ll), _ = jax.lax.scan(_step, (history, ll), samples[:,L:].T)
    return ll


def estimate_artg_entropy_rates(samples, params):
    """
    Estimate the entropy rate of the model given the samples.

    Parameters
    ----------
    samples : jnp.ndarray
        Shape: (d,T)
    params : ARTGParams

    Returns
    -------
    entopy_rates : jnp.ndarray
        Shape: (d,)
    """
    W = params.W # (d, d, L, 2)
    L = W.shape[2]
    T = samples.shape[1]
    d = W.shape[0]

    def _step(carry, x):
        history, er = carry

        # build (cos, sin) features from the last L angles
        features = jnp.stack([jnp.cos(history), jnp.sin(history)], axis=-1)  # (d, L, 2)

        # linear map → a, b parameters for each of the 2 dimensions
        pred = jnp.sum(W * features[None], axis=(1,2))  # (d, 2)

        # Calculate log likelihood.
        er = er + vm_entropy(jnp.linalg.norm(pred, axis=1))

        # Update history and return.
        history = jnp.concatenate([history[:, 1:], x[:, None]], axis=1)
        return (history, er), None

    history = samples[:,:L]
    er = jnp.zeros(d)

    (_, er), _ = jax.lax.scan(_step, (history, er), samples[:,L:].T)
    return er / (T - L)
        
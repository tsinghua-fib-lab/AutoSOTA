"""
Estimate AR-TG-based multivariate transfer entropies.

"""
__date__ = "June - September 2025"

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
import numpy as np
from scipy.stats import vonmises
from tqdm import tqdm

from .imputation_model import GaussianChannelConditioner
from .von_mises import vm_log_pdf

KAPPA_MIN = 1e-6
KAPPA_MAX = 100.0


def estimate_mv_te(
    key,
    loader,
    W,             # (F, RL2, R2)
    means,         # (F, RL2)
    covars,        # (F, RL2, RL2)
    R: int,
    L: int,
    F: int,
    max_num_batches: int = None,
    show_progress: bool = True,
    K: int = 16,                           # MC samples for x_s | x_{-s}
    return_log_terms: bool = False,        # also return E[log p(y|x)], E[log p(y|x_{-s})]
):
    """
    Estimate multivariate transfer entropy with the log-probability estimator:

        TE_{s->r} = E[ log p(y_r | x) - log p(y_r | x_{-s}) ],

    where p(y_r | x_{-s}) is approximated by a K-component mixture over draws of
    x_s ~ p(x_s | x_{-s}). All frequencies are processed together per loader batch.

    Returns
    -------
    te_arr : (F, R, R) array
        TE per (frequency f, source s, target r).

    If `return_log_terms` is True, also returns:
    logp_obs_arr : (F, R, R)
        Batch-mean E[log p(y_r|x)] broadcast along source axis s.
    logp_imp_arr : (F, R, R)
        Batch-mean E[log p(y_r|x_{-s})].
    """
    RL2 = R * L * 2
    R2  = R * 2

    # ---- helpers -------------------------------------------------------------
    def flatten_like_training(w_r_2l: jnp.ndarray) -> jnp.ndarray:
        # w_r_2l: [B, R, 2L] packs [L, 2] with 2 as the fast axis
        B = w_r_2l.shape[0]
        w = w_r_2l.reshape(B, R, L, 2)         # [B, R, L, 2]
        w = jnp.transpose(w, (0, 2, 1, 3))     # [B, L, R, 2]
        return w.reshape(B, RL2)               # [B, RL2]

    def predict_ab(W_i: jnp.ndarray, windows_r_2l: jnp.ndarray) -> jnp.ndarray:
        # W_i: [RL2, R*2]; windows_r_2l: [B, R, 2L]
        B = windows_r_2l.shape[0]
        X = flatten_like_training(windows_r_2l)      # [B, RL2]
        pred = (X @ W_i).reshape(B, R, 2)            # [B, R, 2]
        return pred[..., 0], pred[..., 1]            # (a, b) each [B, R]

    @jax.jit
    def kernel_log_prob(W_i, windows_r_2l, targets, sampler_keys, cond, mu):
        """
        Returns:
          te    : [R, R]
          lpobs : [R]      (batch mean)
          lpimp : [R, R]   (batch mean)
        """
        a_full, b_full = predict_ab(W_i, windows_r_2l)          # [B, R]
        logp_full = vm_log_pdf(a_full, b_full, targets)         # [B, R]
        lpobs = jnp.mean(logp_full, axis=0)                     # [R]

        def per_source(s):
            def one_draw(k):
                imp = cond.conditional_sample(k, windows_r_2l, mu)  # [B, R, 2L]
                Xs = windows_r_2l.at[:, s, :].set(imp[:, s, :])
                a_k, b_k = predict_ab(W_i, Xs)                      # [B, R]
                return vm_log_pdf(a_k, b_k, targets)                # [B, R]

            logps = jax.vmap(one_draw)(sampler_keys)                # [K, B, R]
            logp_mix = logsumexp(logps, axis=0) - jnp.log(logps.shape[0])  # [B, R]
            return jnp.mean(logp_mix, axis=0)                       # [R]

        lpimp = jax.vmap(per_source)(jnp.arange(R))                 # [R, R]
        te = lpobs[None, :] - lpimp                                  # [R, R]
        return te, lpobs, lpimp

    # Pre-build conditioners per frequency
    conds = [GaussianChannelConditioner.from_cov(covars[f], R, 2 * L) for f in range(F)]
    mu_all = means.reshape(F, R, 2 * L)                              # [F, R, 2L]

    W = jnp.asarray(W)

    te_sum = jnp.zeros((F, R, R))
    lpobs_sum = jnp.zeros((F, R, R))  # broadcast across source dim
    lpimp_sum = jnp.zeros((F, R, R))
    n_batches = 0

    it = loader
    if show_progress:
        it = tqdm(loader, desc="mvTE (all F)", leave=False)

    for _, batch in enumerate(it):
        # batch: (B, L+1, R, F)
        B = batch.shape[0]
        batch_lr = batch[:, :-1, :, :]            # (B, L, R, F)
        targets = batch[:, -1, :, :]              # (B, R, F)

        # Build windows for all F at once: [F, B, R, 2L]
        cs = jnp.stack([jnp.cos(batch_lr), jnp.sin(batch_lr)], axis=-1)   # (B, L, R, F, 2)
        windows_all = jnp.transpose(cs, (3, 0, 2, 1, 4)).reshape(F, B, R, 2 * L)  # (F,B,R,2L)
        targets_all = jnp.transpose(targets, (2, 0, 1))                   # (F, B, R)

        # Draw keys for each (f, k)
        key, sub = jr.split(key)
        fk_keys = jr.split(sub, F * K).reshape(F, K, 2)                   # (F, K, 2)

        for f in range(F):
            te_f, lpobs_f, lpimp_f = kernel_log_prob(
                W[f],
                windows_all[f],
                targets_all[f],
                fk_keys[f],
                conds[f],
                mu_all[f],
            )
            # Broadcast lpobs_f over source axis for consistent [R,R] storage
            lpobs_rr = jnp.broadcast_to(lpobs_f[None, :], (R, R))

            te_sum = te_sum.at[f].add(te_f)
            lpobs_sum = lpobs_sum.at[f].add(lpobs_rr)
            lpimp_sum = lpimp_sum.at[f].add(lpimp_f)

        n_batches += 1
        if (max_num_batches is not None) and (n_batches >= max_num_batches):
            break
    denom = jnp.maximum(n_batches, 1)
    te_arr = te_sum / denom
    if return_log_terms:
        logp_obs_arr = lpobs_sum / denom
        logp_imp_arr = lpimp_sum / denom
        return te_arr, logp_obs_arr, logp_imp_arr
    else:
        return te_arr


def draw_random_process(key, pattern, L, self_scale=1.0, cross_scale=1.0):
    """
    Draw random AR-TG weights consistent with a TE pattern.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    pattern : array_like, shape (R, R)
        Binary matrix where ``pattern[s, r]`` is 1 if transfer entropy from
        source ``s`` to target ``r`` should be non-zero.
    L : int
        Number of lags.
    self_scale : float, optional
        Standard deviation for self-couplings.
    cross_scale : float, optional
        Standard deviation for allowed cross-couplings.

    Returns
    -------
    jnp.ndarray, shape (R, R, L, 2)
        Weight tensor for an AR-TG process with angles encoded via cosine/sine
        features.  Indexing is ``W[target, source, lag, feature]``.
    """
    pattern = jnp.asarray(pattern)
    R = pattern.shape[0]
    base = jnp.eye(R) * self_scale
    cross = cross_scale * pattern.T
    scales = base + cross
    scales = scales[..., None, None]  # (R,R,1,1)
    W = scales * jr.normal(key, (R, R, L, 2))
    return W


def sample_process(key, W, T):
    """
    Sample angles from an AR-TG defined by ``W``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    W : array_like, shape (R, R, L, 2)
        Weight tensor as returned by :func:`draw_random_process`.
    T : int
        Number of time steps to simulate.

    Returns
    -------
    jnp.ndarray, shape (T, R)
        Simulated angular time series for each channel.
    """
    W = jnp.asarray(W) # (R, R, L, 2)
    R, _, L, _ = W.shape

    key, sub = jr.split(key)
    history = jr.uniform(sub, shape=(R, L), minval=0.0, maxval=2 * jnp.pi) # (R, L)

    samples = []
    rng = key
    for _ in range(T):
        rng, step_key = jr.split(rng)

        features = jnp.stack([jnp.cos(history), jnp.sin(history)], axis=-1) # (R, L, 2)
        pred = jnp.einsum('rslc,slc->rc', W, features)
        loc = jnp.angle(pred[:, 0] + 1j * pred[:, 1]) # (R,)
        kappa = jnp.linalg.norm(pred, axis=-1) # (R,)

        seed = int(jr.randint(step_key, (), 0, 2**30))
        rng_np = np.random.default_rng(seed)
        sample_np = vonmises.rvs(kappa, loc=loc, random_state=rng_np)
        sample = jnp.asarray(sample_np)

        samples.append(sample)
        history = jnp.concatenate([history[:, 1:], sample[:, None]], axis=1) # (R, L)

    return jnp.stack(samples, axis=0)

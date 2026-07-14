"""
Fit AR-TG models to data

"""
__date__ = "May - July 2025"


import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import optax
from tqdm import tqdm
from typing import Union

from src.simulate_ar import (
    ARTGPriorParams,
    ARTGParams,
    univariate_to_bivariate,
    get_W_std,
    sample_artg_params,
    estimate_artg_entropy_rates,
    get_artg_loglike,
)
from src.von_mises import vm_log_pdf
from src.conditional_ssm import fit_conditional_ssm



def fit_univariate_artg_mle_bfgs(key, X, L, prior):
    """
    Fit the univariate AR-TG via MAP estimation with BFGS.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : jnp.ndarray
        Shape: (T,)
    L : int
        Lag parameter
    prior : None or ARTGPriorParams, optional

    Returns
    -------
    W : jnp.ndarray
        Shape: (L,2)
    res : jax.scipy.optimize.OptimizeResults
    """
    T = len(X)
    assert T > L, f"{T} <= {L}"

    if prior is None:
        W = 1e-3 * jax.random.normal(key, shape=(L, 2)) # (L, 2)
        w_logp_fn = lambda x: 0.0
    else:
        W_std = get_W_std(prior, L)[0,0] # (L, 2)
        W = sample_artg_params(key, prior, L, W_std).W[0,0] # (L, 2)
        w_logp_fn = lambda x: -0.5 * jnp.sum((jnp.log(2 * jnp.pi) + (x / W_std)**2))

    features = jnp.stack([X[i:i+L] for i in jnp.arange(T-L)], 0) # (t,L)
    features = jnp.stack([jnp.cos(features), jnp.sin(features)], -1) # (t,L,2)
    targets = X[L:] # (t,)

    @jax.jit
    def loss_fn(W):
        """Negative log likelihood"""
        W = W.reshape(L,2)
        pred = jnp.sum(W[None] * features, axis=1) # (t,2)
        loss = -jnp.sum(vm_log_pdf(pred[:,0], pred[:,1], targets))
        loss = loss - w_logp_fn(W)
        return loss / (T - L)

    res = minimize(loss_fn, W.flatten(), method="BFGS")

    return res.x.reshape(L,2), res



def fit_bivariate_artg_bfgs(key, X, L, prior):
    """
    Fit the biivariate AR-TG via MAP estimation with BFGS.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : jnp.ndarray
        Shape: (2,T)
    L : int
        Lag parameter
    prior : None or ARTGPriorParams, optional

    Returns
    -------
    W : jnp.ndarray
        Shape: (2_out, 2_in, L, 2)
    res : jax.scipy.optimize.OptimizeResults
    """
    _, T = X.shape
    assert T > L, f"T={T} must exceed lag L={L}"
    t = T - L

    if prior is None:
        W = 1e-3 * jax.random.normal(key, shape=(2, 2, L, 2)) # (2_out, 2_in, L, 2)
        w_logp_fn = lambda x: 0.0
    else:
        W_std = get_W_std(prior, L) # (2_out, 2_in, L, 2)
        W = sample_artg_params(key, prior, L, W_std).W # (2_out, 2_in, L, 2)
        w_logp_fn = lambda x: -0.5 * jnp.sum((jnp.log(2 * jnp.pi) + (x / W_std)**2))

    # precompute all lagged windows: shape (t, 2_in=2, L)
    # windows[i, c, l] = X[c, i + l]
    windows = jnp.stack([ X[:, i : i + t] for i in range(L) ], axis=2)
    # now windows.shape == (2, t, L) → transpose to (t, 2, L)
    windows = windows.transpose(1, 0, 2)
    # build (cos, sin) features: (t, 2, L, 2)
    features = jnp.stack([jnp.cos(windows), jnp.sin(windows)], axis=-1)
    targets = X[:, L:].T # (t, 2)

    @jax.jit
    def loss_fn(W):
        """Negative log likelihood"""
        W = W.reshape(2, 2, L, 2)
        # features: (t, 2_out, L, 2)
        # W:        (2_out, 2_in, L, 2)
        pred = jnp.einsum('tila, oila -> toa', features, W)
        a = pred[..., 0]   # shape (t, 2_out)
        b = pred[..., 1]   # shape (t, 2_out)
        loss = -jnp.sum(vm_log_pdf(a, b, targets))
        loss = loss - w_logp_fn(W)
        return loss / (T - L)

    res = minimize(loss_fn, W.flatten(), method="BFGS")

    return res.x.reshape(2, 2, L, 2), res


def fit_univariate_artg_sgd(
        key : jax.random.PRNGKey,
        X: jnp.ndarray,
        L: int,
        lr: float = 1e-3,
        n_iter: int = 1000,
        prior : Union[None, ARTGPriorParams] = None,
    ):
    """
    Fit the univariate AR-TG via MAP estimation with ssm.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : jnp.ndarray, shape (T,)
        Angle time series in [0, 2π).
    L : int
        Lag parameter.
    lr : float
        Step-size for SSM.
    n_iter : int
        Number of SSM steps.
    prior : None or ARTGPriorParams, optional

    Returns
    -------
    W_final : jnp.ndarray, shape (L,2)
        Learned weights mapping cos/sin history to von Mises parameters.
    losses : jnp.ndarray, shape (n_iter,)
        Training loss at each step.
    """
    T = X.shape[0]
    assert T > L, f"Sequence length T={T} must exceed lag L={L}."

    # Precompute feature matrix and targets
    t = T - L
    xs = jnp.stack([X[i:i+L] for i in jnp.arange(t)], axis=0)   # (t, L)
    features = jnp.stack([jnp.cos(xs), jnp.sin(xs)], axis=-1)   # (t, L, 2)
    targets = X[L:]                                             # (t,)

    # Initialize parameters and optimizer
    if prior is None:
        W = 1e-3 * jax.random.normal(key, shape=(L, 2))
        w_logp_fn = lambda x: 0.0
    else:
        W_std = get_W_std(prior, L)[0,0] # (L, 2)
        W = sample_artg_params(key, prior, L, W_std).W[0,0] # (L, 2)
        w_logp_fn = lambda x: -0.5 * jnp.sum((jnp.log(2 * jnp.pi) + (x / W_std)**2))

    opt = optax.adam(lr)
    opt_state = opt.init(W)

    # Loss + gradient function
    def loss_fn(W):
        # Predict a,b parameters for each time step
        pred = jnp.sum(W[None] * features, axis=1)  # (t, 2)
        a, b = pred[:, 0], pred[:, 1]
        # Negative log-likelihood via vm_log_pdf
        loss = -jnp.mean(vm_log_pdf(a, b, targets))
        loss = loss - w_logp_fn(W)
        return loss / (T - L)

    @jax.jit
    def update_step(W, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(W)
        updates, opt_state = opt.update(grads, opt_state)
        W = optax.apply_updates(W, updates)
        return W, opt_state, loss

    # Run SSM.
    losses = []
    pbar = tqdm(range(n_iter))
    for _ in pbar:
        W, opt_state, loss = update_step(W, opt_state)
        losses.append(loss)
        pbar.set_description(f"NLL: {loss:.5f}")

    return W, jnp.array(losses)


def fit_bivariate_artg_ssm(
        key : jax.random.PRNGKey,
        X : jnp.ndarray,
        L : int,
        lr: float = 1e-2,
        n_iter: int = 5000,
        prior : Union[None, ARTGPriorParams] = None,
    ):
    """
    Fit the bivariate AR-TG via maximum likelihood using stochastic score matching.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : jnp.ndarray
        Shape: (2, T)  — two time series along axis 0
    L : int
        Lag parameter
    lr : float
        Step-size for SSM.
    n_iter : int
        Number of SSM steps.
    prior : None or ARTGPriorParams, optional

    Returns
    -------
    W : jnp.ndarray
        Shape: (2_out, 2_in, L, 2), the learned weight tensor
    """
    # dims
    _, T = X.shape
    assert T > L, f"T={T} must exceed lag L={L}"
    t = T - L

    # initialize W small
    key, subkey = jax.random.split(key)
    if prior is None:
        W = 1e-2 * jax.random.normal(subkey, (2, 2, L, 2))
        w_logp_fn = lambda x: 0.0
    else:
        W_std = get_W_std(prior, L) # (2, 2, L, 2)
        W = sample_artg_params(subkey, prior, L, W_std).W # (2, 2, L, 2)
        w_logp_fn = lambda x: -0.5 * jnp.sum((jnp.log(2 * jnp.pi) + (x / W_std)**2))

    # precompute all lagged windows: shape (t, 2_in=2, L)
    # windows[i, c, l] = X[c, i + l]
    windows = jnp.stack([ X[:, i : i + t] for i in range(L) ], axis=2)
    # now windows.shape == (2, t, L) → transpose to (t, 2, L)
    windows = windows.transpose(1, 0, 2)

    # build (cos, sin) features: (t, 2, L, 2)
    features = jnp.stack([jnp.cos(windows), jnp.sin(windows)], axis=-1)

    # targets: shape (t, 2_out=2)
    targets = X[:, L:].T

    # negative log-likelihood
    def loss_fn(W):
        # linear map → (t, 2_out, 2) giving (a, b) for each output channel
        #   pred[i_out, i_in, l, p] · features[t, i_in, l, p]

        # features: (t, 2_out, L, 2)
        # W:        (2_out, 2_in, L, 2)
        pred = jnp.einsum('tila, oila -> toa', features, W)
        a = pred[..., 0]   # shape (t, 2_out)
        b = pred[..., 1]   # shape (t, 2_out)

        # independent von Mises on each channel:
        # vm_log_pdf(a, b, x) returns log p(x|a,b), shape must broadcast to (t,2)
        logp = vm_log_pdf(a, b, targets)

        # average negative log-likelihood
        loss = -jnp.sum(logp) - w_logp_fn(W)
        
        return loss / (T - L)

    # set up Optax Adam
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(W)

    @jax.jit
    def step(W, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(W)
        updates, opt_state = optimizer.update(grads, opt_state)
        W = optax.apply_updates(W, updates)
        return W, opt_state, loss

    # run the loop
    pbar = tqdm(range(n_iter))
    for _ in pbar:
        W, opt_state, loss = step(W, opt_state)
        pbar.set_description(f"NLL: {loss:.5f}")

    return W


def fit_univariate_artg_score_matching(key, X, L, n_iter: int = 500, batch_size: int = 64):
    """Fit the univariate AR-TG via score matching."""
    T = X.shape[0]
    assert T > L, f"Sequence length T={T} must exceed lag L={L}."

    t = T - L
    X_hist = jnp.stack([X[i : i + L] for i in range(t)], 0)  # (t, L)
    Y = X[L:]

    params, _ = fit_conditional_ssm(key, X_hist, Y[:, None], n_iter=n_iter, batch_size=batch_size)
    W_flat, _ = params  # [2L, 2]

    W = jnp.zeros((L, 2))
    W = W.at[:, 0].set(W_flat[:L, 0])
    W = W.at[:, 1].set(W_flat[L:2 * L, 1])
    return W


def fit_bivariate_artg_score_matching(key, X, L, n_iter: int = 500, batch_size: int = 64):
    """Fit the bivariate AR-TG via score matching."""
    _, T = X.shape
    assert T > L, f"T={T} must exceed lag L={L}"
    t = T - L

    hist1 = jnp.stack([X[0, i : i + L] for i in range(t)], 0)
    hist2 = jnp.stack([X[1, i : i + L] for i in range(t)], 0)
    X_hist = jnp.concatenate([hist1, hist2], axis=1)  # (t, 2L)
    Y = X[:, L:].T  # (t, 2)

    params, _ = fit_conditional_ssm(key, X_hist, Y, n_iter=n_iter, batch_size=batch_size)
    W_flat, _ = params  # (4L, 4)

    W = jnp.zeros((2, 2, L, 2))
    m = 2 * L
    for j in range(2):
        col_a = j
        col_b = 2 + j
        # input 0 weights
        W = W.at[j, 0, :, 0].set(W_flat[:L, col_a])
        W = W.at[j, 0, :, 1].set(W_flat[m : m + L, col_b])
        # input 1 weights
        W = W.at[j, 1, :, 0].set(W_flat[L : 2 * L, col_a])
        W = W.at[j, 1, :, 1].set(W_flat[m + L : m + 2 * L, col_b])
    return W


def estimate_entropy_rates(samples, params, method: str = "log_prob"):
    """Estimate entropy rates with the specified method."""
    if method == "log_prob":
        ll = get_artg_loglike(samples, params)
        L = params.W.shape[2]
        T = samples.shape[1]
        return -ll / (T - L)
    elif method == "vm_entropy":
        return estimate_artg_entropy_rates(samples, params)
    else:
        raise NotImplementedError(method)


_fit_both_univariate_mle_bfgs = jax.vmap(
    fit_univariate_artg_mle_bfgs,
    in_axes=(None, 0, None, None),
    out_axes=(0, 0),
)


def _fit_both_univariate_score_matching(key, X, L, n_iter, batch_size):
    d = X.shape[0]
    Ws = []
    for i in range(d):
        key, subkey = jax.random.split(key)
        W = fit_univariate_artg_score_matching(subkey, X[i], L, n_iter, batch_size)
        Ws.append(W)
    return jnp.stack(Ws), None


def estimate_transfer_entropies(
    key,
    samples,
    samples_test,
    bivariate_params=None,
    L=5,
    prior=None,
    objective: str = "mle",
    fit_method: str = "bfgs",
    entropy_method: str = "log_prob",
    n_iter: int = 500,
    batch_size: int = 64,
):
    """
    Estimate the transfer entropies of the bivariate signal.

    Parameters
    ----------
    key : jax.random.PRNGKey
    samples : jnp.ndarray
        Shape : (2, T)
    samples_test : jnp.ndarray
        Shape: (2, T_test)
    bivariate_params : None or ARTGParams
    L : int, optional
    batch_size : int, optional

    Returns
    -------
    te : jnp.ndarray
        [T_{1 → 2}, T_{2 → 1}]
    """
    assert objective in ["mle", "score_matching"]
    assert fit_method in ["bfgs", "adam"]
    assert entropy_method in ["log_prob", "vm_entropy"]
    
    # Split key.
    key1, key2 = jax.random.split(key)

    # Fit the univariate models.
    if objective == "mle":
        if fit_method == "bfgs":
            W_u, _ = _fit_both_univariate_mle_bfgs(key1, samples, L, prior) # (2, L, 2)
        else:
            raise NotImplementedError
    else:
        if fit_method == "bfgs":
            raise NotImplementedError # TODO
        else:
            W_u, _ = _fit_both_univariate_score_matching(key1, samples, L, n_iter, batch_size) # (2, L, 2)
    univariate_params = ARTGParams(univariate_to_bivariate(W_u))

    # Fit the bivariate models.
    if bivariate_params is None:
        if objective == "mle":
            if fit_method == "bfgs":
                W_b, _ = fit_bivariate_artg_bfgs(key2, samples, L, prior=prior)
            else:
                W_b = fit_bivariate_artg_ssm(key2, samples, L, prior=prior) # (2_out, 2_in, L, 2)
        else:
            if fit_method == "bfgs":
                raise NotImplementedError # TODO
            else:
                W_b = fit_bivariate_artg_score_matching(key2, samples, L, n_iter, batch_size)
        bivariate_params = ARTGParams(W_b)

    # Estimate univariate and bivariate entropy rates.
    er_u = estimate_entropy_rates(samples_test, univariate_params, method=entropy_method)
    er_b = estimate_entropy_rates(samples_test, bivariate_params, method=entropy_method)

    # Form TE estimates and return.
    te = er_u - er_b
    return te[::-1] # (2,)


def estimate_all_pairwise_transfer_entropies(
    key,
    samples,
    samples_test,
    L=5,
    prior=None,
    objective: str = "mle",
    fit_method: str = "bfgs",
    entropy_method: str = "log_prob",
    n_iter: int = 500,
    batch_size: int = 64,
):
    """
    Estimate pairwise transfer entropies for a multivariate phase time series.

    Parameters
    ----------
    key : jax.random.PRNGKey
    samples : jnp.ndarray, shape (d, T)
        Training data.
    samples_test : jnp.ndarray, shape (d, T_test)
        Held-out data for entropy rate estimation.
    L : int, optional
        Lag parameter for the AR-TG models.
    prior : None or ARTGPriorParams
    objective : str
    fit_method : str
    entropy_method : str
    n_iter : int

    Returns
    -------
    te_mat : jnp.ndarray, shape (d, d)
        ``te_mat[i, j]`` contains the estimated transfer entropy T_{i → j}.
    """
    d = samples.shape[0]
    te_mat = jnp.zeros((d, d))

    kwargs = dict(
        L=L,
        prior=prior,
        objective=objective,
        fit_method=fit_method,
        entropy_method=entropy_method,
        n_iter=n_iter,
        batch_size=batch_size,
    )

    for i in range(d):
        for j in range(i + 1, d):
            key, subkey = jax.random.split(key)
            pair_samples = jnp.stack([samples[i], samples[j]])
            pair_test = jnp.stack([samples_test[i], samples_test[j]])
            te = estimate_transfer_entropies(
                subkey,
                pair_samples,
                pair_test,
                **kwargs,
            )
            te_mat = te_mat.at[i, j].set(te[0])
            te_mat = te_mat.at[j, i].set(te[1])

    return te_mat
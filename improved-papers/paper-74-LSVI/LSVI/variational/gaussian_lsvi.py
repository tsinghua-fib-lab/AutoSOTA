from typing import Callable

import jax
import jax.numpy as jnp

from variational.exponential_family import GenericNormalDistribution
from variational.utils import vec, unvec, get_residual


# jax.config.update('jax_platform_name', 'cpu')  # Required to use sqrtm.


def gaussian_lsvi(OP_key: jax.Array, tgt_log_density: Callable, eta_init: jnp.ndarray, n_iter: int, n_samples: int,
                  lr_schedule=1.0, return_all=False,
                  target_residual_schedule=jnp.inf):
    """
    Dense Gaussian scheme.
    See Section 4. of https://arxiv.org/abs/2502.18475,
    Least Squares Variational Inference, Le Fay Y., Chopin N. Barthelmé S. 2025.

    Important note:
        To clearly understand each intermediary steps in converting the new regressor (\gamma) to
        the natural parameter (\eta), it is highly advised to properly read and understand
        the appendix section D.4 in Least Squares Variational Inference.
    """
    # jax.config.update('jax_platform_name', 'cpu')  # Required to use jax.scipy.linalg.sqrtm.
    dimension = int(jnp.sqrt(len(eta_init) - 3 / 4) - 1 / 2)
    normal = GenericNormalDistribution(dimension=dimension)
    sanity = normal.sanity
    statistic = jax.vmap(normal.sufficient_statistic)

    def from_gammatildetilde_to_gammatilde(gammatildetilde):
        gamma1tilde = gammatildetilde.at[:dimension].get()
        gamma0tilde = gammatildetilde.at[-1].get()
        # Next lines are due to Eq. 64. In this function, we first create a symmetric square matrix for the vec. gammatildetilde2.
        gamma2tildetilde = gammatildetilde.at[dimension:int(dimension * (dimension + 1) / 2) + dimension].get()
        gamma2tilde_matrix = jnp.zeros((dimension, dimension))
        gamma2tilde_matrix = 0.5 * gamma2tilde_matrix.at[jnp.triu_indices(dimension)].set(gamma2tildetilde)
        gamma2tilde_matrix = gamma2tilde_matrix + gamma2tilde_matrix.T
        gamma2tilde = gamma2tilde_matrix.reshape(-1)
        return jnp.concatenate([gamma1tilde, gamma2tilde, jnp.array([gamma0tilde])])

    def from_gammatilde_to_gamma(gammatilde):
        gamma1 = gammatilde.at[:dimension].get()
        gamma2 = gammatilde.at[dimension:dimension ** 2 + dimension].get()
        gamma2_of_interest = gamma2.at[0::(dimension + 1)].get()
        gamma2 = gamma2.at[0::(dimension + 1)].set(gamma2_of_interest
                                                   * 1 / jnp.sqrt(
            2))  # We need to scale the diagonal by 1/sqrt(2). This is due to the line following Eq. 63.
        gamma0 = gammatilde.at[-1].get() - jnp.sum(gamma2_of_interest) * 1 / jnp.sqrt(
            2)  # We need to change the intercept, again due to the line following Eq. 63.
        return jnp.concatenate([gamma1, gamma2, jnp.array([gamma0])])

    def from_gamma_to_eta(current_mean, current_sqrt, gamma):
        # This gamma is \hat{\gamma} is the Appendix D.4. We can recover \eta from Eqs. 56, 58, and 59.
        inv_chol = jax.scipy.linalg.inv(current_sqrt)  # O(n^3/2)
        gamma2 = gamma.at[dimension:dimension ** 2 + dimension].get()
        gamma1 = gamma.at[:dimension].get()
        gamma0 = gamma.at[-1].get()
        B = unvec(gamma2, shape=(dimension, dimension))
        eta2 = vec(inv_chol @ B @ inv_chol.T)  # O(n^3) equal to jnp.kron(inv_chol, inv_chol)@gamma2, Eq. 56
        eta1 = ((gamma1.T - 2 * eta2.T @ (jnp.kron(current_mean[:, jnp.newaxis], current_sqrt))) @ inv_chol).T  # Eq. 58
        eta0 = gamma0 - eta1.T @ current_mean - eta2.T @ vec(
            current_mean[:, jnp.newaxis] @ current_mean[:, jnp.newaxis].T)  # Eq. 59
        eta = jnp.concatenate([eta1, eta2, jnp.array([eta0])])
        return eta

    def from_gammatildetilde_to_gamma(gammatildetilde):
        return from_gammatilde_to_gamma(from_gammatildetilde_to_gammatilde(gammatildetilde))

    @jax.vmap
    def modified_statistic(z):
        # This statistic is given in Eq. 16.
        vecZZt = vec(z[:, jnp.newaxis] @ z[:, jnp.newaxis].T)
        vecZZt = vecZZt.at[0::(dimension + 1)].set((vecZZt.at[0::(dimension + 1)].get() - 1) / jnp.sqrt(2))
        vectriuunvecvecZZt = vec(unvec(vecZZt, (dimension, dimension)).at[jnp.triu_indices(dimension)].get())
        return jnp.concatenate([z, vectriuunvecvecZZt, jnp.array([1.])])

    def momentum_backtracking(lr, eta, next_eta, y, X, target_residual):
        lr = jax.lax.while_loop(lambda _lr: sanity(next_eta * _lr + (1 - _lr) * eta),
                                lambda _lr: _lr / 2, lr)
        current_residual = get_residual(y, X, next_eta * lr + (1 - lr) * eta)
        lr_tempering = jax.lax.cond(current_residual <= target_residual, lambda _: lr,
                                    lambda _: jnp.sqrt(target_residual / current_residual), None)
        lr = jax.lax.min(lr, lr_tempering)
        new_residual = get_residual(y, X, next_eta * lr + (1 - lr) * eta)
        return lr, new_residual

    def iter_routine(eta, inps):
        # See Alg. 3.
        key, lr, target_residual = inps
        theta = eta.at[:-1].get()
        current_mean, current_cov = normal.get_mean_cov(theta)
        # sqrtm = jnp.real(jax.scipy.linalg.sqrtm(current_cov))  # seems more stable, but CPU only compatible.
        # sqrtm = jax.scipy.linalg.cholesky(current_cov) # numerical issues
        # sqrtm = jax.scipy.linalg.cholesky((current_cov + current_cov.T) / 2)  # seems more stable and GPU compatible
        D, V = jax.scipy.linalg.eigh(current_cov)
        sqrtm = (V * jnp.sqrt(D)) @ V.T
        samples = sampling(key)
        y = vmapped_tgt_log_density(current_mean[jnp.newaxis, :] + samples @ sqrtm)
        X = modified_statistic(samples)
        next_gamma_tilde_tilde = X.T @ y / n_samples  # OLS(X, y) works well..
        next_gamma = from_gammatildetilde_to_gamma(next_gamma_tilde_tilde)
        next_eta = from_gamma_to_eta(current_mean, sqrtm, next_gamma)
        lr, residual = momentum_backtracking(lr, eta, next_eta, y, statistic(samples), target_residual)
        next_eta = next_eta * lr + (1 - lr) * eta
        return next_eta, next_gamma_tilde_tilde, residual

    def fun_iter(eta, inps):
        next_eta, *_ = iter_routine(eta, inps)
        return next_eta, next_eta

    def fun_iter_return_all(eta, inps):
        next_eta, next_gamma_tilde_tilde, residual = iter_routine(eta, inps)
        return next_eta, (next_eta, next_gamma_tilde_tilde, residual)

    sampling = lambda keys: jax.random.normal(keys, shape=(n_samples, dimension))
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    if isinstance(target_residual_schedule, float):
        target_residual_schedule = jnp.full(n_iter, target_residual_schedule)

    if return_all:
        _, all_results = jax.lax.scan(fun_iter_return_all, eta_init,
                                      (iter_keys, lr_schedule, target_residual_schedule))
        etas = all_results[0]
        etas = jnp.insert(etas, 0, eta_init, axis=0)
        return etas, all_results[1:]
    else:
        _, etas = jax.lax.scan(fun_iter, eta_init, (iter_keys, lr_schedule, target_residual_schedule))
        etas = jnp.insert(etas, 0, eta_init, axis=0)
        return etas, None

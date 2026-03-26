import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as slinalg
from scipy.stats import qmc

from variational.utils import unvec, vec


def mean_field_gaussian_lsvi(tgt_log_density, eta_init, n_iter, n_samples, lr_schedule=1.0, backtracking=True,
                             sampling_method="standard"):
    """
    Mean-field Gaussian scheme.
    See gaussian_lsvi.py for the documentation.
    """
    dimension = int((len(eta_init) - 1) / 2)

    def get_mean_cov(theta):
        vec_diag_cov = 1. / (-2 * theta[dimension:])
        mean = vec_diag_cov * theta[:dimension]
        return mean, np.diag(vec_diag_cov)

    if isinstance(lr_schedule, float):
        lr_schedule = np.full(n_iter, lr_schedule)

    def from_gamma_to_eta(current_mean, current_vec_diag_cov, gamma):
        gamma2 = gamma[dimension:2 * dimension]
        gamma1 = gamma[:dimension]
        gamma0 = gamma[-1]
        eta2 = gamma2 * 1 / current_vec_diag_cov * 1 / np.sqrt(2)
        eta1 = gamma1 * (1 / np.sqrt(current_vec_diag_cov)) - 2 * eta2 * current_mean
        eta0 = gamma0 - eta1.T @ current_mean - eta2.T @ (current_mean ** 2 + current_vec_diag_cov)
        eta = np.concatenate([eta1, eta2, np.array([eta0])])
        return eta

    @jax.vmap
    def modified_statistic(z):
        return jnp.concatenate([z, (z ** 2 - 1) / jnp.sqrt(2), jnp.array([1.])])

    def sanity(eta):
        mean, cov = get_mean_cov(eta[:-1])
        return np.isnan(np.random.multivariate_normal(mean=mean, cov=cov)).any()

    def momentum_backtracking(lr, eta, next_eta):
        while sanity(next_eta + lr + (1 - lr) * eta):
            lr /= 2
        return lr

    etas = np.array([eta_init] * (n_iter + 1))

    if sampling_method == "qmc":
        dist_qmc = qmc.MultivariateNormalQMC(mean=np.zeros(dimension), cov_root=np.identity(dimension))

        def sampling(n_samples):
            return dist_qmc.random(n_samples)
    else:
        def sampling(n_samples):
            return np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension), size=n_samples)

    vmapped_tgt_log_density = jax.vmap(tgt_log_density)

    for i_iter in range(1, n_iter + 1):
        lr = lr_schedule[i_iter - 1]
        current_eta = etas[i_iter]
        theta = current_eta[:-1]
        current_mean, current_vec_diag_cov = get_mean_cov(theta)
        current_vec_diag_cov = np.diag(current_vec_diag_cov)
        samples = sampling(n_samples)
        y = vmapped_tgt_log_density(current_mean + np.sqrt(current_vec_diag_cov) * samples)
        X = modified_statistic(samples)
        next_gamma = X.T @ y / n_samples
        next_eta = from_gamma_to_eta(current_mean, current_vec_diag_cov, next_gamma)
        lr = momentum_backtracking(lr, current_eta, next_eta) if backtracking else lr
        next_eta = next_eta * lr + (1 - lr) * current_eta
        etas[i_iter] = next_eta

    return next_eta, None


def gaussian_lsvi(tgt_log_density, eta_init, n_iter, n_samples, lr_schedule=1.0,
                  backtracking=True, sampling_method="standard"):
    """
    Dense Gaussian scheme following Nicolas' note.
    """
    dimension = int(np.sqrt(len(eta_init) - 3 / 4) - 1 / 2)

    def get_mean_cov(theta):
        """
        Given the natural parameter theta, returns the mean and the covariance matrix.
        """
        invcov = -2 * theta[dimension:]
        cov = np.linalg.pinv(unvec(invcov, (dimension, dimension)))
        mean = cov @ theta[:dimension]
        return mean, cov

    def from_gammatildetilde_to_gammatilde(gammatildetilde):
        gamma2tildetilde = gammatildetilde[dimension:int(dimension * (dimension + 1) / 2) + dimension]
        gamma1tilde = gammatildetilde[:dimension]
        gamma0tilde = gammatildetilde[-1]
        gamma2tilde_matrix = np.zeros((dimension, dimension))
        gamma2tilde_matrix[np.triu_indices(dimension)] = gamma2tildetilde
        gamma2tilde_matrix = 0.5 * gamma2tilde_matrix
        gamma2tilde_matrix = gamma2tilde_matrix + gamma2tilde_matrix.T
        gamma2tilde = gamma2tilde_matrix.reshape(-1)
        return np.concatenate([gamma1tilde, gamma2tilde, np.array([gamma0tilde])])

    def from_gammatilde_to_gamma(gammatilde):
        gamma1 = gammatilde[:dimension]
        gamma2 = gammatilde[dimension:dimension ** 2 + dimension]
        gamma2_of_interest = gamma2[0::(dimension + 1)]
        gamma2[0::(dimension + 1)] = gamma2_of_interest * 1 / np.sqrt(2)
        gamma0 = gammatilde[-1] - np.sum(gamma2_of_interest) * 1 / np.sqrt(2)
        return np.concatenate([gamma1, gamma2, np.array([gamma0])])

    def from_gamma_to_eta(current_mean, current_sqrt, gamma):
        inv_chol = slinalg.inv(current_sqrt)  # O(n^3/2)
        gamma2 = gamma[dimension:dimension ** 2 + dimension]
        gamma1 = gamma[:dimension]
        gamma0 = gamma[-1]
        B = unvec(gamma2)
        eta2 = vec(inv_chol @ B @ inv_chol.T)  # O(n^3) equal to jnp.kron(inv_chol, inv_chol)@gamma2
        eta1 = ((gamma1.T - 2 * eta2.T @ (np.kron(current_mean[:, np.newaxis], current_sqrt))) @ inv_chol).T
        eta0 = gamma0 - eta1.T @ current_mean - eta2.T @ vec(
            current_mean[:, np.newaxis] @ current_mean[:, np.newaxis].T)
        eta = np.concatenate([eta1, eta2, np.array([eta0])])
        return eta

    def from_gammatildetilde_to_gamma(gammatildetilde):
        return from_gammatilde_to_gamma(from_gammatildetilde_to_gammatilde(gammatildetilde))

    @jax.vmap
    def modified_statistic(z):
        vecZZt = vec(z[:, jnp.newaxis] @ z[:, jnp.newaxis].T)
        vecZZt = vecZZt.at[0::(dimension + 1)].set((vecZZt.at[0::(dimension + 1)].get() - 1) / jnp.sqrt(2))
        vectriuunvecvecZZt = vec(unvec(vecZZt, (dimension, dimension)).at[jnp.triu_indices(dimension)].get())
        return jnp.concatenate([z, vectriuunvecvecZZt, jnp.array([1.])])

    def sanity(eta):
        mean, cov = get_mean_cov(eta[:-1])
        return np.isnan(np.random.multivariate_normal(mean=mean, cov=cov)).any()

    def momentum_backtracking(lr, eta, next_eta):
        while sanity(next_eta * lr + (1 - lr) * eta):
            lr /= 2
        return lr

    if sampling_method == "qmc":
        dist_qmc = qmc.MultivariateNormalQMC(mean=np.zeros(dimension), cov_root=np.identity(dimension))

        def sampling(n_samples):
            return dist_qmc.random(n_samples)
    else:
        def sampling(n_samples):
            return np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension), size=n_samples)

    vmapped_tgt_log_density = jax.vmap(tgt_log_density)

    if isinstance(lr_schedule, float):
        lr_schedule = np.full(n_iter, lr_schedule)

    etas = np.array([eta_init] * (n_iter + 1))
    for i_iter in tqdm(range(1, n_iter + 1)):
        lr = lr_schedule[i_iter - 1]
        current_eta = etas[i_iter]
        theta = current_eta[:-1]
        current_mean, current_cov = get_mean_cov(theta)
        sqrtm = np.real(slinalg.sqrtm(current_cov))
        samples = sampling(n_samples)
        y = vmapped_tgt_log_density(current_mean[np.newaxis, :] + samples @ sqrtm)
        X = modified_statistic(samples)
        next_gamma_tilde_tilde = X.T @ y / n_samples  # OLS(X, y) works well..
        next_gamma = from_gammatildetilde_to_gamma(next_gamma_tilde_tilde)
        next_eta = from_gamma_to_eta(current_mean, sqrtm, next_gamma)
        lr = momentum_backtracking(lr, current_eta, next_eta) if backtracking else lr
        next_eta = next_eta * lr + (1 - lr) * current_eta
        etas[i_iter] = next_eta

    return etas, None

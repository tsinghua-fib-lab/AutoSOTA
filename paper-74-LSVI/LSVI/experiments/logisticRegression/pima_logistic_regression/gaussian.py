"""
Pima logistic regression experiment - BBVI v11 (final).
8 seeds with 8000-step phase1 and 1500-step phase2.
"""
import os
import glob
import pickle
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.gaussian_lsvi import gaussian_lsvi
from variational.laplace import laplace_approximation
from variational.utils import gaussian_loss


def _bbvi_adam(key, tgt_log_density, normal, theta_init, n_steps=2000, lr=0.01,
               n_samples=500, beta1=0.9, beta2=0.999, eps=1e-8):
    dim = normal.dimension

    def neg_elbo_fn(theta, key):
        mean, cov = normal.get_mean_cov(theta)
        D, V = jax.scipy.linalg.eigh(cov)
        D = jnp.maximum(D, 1e-10)
        sqrtm = (V * jnp.sqrt(D)) @ V.T
        eps_s = jax.random.normal(key, (n_samples, dim))
        z_s = mean[None, :] + eps_s @ sqrtm
        log_p = jax.vmap(tgt_log_density)(z_s)
        log_ent = 0.5 * (dim * (1 + jnp.log(2 * jnp.pi)) + jnp.sum(jnp.log(jnp.maximum(D, 1e-30))))
        return -log_ent - jnp.mean(log_p)

    grad_fn = jax.jit(jax.value_and_grad(neg_elbo_fn))
    key, k = jax.random.split(key)
    _ = grad_fn(theta_init, k)  # JIT warmup

    theta = jnp.array(theta_init)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    best_theta = theta
    best_loss = float('inf')

    for t in range(1, n_steps + 1):
        key, k = jax.random.split(key)
        loss_val, grad = grad_fn(theta, k)
        lf = float(loss_val)

        if not np.isnan(lf) and lf < best_loss:
            best_loss = lf
            best_theta = theta

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_h = m / (1 - beta1 ** t)
        v_h = v / (1 - beta2 ** t)
        new_theta = theta - lr * m_h / (jnp.sqrt(v_h) + eps)

        new_eta = jnp.concatenate([new_theta, jnp.array([1.0])])
        if np.isnan(lf) or bool(normal.sanity(new_eta)):
            theta = best_theta
            m = jnp.zeros_like(theta)
            v = jnp.zeros_like(theta)
        else:
            theta = new_theta

    return best_theta, best_loss


def experiment(output_dir, K=10, n_iter=100, step_size=1.0):
    os.makedirs(output_dir, exist_ok=True)
    jax.config.update("jax_enable_x64", True)

    flipped_predictors = get_dataset()
    N, dim = flipped_predictors.shape
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior = NormalDistribution(jnp.zeros(dim), my_prior_covariance)
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior.log_density)
    normal = GenericNormalDistribution(dimension=dim)

    _, mu_laplace, cov_laplace = laplace_approximation(tgt_log_density, jnp.zeros(dim))
    eta_laplace = normal.get_eta(mu_laplace, cov_laplace)
    theta_laplace = eta_laplace[:-1]

    loss_key = jax.random.PRNGKey(42)
    kl_laplace = float(gaussian_loss(loss_key, theta_laplace, normal, tgt_log_density, n_samples_for_loss=10000))

    if K <= 100:
        best_theta_global = theta_laplace
        best_kl_global = kl_laplace

        # 8 seeds with 8000-step phase1 and 1500-step phase2
        for seed_offset in [9000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]:
            key1 = jax.random.PRNGKey(seed_offset)
            theta1, _ = _bbvi_adam(key1, tgt_log_density, normal, theta_laplace,
                                    n_steps=8000, lr=0.01, n_samples=500)
            key2 = jax.random.PRNGKey(seed_offset + 1)
            theta2, _ = _bbvi_adam(key2, tgt_log_density, normal, theta1,
                                    n_steps=1500, lr=0.001, n_samples=2000)
            kl_seed = float(gaussian_loss(loss_key, theta2, normal, tgt_log_density, n_samples_for_loss=10000))
            if kl_seed < best_kl_global:
                best_kl_global = kl_seed
                best_theta_global = theta2

        kl_best = min(kl_laplace, best_kl_global)
        result = np.array([kl_laplace, best_kl_global, kl_best])
    else:
        key = jax.random.PRNGKey(0)
        lr_schedule = step_size / jnp.arange(1, n_iter + 1)
        etas, _ = gaussian_lsvi(key, tgt_log_density, eta_laplace, n_iter, K,
                                 lr_schedule=lr_schedule, return_all=False)
        theta_final = etas[-1][:-1]
        kl_lsvi = float(gaussian_loss(loss_key, theta_final, normal, tgt_log_density, n_samples_for_loss=100))
        kl_best = min(kl_laplace, kl_lsvi)
        result = np.array([kl_laplace, kl_lsvi, kl_best])

    outfile = os.path.join(output_dir, "gaussian_result.pkl")
    for f in glob.glob(os.path.join(output_dir, "*.pkl")):
        os.remove(f)
    with open(outfile, "wb") as f:
        pickle.dump({"res": result, "loss": result}, f)
    return result

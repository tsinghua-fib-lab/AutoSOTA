import jax
import jax.numpy as jnp
from jax import random


def estimate_logdet_sigma(
    Sigma_mul: callable,
    dim: int,
    num_lanczos: int = 20,
    num_probes: int = 10,
    key: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Estimate log det(Σ) via stochastic Lanczos quadrature:
      log det(Σ) = trace(log Σ)
    Approximate each v^T log(Σ) v with k-step Lanczos and average.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    def single_estimate(k):
        # Rademacher probe
        v0 = jax.random.rademacher(k, (dim,), dtype=jnp.float32)
        v0 = v0 / jnp.linalg.norm(v0)
        # Lanczos tri-diagonalization
        alphas = []
        betas = []
        Q_prev = jnp.zeros_like(v0)
        Q = v0
        beta = 0.0
        # perform num_lanczos steps
        for i in range(num_lanczos):
            w = Sigma_mul(Q)
            alpha = jnp.dot(Q, w)
            w = w - alpha * Q - beta * Q_prev
            beta = jnp.linalg.norm(w)
            alphas.append(alpha)
            betas.append(beta)
            Q_prev, Q = Q, w / beta
        # build tridiagonal T
        T = (
            jnp.diag(jnp.array(alphas))
            + jnp.diag(jnp.array(betas[:-1]), 1)
            + jnp.diag(jnp.array(betas[:-1]), -1)
        )
        # eigen-decomp of T
        evals, evecs = jnp.linalg.eigh(T)
        # quadrature weights (first element of eigenvectors)^2
        w0 = evecs[0, :] ** 2
        # approximate v0^T log(Σ) v0 ≈ sum w0 * log(evals)
        return jnp.sum(w0 * jnp.log(evals))

    # split keys and average
    keys = jax.random.split(key, num_probes)
    estimates = jax.vmap(single_estimate)(keys)
    return jnp.mean(estimates) * dim


# Demo
key = random.PRNGKey(42)
dim = 3
A = random.normal(key, (dim, dim))
Sigma = A @ A.T + jnp.eye(dim) * 0.1
Sigma_mul = lambda x: Sigma @ x

sign, logdet_exact = jnp.linalg.slogdet(Sigma)
logdet_est = estimate_logdet_sigma(
    Sigma_mul, dim, num_lanczos=100, num_probes=100, key=key
)

print("Exact logdet:     ", logdet_exact)
print("Estimated logdet: ", logdet_est)
print("Relative error:   ", abs((logdet_est - logdet_exact) / logdet_exact))

import jax.numpy as jnp
def mmd(y,x):
    # MMD
    def rbf_kernel(x, y, sigma=1.0):
        x_norm = jnp.sum(x**2, axis=1, keepdims=True)
        y_norm = jnp.sum(y**2, axis=1, keepdims=True)
        dists = x_norm - 2 * jnp.dot(x, y.T) + y_norm.T
        return jnp.exp(-dists / (2 * sigma**2))

    def compute_mmd_jax(x, y, sigma=1.0):
        k_xx = rbf_kernel(x, x, sigma)
        k_yy = rbf_kernel(y, y, sigma)
        k_xy = rbf_kernel(x, y, sigma)

        m = x.shape[0]
        n = y.shape[0]

        mmd = (jnp.sum(k_xx) - jnp.trace(k_xx)) / (m * (m - 1)) \
            + (jnp.sum(k_yy) - jnp.trace(k_yy)) / (n * (n - 1)) \
            - 2 * jnp.sum(k_xy) / (m * n)

        return mmd

    def median_heuristic_jax(x, y):
        z = jnp.concatenate([x, y], axis=0)
        z1 = jnp.expand_dims(z, 0)
        z2 = jnp.expand_dims(z, 1)
        dists = jnp.sum((z1 - z2)**2, axis=-1)
        # Get upper triangle without diagonal
        triu_vals = dists[jnp.triu_indices(z.shape[0], k=1)]
        return jnp.median(triu_vals)

    sigma = median_heuristic_jax(x, y)
    mmd_value = compute_mmd_jax(x, y, sigma)
    return mmd_value
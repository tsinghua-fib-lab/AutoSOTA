"""
Data Utilities for 1D Heat Equation Decentralized DPC Example
Generates smooth Gaussian Random Fields with zero boundary conditions
"""
import jax
import jax.numpy as jnp

def rbf_kernel(x1, x2, length_scale=0.2, sigma=1.0):
    """
    Computes the RBF (Squared Exponential) kernel covariance matrix.
    Args:
        x1: array of shape (N, D) or (N,)
        x2: array of shape (M, D) or (M,)
        length_scale: scale parameter controlling smoothness
        sigma: signal variance
    """
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
        
    dist_sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)

def generate_grf(key, n_points=100, length_scale=0.4, sigma=1.0):
    """
    Generates a smooth Gaussian Random Field on [0, 1] with zero boundary conditions.
    Uses Eigenvalue Decomposition for better numerical stability with smooth kernels.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        x_grid = jnp.linspace(0, 1, n_points)
        
        # 1. Compute Covariance Matrix
        K = rbf_kernel(x_grid, x_grid, length_scale, sigma)
        
        # 2. Eigenvalue decomposition for stable sampling
        # K = V @ diag(w) @ V.T
        w, V = jnp.linalg.eigh(K)
        
        # Clip negative/tiny eigenvalues to zero and take square root
        # This effectively removes the high-frequency noise from the nugget
        w_sqrt = jnp.sqrt(jnp.maximum(w, 1e-12))
        
        z = jax.random.normal(key, shape=(n_points,))
        f_sample = V @ (w_sqrt * z)
    
    # 3. Apply Bridge Correction to enforce f(0) = 0 and f(1) = 0
    f_0 = f_sample[0]
    f_1 = f_sample[-1]
    
    linear_trend = f_0 + x_grid * (f_1 - f_0)
    field = f_sample - linear_trend
    
    return x_grid, field


if __name__ == "__main__":
    # Test block to visualize if run directly
    import matplotlib.pyplot as plt
    
    print("Generating GRF samples...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    plt.figure(figsize=(10, 6))
    
    for k in keys:
        x, y = generate_grf(k, n_points=100, length_scale=0.4)
        plt.plot(x, y)
        
    plt.title("Smooth GRF Samples with Zero Boundaries")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("grf_samples.png")
    print("Saved grf_samples.png")

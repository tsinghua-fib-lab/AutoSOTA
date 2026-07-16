"""Data utilities for 2D Heat Equation Control example.
Generates smooth Gaussian Random Fields with zero Dirichlet boundary conditions."""
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path

def rbf_kernel_2d(x1, x2, length_scale=0.3, sigma=1.0):
    """
    2D RBF kernel for GRF generation.

    Args:
        x1: Grid points (N1, 2)
        x2: Grid points (N2, 2)
        length_scale: Correlation length
        sigma: Signal variance

    Returns:
        Covariance matrix (N1, N2)
    """
    # Pairwise distances: (N1, 1, 2) - (1, N2, 2) -> (N1, N2, 2)
    diff = x1[:, None, :] - x2[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)  # (N1, N2)
    return sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)

def generate_grf_2d(key, n_points=32, length_scale=0.4, sigma=1.0):
    """
    Generate 2D Gaussian Random Field with zero Dirichlet BCs.

    Strategy:
    1. Generate GRF on full grid using eigendecomposition
    2. Apply 2D bridge correction to enforce z=0 on all edges

    Args:
        key: JAX random key
        n_points: Number of grid points per dimension
        length_scale: Correlation length for smoothness
        sigma: Signal variance

    Returns:
        meshgrid (xx, yy), field (N, N)
    """
    with jax.default_device(jax.devices("cpu")[0]):  # CPU for eigen-decomp
        # Grid
        x_grid = jnp.linspace(0, 1, n_points)
        y_grid = jnp.linspace(0, 1, n_points)
        xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')

        # Flatten grid to (N², 2)
        grid_flat = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

        # Covariance matrix (N², N²) - this is large!
        K = rbf_kernel_2d(grid_flat, grid_flat, length_scale, sigma)

        # Eigendecomposition
        w, V = jnp.linalg.eigh(K)
        w_sqrt = jnp.sqrt(jnp.maximum(w, 1e-12))

        # Sample
        z = jax.random.normal(key, shape=(n_points**2,))
        f_flat = V @ (w_sqrt * z)

    # Reshape to 2D
    f = f_flat.reshape((n_points, n_points))

    # --- 2D Bridge Correction ---
    # We need to enforce f=0 on edges while minimizing distortion
    # Use bilinear interpolation from edges

    # Extract edge values
    f_left = f[0, :]      # x=0 edge
    f_right = f[-1, :]    # x=1 edge
    f_bottom = f[:, 0]    # y=0 edge
    f_top = f[:, -1]      # y=1 edge

    # Extract corner values
    f_00 = f[0, 0]
    f_10 = f[-1, 0]
    f_01 = f[0, -1]
    f_11 = f[-1, -1]

    # Build trend surface using bilinear interpolation
    x_norm = xx
    y_norm = yy

    # Corner contribution (bilinear interpolation of corners)
    corner_trend = ((1 - x_norm) * (1 - y_norm) * f_00 +
                    x_norm * (1 - y_norm) * f_10 +
                    (1 - x_norm) * y_norm * f_01 +
                    x_norm * y_norm * f_11)

    # Edge contributions (linear interpolation along each edge, minus corner overlap)
    # Left edge: varies with y, weighted by (1-x)
    left_edge_interp = (1 - y_norm) * f_00 + y_norm * f_01
    left_contrib = (1 - x_norm) * (f_left[None, :] - left_edge_interp)

    # Right edge: varies with y, weighted by x
    right_edge_interp = (1 - y_norm) * f_10 + y_norm * f_11
    right_contrib = x_norm * (f_right[None, :] - right_edge_interp)

    # Bottom edge: varies with x, weighted by (1-y)
    bottom_edge_interp = (1 - x_norm) * f_00 + x_norm * f_10
    bottom_contrib = (1 - y_norm) * (f_bottom[:, None] - bottom_edge_interp)

    # Top edge: varies with x, weighted by y
    top_edge_interp = (1 - x_norm) * f_01 + x_norm * f_11
    top_contrib = y_norm * (f_top[:, None] - top_edge_interp)

    # Total trend surface
    trend = corner_trend + left_contrib + right_contrib + bottom_contrib + top_contrib

    # Subtract trend to get zero BCs
    field = f - trend

    return xx, yy, field


def load_dataset(dataset_path):
    """
    Load pre-generated dataset from .npz file.

    Args:
        dataset_path: Path to .npz file

    Returns:
        z_init_all: (n_samples, n_grid, n_grid) initial conditions
        z_target_all: (n_samples, n_grid, n_grid) target conditions
        n_grid: Grid size
    """
    data = np.load(dataset_path)
    z_init_all = jnp.array(data['z_init'])
    z_target_all = jnp.array(data['z_target'])
    n_grid = int(data['grid_size'])

    print(f"Loaded dataset from {dataset_path}")
    print(f"  Shape: {z_init_all.shape}")
    print(f"  Grid size: {n_grid}×{n_grid}")

    return z_init_all, z_target_all, n_grid


def generate_dataset(n_samples, n_points=32, length_scale_init=0.25,
                     length_scale_target=0.4, seed=42, verbose=True):
    """
    Generate multiple GRF samples with progress reporting.

    Args:
        n_samples: Number of (init, target) pairs to generate
        n_points: Grid resolution (default 32 for speed)
        length_scale_init: Length scale for initial conditions
        length_scale_target: Length scale for targets
        seed: Random seed
        verbose: Print progress

    Returns:
        z_init_all: (n_samples, n_points, n_points)
        z_target_all: (n_samples, n_points, n_points)
    """
    if verbose:
        print(f"Generating dataset: {n_samples} samples at {n_points}×{n_points} grid")

    key = jax.random.PRNGKey(seed)
    all_keys = jax.random.split(key, n_samples * 2)

    z_init_list = []
    z_target_list = []

    start_time = time.time()
    last_print_time = start_time

    # Generate initial conditions
    if verbose:
        print("Generating initial conditions...")
    for i in range(n_samples):
        # Progress reporting every 50 samples or 5 seconds
        if verbose:
            current_time = time.time()
            if i % 50 == 0 or (current_time - last_print_time) > 5.0:
                if i > 0:
                    elapsed = current_time - start_time
                    time_per_sample = elapsed / i
                    remaining = (n_samples - i) * time_per_sample
                    print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%) | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
                    last_print_time = current_time
                else:
                    print(f"  Progress: {i}/{n_samples}")

        _, _, z_i = generate_grf_2d(all_keys[i], n_points=n_points,
                                     length_scale=length_scale_init)
        z_init_list.append(z_i)

    if verbose:
        print(f"  Completed: {n_samples}/{n_samples} (100.0%)")

    # Generate target conditions
    if verbose:
        print("Generating target conditions...")
    target_start = time.time()
    last_print_time = target_start

    for i in range(n_samples):
        # Progress reporting
        if verbose:
            current_time = time.time()
            if i % 50 == 0 or (current_time - last_print_time) > 5.0:
                if i > 0:
                    elapsed = current_time - target_start
                    time_per_sample = elapsed / i
                    remaining = (n_samples - i) * time_per_sample
                    print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%) | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
                    last_print_time = current_time
                else:
                    print(f"  Progress: {i}/{n_samples}")

        _, _, z_t = generate_grf_2d(all_keys[n_samples + i], n_points=n_points,
                                     length_scale=length_scale_target)
        z_target_list.append(z_t)

    if verbose:
        print(f"  Completed: {n_samples}/{n_samples} (100.0%)")
        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.1f}s ({total_time/60:.1f}m)")

    # Convert to arrays
    z_init_all = jnp.array(z_init_list)
    z_target_all = jnp.array(z_target_list)

    return z_init_all, z_target_all


def get_training_data(n_samples=5000, n_grid=32, dataset_dir='../data'):
    """
    Get training data with smart fallback logic.

    Strategy:
    1. Check for pre-generated dataset matching n_grid
    2. If not found, generate at specified n_grid (default 32)
    3. Return data ready for training

    Args:
        n_samples: Number of samples (used if generating)
        n_grid: Grid resolution (32 or 64)
        dataset_dir: Directory to look for/save datasets

    Returns:
        z_init_all: (n_samples, n_grid, n_grid)
        z_target_all: (n_samples, n_grid, n_grid)
        n_grid: Actual grid size used
    """
    dataset_path = Path(dataset_dir) / f'heat2d_dataset_{n_grid}x{n_grid}.npz'

    if dataset_path.exists():
        # Load pre-generated dataset
        z_init_all, z_target_all, loaded_n_grid = load_dataset(dataset_path)

        # Verify grid size matches
        if loaded_n_grid != n_grid:
            print(f"WARNING: Requested n_grid={n_grid} but loaded dataset has {loaded_n_grid}")
            n_grid = loaded_n_grid

        # Verify sample count
        if len(z_init_all) < n_samples:
            print(f"WARNING: Dataset has {len(z_init_all)} samples, requested {n_samples}")
            print(f"Using all {len(z_init_all)} available samples")
        elif len(z_init_all) > n_samples:
            print(f"Using first {n_samples} samples from dataset")
            z_init_all = z_init_all[:n_samples]
            z_target_all = z_target_all[:n_samples]

        return z_init_all, z_target_all, n_grid

    else:
        # Generate new dataset
        print(f"Pre-generated dataset not found at {dataset_path}")
        print(f"Generating {n_samples} samples at {n_grid}×{n_grid} resolution...")
        print(f"(This will take ~{0.5*n_samples*(n_grid/32)**3:.0f} seconds)")
        print()

        z_init_all, z_target_all = generate_dataset(
            n_samples=n_samples,
            n_points=n_grid,
            verbose=True
        )

        # Optionally save for future use
        print()
        print(f"Dataset generated successfully!")
        print(f"To avoid regeneration, save with:")
        print(f"  np.savez_compressed('{dataset_path}',")
        print(f"                      z_init=z_init_all,")
        print(f"                      z_target=z_target_all,")
        print(f"                      grid_size={n_grid},")
        print(f"                      n_samples={n_samples})")
        print()

        return z_init_all, z_target_all, n_grid


if __name__ == "__main__":
    # Test block to visualize if run directly
    import matplotlib.pyplot as plt

    print("Generating 2D GRF samples...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, k in enumerate(keys):
        xx, yy, field = generate_grf_2d(k, n_points=64, length_scale=0.4)

        # Verify boundary conditions
        max_boundary_error = max(
            jnp.max(jnp.abs(field[0, :])),   # Left edge
            jnp.max(jnp.abs(field[-1, :])),  # Right edge
            jnp.max(jnp.abs(field[:, 0])),   # Bottom edge
            jnp.max(jnp.abs(field[:, -1]))   # Top edge
        )
        print(f"Sample {i+1}: Max boundary error = {max_boundary_error:.2e}")

        im = axes[i].imshow(field, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("grf_2d_samples.png")
    print("Saved grf_2d_samples.png")

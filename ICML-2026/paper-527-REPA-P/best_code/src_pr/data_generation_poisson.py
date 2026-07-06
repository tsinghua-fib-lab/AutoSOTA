"""
Poisson equation data generation script.

Governing equation: (-Delta)U = rho
Boundary condition: U|d_Omega = 0 (homogeneous Dirichlet)
Grid: 64x64 (including boundary), h = 1/63
Solver: Discrete Sine Transform (DST)

Usage:
    python src_pr/data_generation_poisson.py --train_samples 2000 --valid_samples 200 --output_dir ./data/poisson
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def dst1(x):
    """Type-I Discrete Sine Transform."""
    x = np.asarray(x)
    n = x.shape[-1]
    y = np.zeros(x.shape[:-1] + (2 * (n + 1),), float)
    y[..., 1:n+1] = x
    y[..., n+2:] = -x[..., ::-1]
    Y = np.fft.fft(y, axis=-1)
    return -Y.imag[..., 1:n+1]


def idst1(X):
    """Inverse Type-I Discrete Sine Transform."""
    return dst1(X) / (2 * (X.shape[-1] + 1))


def solve_poisson_dst(rho, h):
    """Solve Poisson equation -Delta U = rho using DST."""
    N = rho.shape[0]
    rho_hat = dst1(dst1(rho.T).T)
    m = np.arange(1, N + 1)
    n = np.arange(1, N + 1)
    lam_m = 2.0 * (1 - np.cos(np.pi * m / (N + 1))) / (h * h)
    lam_n = 2.0 * (1 - np.cos(np.pi * n / (N + 1))) / (h * h)
    lam_2d = lam_m[:, None] + lam_n[None, :]
    U_hat = rho_hat / lam_2d
    return idst1(idst1(U_hat.T).T)


def random_charge(rng, q_min=0.5, q_max=1.5):
    """Generate random charge magnitude with random sign."""
    sign = rng.choice([-1.0, 1.0])
    mag = rng.uniform(q_min, q_max)
    return sign * mag


def generate_single_sample(rng, N_full=64, K=2, q_min=0.5, q_max=1.5):
    """Generate a single (rho, U) sample."""
    N_inner = N_full - 2
    h = 1.0 / (N_full - 1)

    rho_inner_solver = np.zeros((N_inner, N_inner), dtype=np.float64)
    rho_inner_normalized = np.zeros((N_inner, N_inner), dtype=np.float64)

    for _ in range(K):
        q = random_charge(rng, q_min, q_max)
        i = rng.integers(0, N_inner)
        j = rng.integers(0, N_inner)
        rho_inner_solver[i, j] += q / (h * h)
        rho_inner_normalized[i, j] += q

    U_inner = solve_poisson_dst(rho_inner_solver, h)

    rho_full = np.zeros((N_full, N_full), dtype=np.float64)
    rho_full[1:-1, 1:-1] = rho_inner_normalized

    U_full = np.zeros((N_full, N_full), dtype=np.float64)
    U_full[1:-1, 1:-1] = U_inner

    return rho_full, U_full


def generate_dataset(num_samples, N_full=64, K=2, q_min=0.5, q_max=1.5, seed=None):
    """Batch generate dataset."""
    rng = np.random.default_rng(seed)
    rho_list, U_list = [], []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        rho, U = generate_single_sample(rng, N_full, K, q_min, q_max)
        rho_list.append(rho.flatten())
        U_list.append(U.flatten())
    return np.stack(rho_list, axis=0), np.stack(U_list, axis=0)


def save_dataset(rho_data, U_data, output_dir):
    """Save dataset as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, 'rho_data.csv'), rho_data, delimiter=',', fmt='%.10e')
    np.savetxt(os.path.join(output_dir, 'U_data.csv'), U_data, delimiter=',', fmt='%.10e')
    print(f"  rho: {rho_data.shape}, range [{rho_data.min():.4f}, {rho_data.max():.4f}]")
    print(f"  U:   {U_data.shape}, range [{U_data.min():.4f}, {U_data.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Generate Poisson equation dataset")
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--valid_samples", type=int, default=2048)
    parser.add_argument("--N", type=int, default=64, help="Grid size including boundary")
    parser.add_argument("--K", type=int, default=2, help="Number of point charges per sample")
    parser.add_argument("--q_min", type=float, default=0.5)
    parser.add_argument("--q_max", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default="./data/poisson")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Poisson data generation")
    print(f"  Grid: {args.N}x{args.N}, charges: {args.K}, q_range: [{args.q_min}, {args.q_max}]")
    print(f"  Train: {args.train_samples}, Valid: {args.valid_samples}")
    print(f"  Output: {args.output_dir}")

    print("\n=== Train ===")
    rho_train, U_train = generate_dataset(args.train_samples, args.N, args.K,
                                           args.q_min, args.q_max, seed=args.seed)
    save_dataset(rho_train, U_train, os.path.join(args.output_dir, 'train'))

    print("\n=== Valid ===")
    rho_valid, U_valid = generate_dataset(args.valid_samples, args.N, args.K,
                                           args.q_min, args.q_max, seed=args.seed + 1000)
    save_dataset(rho_valid, U_valid, os.path.join(args.output_dir, 'valid'))

    print("\nDone.")


if __name__ == "__main__":
    main()

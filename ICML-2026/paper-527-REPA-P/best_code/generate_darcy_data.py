#!/usr/bin/env python3
"""Generate Darcy Flow dataset matching the paper's data generation process.

Uses KLE expansion for permeability fields and solves the Darcy equation
-via finite differences, identical to the ETHZ dataset generation pipeline.
"""

import os
import sys
import time
import itertools
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm
from scipy.linalg import eigh, lstsq
from findiff import FinDiff, Coef


def uniform_points_pixelwise(n, domain_length, boundary=False, dim=2):
    xi = []
    for i in range(dim):
        pixel_size = domain_length / n
        if boundary:
            start = 0
            end = domain_length
        else:
            start = pixel_size / 2
            end = domain_length - pixel_size / 2
        xi.append(np.linspace(start, end, num=n))
    x = np.array(list(itertools.product(*xi)))
    return x


def create_f_s(x, y, w=0.125, r=10.0):
    condition1 = np.abs(x - 0.5 * w) <= 0.5 * w
    condition2 = np.abs(x - 1 + 0.5 * w) <= 0.5 * w
    condition3 = np.abs(y - 0.5 * w) <= 0.5 * w
    condition4 = np.abs(y - 1 + 0.5 * w) <= 0.5 * w
    result = np.zeros_like(x)
    result[np.logical_and(condition1, condition3)] = r
    result[np.logical_and(condition2, condition4)] = -r
    return result


def complete_covariance_matrix(grid, l):
    dx = grid[:, None, 0] - grid[None, :, 0]
    dy = grid[:, None, 1] - grid[None, :, 1]
    distances_squared = dx**2 + dy**2
    covariance_matrix = np.exp(-np.sqrt(distances_squared) / l)
    return covariance_matrix


def compute_eigenpairs(cov_matrix, q):
    eigenvalues, eigenvectors = eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues[:q], eigenvectors[:, :q]


def KLE_expansion(eigenvalues, eigenvectors, q, grid_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    z = norm.rvs(size=q)
    G_s = np.zeros(grid_points)
    for k in range(q):
        G_s += np.sqrt(eigenvalues[k]) * z[k] * eigenvectors[:, k]
    return G_s, z


def create_boundary_idcs(shape):
    xmin_bd = np.zeros(shape, dtype=bool)
    xmin_bd[0, :] = 1
    xmin_bd = xmin_bd.reshape(-1)
    xmax_bd = np.zeros(shape, dtype=bool)
    xmax_bd[-1, :] = 1
    xmax_bd = xmax_bd.reshape(-1)
    ymin_bd = np.zeros(shape, dtype=bool)
    ymin_bd[:, 0] = 1
    ymin_bd = ymin_bd.reshape(-1)
    ymax_bd = np.zeros(shape, dtype=bool)
    ymax_bd[:, -1] = 1
    ymax_bd = ymax_bd.reshape(-1)
    return xmin_bd, xmax_bd, ymin_bd, ymax_bd


def create_int_cond(use_trapezoid, shape, d0):
    if use_trapezoid:
        int_cond = np.zeros(shape)
        int_cond[0, 0] = 1.0
        int_cond[0, -1] = 1.0
        int_cond[-1, 0] = 1.0
        int_cond[-1, -1] = 1.0
        int_cond[1:-1, 0] = 2.0
        int_cond[1:-1, -1] = 2.0
        int_cond[0, 1:-1] = 2.0
        int_cond[-1, 1:-1] = 2.0
        int_cond[1:-1, 1:-1] = 4.0
        assert np.all(int_cond != 0)
        int_cond *= d0**2 / 4.0
    else:
        pixels_per_dim = shape[0]
        int_cond = np.ones(shape).reshape(-1, 1) / (pixels_per_dim**2)
    return int_cond


def generate_sample(args):
    (i, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
     acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd,
     reverse_dy) = args

    pid = os.getpid()
    current_time = int(time.time() * 1000)
    unique_seed = (pid * current_time + i * 7919) % (2**32)

    log_permeability_field, z = KLE_expansion(
        eigenvalues, eigenvectors, q, pixels_per_dim**2, seed=unique_seed
    )
    K = np.exp(log_permeability_field.reshape(shape))

    K_d0 = FinDiff(0, d0, 1, acc=acc)(K)
    K_d1 = FinDiff(1, d1, 1, acc=acc)(K)

    darcy_fd = (
        Coef(-K) * FinDiff(0, d0, 2, acc=acc)
        - Coef(K_d0) * FinDiff(0, d0, 1, acc=acc)
        - Coef(K) * FinDiff(1, d1, 2, acc=acc)
        - Coef(K_d1) * FinDiff(1, d1, 1, acc=acc)
    )
    grad_p_d0 = FinDiff(0, d0, 1, acc=acc)
    grad_p_d1 = FinDiff(1, d1, 1, acc=acc)

    A = darcy_fd.matrix(shape).toarray()
    b = f_s.reshape(-1, 1)

    grad_p_d0_np = grad_p_d0.matrix(shape).toarray()
    grad_p_d1_np = grad_p_d1.matrix(shape).toarray()

    if reverse_dy:
        A_bc = np.concatenate(
            (A, -grad_p_d0_np[xmin_bd, :], grad_p_d0_np[xmax_bd, :],
             grad_p_d1_np[ymin_bd, :], -grad_p_d1_np[ymax_bd, :]), axis=0
        )
    else:
        A_bc = np.concatenate(
            (A, -grad_p_d0_np[xmin_bd, :], grad_p_d0_np[xmax_bd, :],
             -grad_p_d1_np[ymin_bd, :], grad_p_d1_np[ymax_bd, :]), axis=0
        )
    b_bc = np.concatenate(
        (b, np.zeros(len(A_bc) - len(A)).reshape(-1, 1)), axis=0
    )

    A_bc_int = np.concatenate((A_bc, int_cond.reshape(1, -1)), axis=0)
    b_bc_int = np.concatenate((b_bc, np.array([0.0]).reshape(1, 1)), axis=0)

    p, residuals, _, _ = lstsq(A_bc_int, b_bc_int)

    residual_test = A_bc_int @ p.reshape(-1) - b_bc_int.reshape(-1)

    return (
        K.flatten(), p.flatten(),
        float(np.abs(residual_test).mean()), unique_seed
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=2000)
    parser.add_argument('--valid_samples', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='./data/darcy')
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()

    start_time = time.time()

    pixels_per_dim = 64
    pixels_at_boundary = True
    domain_length = 1.0
    length_scale = 0.1
    q = 64
    acc = 2
    reverse_dy = True

    shape = (pixels_per_dim, pixels_per_dim)
    evaluation_points = uniform_points_pixelwise(
        pixels_per_dim, domain_length, pixels_at_boundary
    )

    if pixels_at_boundary:
        d0 = domain_length / (pixels_per_dim - 1)
        d1 = domain_length / (pixels_per_dim - 1)
    else:
        d0 = domain_length / pixels_per_dim
        d1 = domain_length / pixels_per_dim

    if reverse_dy:
        d1 *= -1.0

    print("Computing covariance matrix and eigenpairs...")
    cov_matrix = complete_covariance_matrix(evaluation_points, length_scale)
    eigenvalues, eigenvectors = compute_eigenpairs(cov_matrix, q)
    f_s = create_f_s(evaluation_points[:, 0], evaluation_points[:, 1])

    xmin_bd, xmax_bd, ymin_bd, ymax_bd = create_boundary_idcs(shape)

    use_trapezoid = pixels_at_boundary
    int_cond = create_int_cond(use_trapezoid, shape, d0)

    total_samples = args.train_samples + args.valid_samples
    print(f"Generating {total_samples} samples "
          f"({args.train_samples} train + {args.valid_samples} valid)...")

    task_args = [
        (i, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
         acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd,
         reverse_dy)
        for i in range(total_samples)
    ]

    import multiprocessing
    num_workers = args.num_workers or max(1, multiprocessing.cpu_count() - 1)

    all_K = []
    all_p = []
    all_res = []
    all_seed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_sample, ta) for ta in task_args]
        for i, future in enumerate(as_completed(futures)):
            K_flat, p_flat, res, seed = future.result()
            all_K.append(K_flat)
            all_p.append(p_flat)
            all_res.append(res)
            all_seed.append(seed)
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_samples - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total_samples}] "
                      f"rate={rate:.2f} samples/s, ETA={eta:.1f}s")

    gen_time = time.time() - start_time
    print(f"Generation completed in {gen_time:.1f}s")

    # Split into train/valid
    train_K = all_K[:args.train_samples]
    train_p = all_p[:args.train_samples]
    valid_K = all_K[args.train_samples:]
    valid_p = all_p[args.train_samples:]

    # Save
    train_dir = os.path.join(args.output_dir, 'train')
    valid_dir = os.path.join(args.output_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    pd.DataFrame(train_p).to_csv(
        os.path.join(train_dir, 'p_data.csv'), index=False, header=False
    )
    pd.DataFrame(train_K).to_csv(
        os.path.join(train_dir, 'K_data.csv'), index=False, header=False
    )
    pd.DataFrame(valid_p).to_csv(
        os.path.join(valid_dir, 'p_data.csv'), index=False, header=False
    )
    pd.DataFrame(valid_K).to_csv(
        os.path.join(valid_dir, 'K_data.csv'), index=False, header=False
    )

    # Save seeds for reproducibility
    pd.DataFrame(all_seed).to_csv(
        os.path.join(args.output_dir, 'seeds.csv'), index=False, header=False
    )
    pd.DataFrame(all_res).to_csv(
        os.path.join(args.output_dir, 'res_data.csv'), index=False, header=False
    )

    print(f"Data saved to {args.output_dir}/")
    print(f"  train: {args.train_samples} samples")
    print(f"  valid: {args.valid_samples} samples")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

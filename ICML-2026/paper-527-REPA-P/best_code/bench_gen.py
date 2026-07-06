import os, sys, time
sys.path.insert(0, '/repo')
import numpy as np
from generate_darcy_data import (
    uniform_points_pixelwise, complete_covariance_matrix, compute_eigenpairs,
    create_f_s, create_boundary_idcs, create_int_cond, generate_sample
)

pixels_per_dim = 64
pixels_at_boundary = True
domain_length = 1.0
length_scale = 0.1
q = 64; acc = 2; reverse_dy = True
shape = (pixels_per_dim, pixels_per_dim)
evaluation_points = uniform_points_pixelwise(pixels_per_dim, domain_length, pixels_at_boundary)
d0 = domain_length / (pixels_per_dim - 1)
d1 = domain_length / (pixels_per_dim - 1)
if reverse_dy: d1 *= -1.0

print("Computing covariance...", flush=True)
t0 = time.time()
cov_matrix = complete_covariance_matrix(evaluation_points, length_scale)
eigenvalues, eigenvectors = compute_eigenpairs(cov_matrix, q)
f_s = create_f_s(evaluation_points[:, 0], evaluation_points[:, 1])
xmin_bd, xmax_bd, ymin_bd, ymax_bd = create_boundary_idcs(shape)
int_cond = create_int_cond(pixels_at_boundary, shape, d0)
print(f"Covariance done: {time.time()-t0:.1f}s", flush=True)

print("Generating 10 samples...", flush=True)
t0 = time.time()
for i in range(10):
    args = (i, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
            acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd, reverse_dy)
    K_flat, p_flat, res, seed = generate_sample(args)
    if i == 0:
        print(f"  Sample 0: residual={res:.6f}, time={time.time()-t0:.2f}s", flush=True)
elapsed = time.time() - t0
rate = 10 / elapsed
print(f"10 samples in {elapsed:.1f}s ({rate:.1f} samples/s)", flush=True)
print(f"Estimated time for 600 samples: {600/rate:.0f}s = {600/rate/60:.1f}min", flush=True)

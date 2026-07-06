import os, sys
sys.path.insert(0, '/repo')
import numpy as np
import pandas as pd
from generate_darcy_data import (
    uniform_points_pixelwise, complete_covariance_matrix, compute_eigenpairs,
    create_f_s, create_boundary_idcs, create_int_cond, generate_sample
)

pixels_per_dim = 64
pixels_at_boundary = True
domain_length = 1.0
length_scale = 0.1
q = 64
acc = 2
reverse_dy = True

shape = (pixels_per_dim, pixels_per_dim)
evaluation_points = uniform_points_pixelwise(pixels_per_dim, domain_length, pixels_at_boundary)
d0 = domain_length / (pixels_per_dim - 1)
d1 = domain_length / (pixels_per_dim - 1)
if reverse_dy:
    d1 *= -1.0

print("Computing cov matrix...")
cov_matrix = complete_covariance_matrix(evaluation_points, length_scale)
eigenvalues, eigenvectors = compute_eigenpairs(cov_matrix, q)
f_s = create_f_s(evaluation_points[:, 0], evaluation_points[:, 1])
xmin_bd, xmax_bd, ymin_bd, ymax_bd = create_boundary_idcs(shape)
int_cond = create_int_cond(pixels_at_boundary, shape, d0)

print("Testing 1 sample...")
K_flat, p_flat, res, seed = generate_sample(
    0, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
    acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd,
    reverse_dy
)
print(f"K shape: {K_flat.shape}, p shape: {p_flat.shape}, residual: {res}")
print("SUCCESS")

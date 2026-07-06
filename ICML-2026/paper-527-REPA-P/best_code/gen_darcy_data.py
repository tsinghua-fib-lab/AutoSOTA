#!/usr/bin/env python3
"""Serial Darcy Flow data generation (no multiprocessing)."""
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
output_dir = '/repo/data/darcy'
train_samples = 2000
valid_samples = 200

shape = (pixels_per_dim, pixels_per_dim)
evaluation_points = uniform_points_pixelwise(pixels_per_dim, domain_length, pixels_at_boundary)

d0 = domain_length / (pixels_per_dim - 1)
d1 = domain_length / (pixels_per_dim - 1)
if reverse_dy:
    d1 *= -1.0

print("Computing covariance matrix and eigenpairs...")
cov_matrix = complete_covariance_matrix(evaluation_points, length_scale)
eigenvalues, eigenvectors = compute_eigenpairs(cov_matrix, q)
f_s = create_f_s(evaluation_points[:, 0], evaluation_points[:, 1])
xmin_bd, xmax_bd, ymin_bd, ymax_bd = create_boundary_idcs(shape)
int_cond = create_int_cond(pixels_at_boundary, shape, d0)

total_samples = train_samples + valid_samples
print(f"Generating {total_samples} samples ({train_samples} train + {valid_samples} valid)...")

all_K = []
all_p = []
for i in range(total_samples):
    if i % 100 == 0:
        print(f"  Sample {i}/{total_samples}...")
    args = (i, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
            acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd,
            reverse_dy)
    K_flat, p_flat, res, seed = generate_sample(args)
    all_K.append(K_flat)
    all_p.append(p_flat)

print(f"Generated {len(all_K)} samples successfully")

os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)

pd.DataFrame(all_p[:train_samples]).to_csv(os.path.join(output_dir, 'train', 'p_data.csv'), index=False)
pd.DataFrame(all_K[:train_samples]).to_csv(os.path.join(output_dir, 'train', 'K_data.csv'), index=False)
pd.DataFrame(all_p[train_samples:]).to_csv(os.path.join(output_dir, 'valid', 'p_data.csv'), index=False)
pd.DataFrame(all_K[train_samples:]).to_csv(os.path.join(output_dir, 'valid', 'K_data.csv'), index=False)

print("Data saved!")
print(f"Train p: {pd.DataFrame(all_p[:train_samples]).shape}")
print(f"Train K: {pd.DataFrame(all_K[:train_samples]).shape}")

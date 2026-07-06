import sys, os
import numpy as np
import pandas as pd
from generate_darcy_data import (
    uniform_points_pixelwise, complete_covariance_matrix, compute_eigenpairs,
    KLE_expansion, create_f_s, create_boundary_idcs, create_int_cond, generate_sample
)

pixels_per_dim = 64
pixels_at_boundary = True
domain_length = 1.0
length_scale = 0.1
q = 64
acc = 2
reverse_dy = True
output_dir = "./data/darcy"
train_samples = 2000
valid_samples = 200

shape = (pixels_per_dim, pixels_per_dim)
evaluation_points = uniform_points_pixelwise(pixels_per_dim, domain_length, pixels_at_boundary)

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

total_samples = train_samples + valid_samples
print(f"Generating {total_samples} samples serially...")

all_K = []
all_p = []
for i in range(total_samples):
    if i % 100 == 0:
        print(f"  Sample {i}/{total_samples}...")
    try:
        K_flat, p_flat, res, seed = generate_sample(
            i, eigenvalues, eigenvectors, q, pixels_per_dim, shape,
            acc, d0, d1, f_s, int_cond, xmin_bd, xmax_bd, ymin_bd, ymax_bd,
            reverse_dy
        )
        all_K.append(K_flat)
        all_p.append(p_flat)
    except Exception as e:
        print(f"  Error at sample {i}: {e}")
        import traceback
        traceback.print_exc()

print(f"Generated {len(all_K)} samples successfully")

os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
pd.DataFrame(all_p[:train_samples]).to_csv(os.path.join(output_dir, "train", "p_data.csv"), index=False)
pd.DataFrame(all_K[:train_samples]).to_csv(os.path.join(output_dir, "train", "K_data.csv"), index=False)
pd.DataFrame(all_p[train_samples:]).to_csv(os.path.join(output_dir, "valid", "p_data.csv"), index=False)
pd.DataFrame(all_K[train_samples:]).to_csv(os.path.join(output_dir, "valid", "K_data.csv"), index=False)
print("Data saved successfully!")

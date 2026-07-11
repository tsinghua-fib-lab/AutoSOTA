#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

"""
Study D: Optimal alpha for different N
=======================================
For each value of N, sweeps alpha over [N/6, 2N/3] (log-spaced) with
p_e = 1/N and records the optimal alpha that minimises steps to convergence.

Fixed parameters:
    p_w         = 0.5
    batch_size  = 50000

Swept parameters:
    N      in [10, 1000]        (20 values, log-spaced)
    alpha  in [N/6, 2N/3]       (50 values per N, log-spaced)

Derived parameters:
    n_outputs   = 1000 // N (to keep total params constant)
    p_e       = 1/N                       (per theory)
    max_steps = 2 * int(15.5 * N + 20)    (scaled with N, factor 2 safety margin)

Outputs (saved to studies/results/D_alpha_optimal_<timestamp>/):
    n_values.npy               – array of tested N values
    optimal_alpha_per_n.npy    – optimal alpha for each N
    optimal_steps_per_n.npy    – steps at optimal alpha for each N
    N_<N>_alpha_values.npy     – alpha values tested for a given N
    N_<N>_p_diff_matrix.npy    – average distance history matrix for a given N
    N_<N>_steps.npy            – steps to convergence for a given N
    metadata.txt               – summary of parameters
"""

import torch
import numpy as np
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training import create_and_train_model

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"studies/results/D_alpha_optimal_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
n_values = np.logspace(1, 3, 20, dtype=int)  # [10, 1000]
n_outputs = 1_000
batch_size = 50_000
convergence_threshold = 0.01
stagnation_window = 1000
stagnation_threshold = 1e-6
p_w = 0.5
n_alpha_per_n = 50

# Derived parameters (evaluated per N inside the experiment loop)
max_steps_a    = 15.5
max_steps_b    = 20
max_steps_fn   = lambda N: 2 * int(max_steps_a * N + max_steps_b)
p_e_fn         = lambda N: 1.0 / N
alpha_values_fn = lambda N: np.logspace(np.log10(N / 6), np.log10(2 * N / 3), n_alpha_per_n)
n_outputs_fn   = lambda N: n_outputs // N

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"STUDY D: Optimal alpha for different N")
print(f"{'='*80}")
print(f"N range: {n_values[0]} to {n_values[-1]} ({len(n_values)} values)")
print(f"For each N: p_e = 1/N (optimal)")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
results_dict = {}
optimal_alpha_per_n = []
optimal_steps_per_n = []

# ---------------------------------------------------------------------------
# Training loop – outer loop over N, inner loop over alpha
# ---------------------------------------------------------------------------
torch.manual_seed(42)

for n_idx, N in enumerate(n_values):
    print(f"\n{'='*80}")
    print(f"N = {N} ({n_idx+1}/{len(n_values)})")
    print(f"{'='*80}")
    
    # Evaluate derived parameters for this N
    p_e = p_e_fn(N)
    max_steps = max_steps_fn(N)
    alpha_values = alpha_values_fn(N)

    print(f"p_e = 1/N = {p_e:.6f}")
    print(f"max_steps = {max_steps}")
    print(f"alpha range: {alpha_values[0]:.4f} to {alpha_values[-1]:.4f} ({len(alpha_values)} values)")
    
    # Storage for this N
    p_diff_matrix = []
    steps_list = []
    
    for i, alpha in enumerate(alpha_values):
        print(f"  alpha={alpha:.4f} ({i+1}/{len(alpha_values)})", end="")
        
        result = create_and_train_model(
            n_inputs=N,
            n_outputs=n_outputs_fn(N),
            learning_rate=alpha,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            p_e=p_e,
            device=device,
            seed=42,
            verbose=False,
            record_history=True,
            p_w=p_w,
            stagnation_threshold=stagnation_threshold,
            stagnation_window=stagnation_window,    
            batch_size=batch_size,
        )
        
        training_result = result['training_results']
        history = training_result['history']
        
        p_diff_matrix.append(history['p_diff'])
        steps = training_result['steps'] if training_result['converged'] else max_steps
        steps_list.append(steps)
        
        status = "OK" if training_result['converged'] else "NOK"
        print(f" -> {status} steps={steps}")
    
    # Pad and store
    max_len = max(len(h) for h in p_diff_matrix)
    p_diff_matrix_padded = []
    for hist in p_diff_matrix:
        padded = list(hist) + [hist[-1]] * (max_len - len(hist))
        p_diff_matrix_padded.append(padded)
    
    p_diff_matrix_padded = np.array(p_diff_matrix_padded, dtype=float)
    steps_array = np.array(steps_list)
    
    # Find optimal alpha for this N
    best_idx = np.argmin(steps_array)
    optimal_alpha = alpha_values[best_idx]
    optimal_steps = steps_array[best_idx]
    
    optimal_alpha_per_n.append(optimal_alpha)
    optimal_steps_per_n.append(optimal_steps)
    
    print(f"  Optimal: alpha={optimal_alpha:.4f}, steps={optimal_steps}")
    
    # Save results for this N
    results_dict[int(N)] = {
        'alpha_values': alpha_values,
        'p_diff_matrix': p_diff_matrix_padded,
        'steps': steps_array
    }
    
    # Save intermediate results
    np.save(f"{results_dir}/n_values.npy", n_values)
    np.save(f"{results_dir}/optimal_alpha_per_n.npy", np.array(optimal_alpha_per_n))
    np.save(f"{results_dir}/optimal_steps_per_n.npy", np.array(optimal_steps_per_n))
    
    # Save individual N results
    np.save(f"{results_dir}/N_{N}_alpha_values.npy", alpha_values)
    np.save(f"{results_dir}/N_{N}_p_diff_matrix.npy", p_diff_matrix_padded)
    np.save(f"{results_dir}/N_{N}_steps.npy", steps_array)

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
with open(f"{results_dir}/metadata.txt", 'w') as f:
    f.write(f"Study D: Optimal alpha for different N\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"N range: {n_values[0]} to {n_values[-1]}\n")
    f.write(f"Number of N values: {len(n_values)}\n")
    f.write(f"For each N: p_e = 1/N (optimal)\n")
    f.write(f"alpha per N: {n_alpha_per_n} values\n")
    f.write(f"max_steps per N: 2 * int({max_steps_a} * N + {max_steps_b})\n")
    f.write(f"Convergence threshold: {convergence_threshold}\n")
    f.write(f"p_w: {p_w}\n")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")

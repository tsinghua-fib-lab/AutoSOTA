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
Study B: Optimal p_e for different N
=====================================
For each value of N, sweeps p_e over [1/(2N), 10/N] and records the
optimal p_e that minimises steps to convergence.

Fixed parameters:
    alpha       = 0.1
    p_w         = 0.5
    max_steps   = 25000
    batch_size  = 100

Swept parameters:
    N    in [10, 1000]         (50 values, log-spaced)
    p_e  in [1/(2N), 10/N]     (50 values per N, linearly spaced)

Derived paramters:
    n_outputs   = 1000 // N (to keep total params constant)

Outputs (saved to studies/results/B_p_e_optimal_<timestamp>/):
    n_values.npy               – array of tested N values
    optimal_p_e_per_n.npy      – optimal p_e for each N
    optimal_steps_per_n.npy    – steps at optimal p_e for each N
    N_<N>_p_e_values.npy       – p_e values tested for a given N
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
results_dir = f"studies/results/B_p_e_optimal_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
n_values = np.logspace(1, 3, 50, dtype=int)  # [10, 1000]
n_outputs = 1_000
learning_rate = 0.1
max_steps = 25_000
convergence_threshold = 0.01
stagnation_window = 100
stagnation_threshold = 1e-6
p_w = 0.5
batch_size = 100    # default
n_p_e_per_n = 50

# Derived parameters (evaluated per N inside the experiment loop)
p_e_range_fn = lambda N: np.linspace(1.0 / (2 * N), 10.0 / N, n_p_e_per_n)
n_outputs_fn = lambda N: n_outputs // N

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"STUDY B: Optimal p_e for different N")
print(f"{'='*80}")
print(f"N range: {n_values[0]} to {n_values[-1]} ({len(n_values)} values)")
print(f"alpha = {learning_rate}")
print(f"For each N: test {n_p_e_per_n} p_e values in [1/(2N), 10/N]")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
results_dict = {}
optimal_p_e_per_n = []
optimal_steps_per_n = []

# ---------------------------------------------------------------------------
# Training loop – outer loop over N, inner loop over p_e
# ---------------------------------------------------------------------------
torch.manual_seed(42)

for n_idx, N in enumerate(n_values):
    print(f"\n{'='*80}")
    print(f"N = {N} ({n_idx+1}/{len(n_values)})")
    print(f"{'='*80}")
    
    # Evaluate derived parameters for this N
    p_e_range = p_e_range_fn(N)
    
    print(f"p_e range: {p_e_range[0]:.6f} to {p_e_range[-1]:.6f}")
    
    # Storage for this N
    p_diff_matrix = []
    steps_list = []
    
    for i, p_e in enumerate(p_e_range):
        print(f"  p_e={p_e:.6f} ({i+1}/{len(p_e_range)})", end="")
        
        result = create_and_train_model(
            n_inputs=N,
            n_outputs=n_outputs_fn(N),
            learning_rate=learning_rate,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            p_e=p_e,
            batch_size=batch_size,
            device=device,
            seed=42,
            verbose=False,
            record_history=True,
            p_w=p_w,
            stagnation_window=stagnation_window,  
            stagnation_threshold=stagnation_threshold,
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
    
    # Find optimal p_e for this N
    best_idx = np.argmin(steps_array)
    optimal_p_e = p_e_range[best_idx]
    optimal_steps = steps_array[best_idx]
    
    optimal_p_e_per_n.append(optimal_p_e)
    optimal_steps_per_n.append(optimal_steps)
    
    print(f"  Optimal: p_e={optimal_p_e:.6f}, steps={optimal_steps}")
    print(f"  Theory: p_e=1/N={1.0/N:.6f}")
    
    # Save results for this N
    results_dict[int(N)] = {
        'p_e_values': p_e_range,
        'p_diff_matrix': p_diff_matrix_padded,
        'steps': steps_array
    }
    
    # Save intermediate results
    np.save(f"{results_dir}/n_values.npy", n_values)
    np.save(f"{results_dir}/optimal_p_e_per_n.npy", np.array(optimal_p_e_per_n))
    np.save(f"{results_dir}/optimal_steps_per_n.npy", np.array(optimal_steps_per_n))
    
    # Save individual N results
    np.save(f"{results_dir}/N_{N}_p_e_values.npy", p_e_range)
    np.save(f"{results_dir}/N_{N}_p_diff_matrix.npy", p_diff_matrix_padded)
    np.save(f"{results_dir}/N_{N}_steps.npy", steps_array)

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
with open(f"{results_dir}/metadata.txt", 'w') as f:
    f.write(f"Study B: Optimal p_e for different N\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"N range: {n_values[0]} to {n_values[-1]}\n")
    f.write(f"Number of N values: {len(n_values)}\n")
    f.write(f"alpha = {learning_rate}\n")
    f.write(f"p_e per N: {n_p_e_per_n} values\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Max steps: {max_steps}\n")
    f.write(f"Convergence threshold: {convergence_threshold}\n")
    f.write(f"p_w: {p_w}\n")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")

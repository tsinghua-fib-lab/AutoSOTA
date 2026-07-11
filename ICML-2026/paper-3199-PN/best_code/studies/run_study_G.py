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
Study G: Joint impact of alpha, p_e, and batch size for different N
====================================================================
For each value of N, sweeps over discrete sets of alpha, p_e, and batch
size and records convergence.

Fixed parameters:
   n_outputs   = 10
    p_w         = 0.5
    max_steps   = 25_000

Swept parameters:
    N          in logspace(0, 2, 10)            (integer)
    batch_size in {10, 100, 1000, 10000}
    alpha      in {1, N/10}          (per N)
    p_e        in {0.001, 1/N}             (per N)

Outputs (saved to studies/results/G_alpha_p_e_batch_<timestamp>/):
    results_list.npy   – flat list of snapshot dicts {N, batch_size, alpha, p_e, alpha_type, p_e_type, steps, p_diff_history}
    alpha_types.npy    – dict mapping alpha_type index to label string, e.g. {0: "1", 1: "N/10"}
    p_e_types.npy      – dict mapping p_e_type index to label string, e.g. {0: "1/1000", 1: "1/N"}
    n_values.npy       – array of tested N values
    batch_sizes.npy    – array of tested batch sizes
    metadata.txt       – summary of parameters
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
results_dir = f"studies/results/G_alpha_p_e_batch_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
n_values = np.logspace(1, 3, 10, dtype=int)
n_outputs = 1_000
max_steps = 25_000
convergence_threshold = 0.01
stagnation_window = 100
stagnation_threshold = 1e-7
p_w = 0.5
batch_sizes = [10, 100, 1000, 10000]

# Each entry is a (label_string, lambda N: value) pair.
# The label string is used as the human-readable type name in the saved dicts.
alpha_specs = [
    ("1",    lambda N: 1),
    ("N/10", lambda N: N / 10),
]
p_e_specs = [
    ("1/1000", lambda N: 1 / 1000),
    ("1/N",    lambda N: 1 / N),
]
n_outputs_fn = lambda N: n_outputs // N

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"STUDY G: Joint impact of alpha, p_e, and batch size for different N")
print(f"{'='*80}")
print(f"N range: {n_values[0]} to {n_values[-1]} ({len(n_values)} values)")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
results_list = []
# Auto-derive type mapping dicts from specs (index -> label)
alpha_types = {i: label for i, (label, _) in enumerate(alpha_specs)}
p_e_types   = {i: label for i, (label, _) in enumerate(p_e_specs)}

# Save the type dicts and parameter arrays so they are available even if the run is interrupted
np.save(f"{results_dir}/alpha_types.npy",   alpha_types,   allow_pickle=True)
np.save(f"{results_dir}/p_e_types.npy",     p_e_types,     allow_pickle=True)
np.save(f"{results_dir}/batch_sizes.npy",   batch_sizes)

# ---------------------------------------------------------------------------
# Training loop – nested sweep: N x batch_size x alpha x p_e
# ---------------------------------------------------------------------------
torch.manual_seed(42)

for n_idx, N in enumerate(n_values):
    print(f"\n{'='*80}")
    print(f"N = {N} ({n_idx+1}/{len(n_values)})")
    print(f"{'='*80}")
    
    # Evaluate ranges for this N from the spec lambdas
    alpha_range = [(i, label, fn(N)) for i, (label, fn) in enumerate(alpha_specs)]
    p_e_range   = [(i, label, fn(N)) for i, (label, fn) in enumerate(p_e_specs)]
    print(f"p_e range: {[(lbl, v) for _, lbl, v in p_e_range]} - alpha range: {[(lbl, v) for _, lbl, v in alpha_range]}")
    
    for batch_size in batch_sizes:
        print(f"\n Batch size={batch_size}")
        
        for alpha_idx, alpha_label, alpha in alpha_range:
            print(f"-- alpha={alpha:.6f} (type {alpha_idx}: {alpha_label}) --")
            
            for p_e_idx, p_e_label, p_e in p_e_range:
                print(f"  p_e={p_e:.6f} (type {p_e_idx}: {p_e_label})", end="")
                result = create_and_train_model(
                    n_inputs=N,
                    n_outputs=n_outputs_fn(N),
                    learning_rate=alpha,
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
                
                steps = training_result['steps'] if training_result['converged'] else -1
                
                status = "OK" if training_result['converged'] else "NOK"
                print(f" -> {status} steps={steps}")
                
                # Store results as a flat snapshot dict
                results_list.append({
                    'N':          int(N),
                    'batch_size': int(batch_size),
                    'alpha':      float(alpha),
                    'p_e':        float(p_e),
                    'alpha_type': alpha_idx,
                    'p_e_type':   p_e_idx,
                    'steps':      steps,
                    'p_diff_history': history['p_diff'],
                })
        
    # Save intermediate results after each N
    # This allows recovery if interrupted
    np.save(f"{results_dir}/results_list.npy", results_list, allow_pickle=True)
    np.save(f"{results_dir}/n_values.npy", n_values[:n_idx+1])
    print(f"  -> Intermediate save: N={N} completed")

# ---------------------------------------------------------------------------
# Save final results
# ---------------------------------------------------------------------------
print(f"\nSaving final results...")
np.save(f"{results_dir}/results_list.npy",  results_list,  allow_pickle=True)
np.save(f"{results_dir}/alpha_types.npy",   alpha_types,   allow_pickle=True)
np.save(f"{results_dir}/p_e_types.npy",     p_e_types,     allow_pickle=True)
np.save(f"{results_dir}/n_values.npy",      n_values)
np.save(f"{results_dir}/batch_sizes.npy",   batch_sizes)
print(f"Final results saved!")

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
with open(f"{results_dir}/metadata.txt", 'w') as f:
    f.write(f"Study G: Joint impact of alpha, p_e, and batch size for different N\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"N range: {n_values[0]} to {n_values[-1]}\n")
    f.write(f"Number of N values: {len(n_values)}\n")
    f.write(f"Batch sizes: {batch_sizes[0]} to {batch_sizes[-1]}\n")
    f.write(f"Number of batch sizes: {len(batch_sizes)}\n")
    f.write(f"alpha_range = {alpha_specs}\n")
    f.write(f"p_e range: {p_e_specs}\n")
    f.write(f"Max steps: {max_steps}\n")
    f.write(f"Convergence threshold: {convergence_threshold}\n")
    f.write(f"p_w: {p_w}\n")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")

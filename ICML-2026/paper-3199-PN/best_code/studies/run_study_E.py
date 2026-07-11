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
Study E: Impact of p_e on convergence under larger batch size
=============================================================
Same sweep as Study A but with a larger batch size.

Fixed parameters:
    N           = 100
    alpha       = 0.1
    n_outputs   = 1000
    p_w         = 0.5
    batch_size  = 1000
    max_steps   = 10000

Swept parameter:
    p_e in [0.001, 0.1]  (50 values, log-spaced)

Outputs (saved to studies/results/E_p_e_impact_large_batch_<timestamp>/):
    p_e_values.npy             – array of tested p_e values
    p_diff_matrix.npy          – average distance history matrix (n_p_e x max_len), padded
    steps_to_convergence.npy   – steps to convergence per p_e (max_steps if not converged)
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
results_dir = f"studies/results/E_p_e_impact_large_batch_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
n_inputs = 100
n_outputs = 1_000
learning_rate = 0.1
max_steps = 10_000
convergence_threshold = 0.01
stagnation_window = 100
stagnation_threshold = 1e-6
batch_size = 1_000
p_w = 0.5

p_e_values = np.logspace(-3, -1, 50)  # [0.001, 0.1]

print(f"\n{'='*80}")
print(f"STUDY E: Impact of p_e under larger batch size")
print(f"{'='*80}")
print(f"N = {n_inputs}")
print(f"alpha = {learning_rate}")
print(f"p_e range: {p_e_values[0]:.4f} to {p_e_values[-1]:.4f} ({len(p_e_values)} values)")
print(f"batch size = {batch_size}")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
p_diff_matrix = []
steps_to_convergence = []

# ---------------------------------------------------------------------------
# Training loop – sweep over p_e
# ---------------------------------------------------------------------------
torch.manual_seed(42)
for i, p_e in enumerate(p_e_values):
    print(f"Training {i+1}/{len(p_e_values)}: p_e={p_e:.4f}", end="")
    
    result = create_and_train_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=max_steps,
        convergence_threshold=convergence_threshold,
        p_e=p_e,
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
    
    # Store results
    p_diff_matrix.append(history['p_diff'])
    steps = training_result['steps'] if training_result['converged'] else max_steps
    steps_to_convergence.append(steps)
    
    status = "OK" if training_result['converged'] else "NOK"
    print(f" -> {status} steps={steps}, p_diff={training_result['final_p_diff']:.4f}")

# ---------------------------------------------------------------------------
# Pad histories and convert to numpy
# ---------------------------------------------------------------------------
max_len = max(len(h) for h in p_diff_matrix)
p_diff_matrix_padded = []
for hist in p_diff_matrix:
    padded = list(hist) + [hist[-1]] * (max_len - len(hist))
    p_diff_matrix_padded.append(padded)

p_diff_matrix_padded = np.array(p_diff_matrix_padded, dtype=float)
steps_to_convergence = np.array(steps_to_convergence)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.save(f"{results_dir}/p_e_values.npy", p_e_values)
np.save(f"{results_dir}/p_diff_matrix.npy", p_diff_matrix_padded)
np.save(f"{results_dir}/steps_to_convergence.npy", steps_to_convergence)

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
with open(f"{results_dir}/metadata.txt", 'w') as f:
    f.write(f"Study E: Impact of p_e (large batch size)\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"N = {n_inputs}\n")
    f.write(f"alpha = {learning_rate}\n")
    f.write(f"p_e range: {p_e_values[0]:.4f} to {p_e_values[-1]:.4f}\n")
    f.write(f"Number of p_e values: {len(p_e_values)}\n")
    f.write(f"Max steps: {max_steps}\n")
    f.write(f"Convergence threshold: {convergence_threshold}\n")
    f.write(f"p_w: {p_w}\n")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")

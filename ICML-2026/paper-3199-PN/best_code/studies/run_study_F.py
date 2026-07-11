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
Study F: Distribution of weights during training
==================================================
Records weight statistics at every step for several values of p_w.
Weights are split into two groups based on the oracle target (0 or 1).

Fixed parameters:
    N           = 100_000
    alpha       = 10
    p_e         = 1/N
    n_outputs   = 1
    batch_size  = 1_000
    max_steps   = 1_000_000

Swept parameter:
    p_w in {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}

Outputs (saved to studies/results/F_weight_distribution_<timestamp>/):
    p_w_<val>/weight_stats.npy     – dict with per-step mean/std/min/max (overall & per oracle class)
    p_w_<val>/oracle_weights.npy   – oracle weight vector
    p_w_<val>/weights_step_<k>.npy – full weight snapshot at key steps
    p_w_<val>/metadata.txt         – human-readable summary of parameters
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
results_dir = f"studies/results/F_weight_distribution_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
n_inputs = 100_000
n_outputs = 1
learning_rate = 10
max_steps = 1_000_000
convergence_threshold = 0.01
p_e = 1 / n_inputs
batch_size = 1_000

p_w_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"STUDY F: Weight distribution during training")
print(f"{'='*80}")
print(f"N = {n_inputs}")
print(f"alpha = {learning_rate}")
print(f"p_e = {p_e}")
print(f"Testing p_w values: {p_w_values}")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Training loop – sweep over p_w
# ---------------------------------------------------------------------------
for p_w in p_w_values:
    print(f"\n{'='*80}")
    print(f"Running with p_w = {p_w}")
    print(f"{'='*80}")
    
    # Create subdirectory for this p_w
    p_w_dir = f"{results_dir}/p_w_{p_w:.2f}"
    os.makedirs(p_w_dir, exist_ok=True)
    
    # Run training with weight recording using create_and_train_model
    torch.manual_seed(42)
    result = create_and_train_model(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        p_e=p_e,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=max_steps,
        convergence_threshold=convergence_threshold,
        device=device,
        seed=42,
        verbose=False,
        record_history=True,
        record_weights=True,
        p_w=p_w
    )

    training_result = result['training_results']
    history = training_result['history']
    oracle_weights = training_result['oracle_weights']

    # Extract data from history
    weights_history = history['model_weights']  # List of weight arrays at each recorded step
    p_diff_history = np.array(history['p_diff'])
    steps = np.arange(len(p_diff_history))

    # Compute weight statistics from the recorded weights
    weight_stats = {
        'steps': steps,
        'p_w': p_w,
        'overall': {
            'w_mean': np.array([np.mean(w) for w in weights_history]),
            'w_std': np.array([np.std(w) for w in weights_history]),
            'w_min': np.array([np.min(w) for w in weights_history]),
            'w_max': np.array([np.max(w) for w in weights_history]),
        },
        'weights_associated_to_oracle_1': { # Weights stats at index where oracle weight is 1
            'w_mean': np.array([np.mean(w[oracle_weights == 1]) if np.sum(oracle_weights == 1) > 0 else np.nan for w in weights_history]),
            'w_std': np.array([np.std(w[oracle_weights == 1]) if np.sum(oracle_weights == 1) > 0 else np.nan for w in weights_history]),
            'w_min': np.array([np.min(w[oracle_weights == 1]) if np.sum(oracle_weights == 1) > 0 else np.nan for w in weights_history]),
            'w_max': np.array([np.max(w[oracle_weights == 1]) if np.sum(oracle_weights == 1) > 0 else np.nan for w in weights_history]),
        },
        'weights_associated_to_oracle_0': { # Weights stats at index where oracle weight is 0
            'w_mean': np.array([np.mean(w[oracle_weights == 0]) if np.sum(oracle_weights == 0) > 0 else np.nan for w in weights_history]),
            'w_std': np.array([np.std(w[oracle_weights == 0]) if np.sum(oracle_weights == 0) > 0 else np.nan for w in weights_history]),
            'w_min': np.array([np.min(w[oracle_weights == 0]) if np.sum(oracle_weights == 0) > 0 else np.nan for w in weights_history]),
            'w_max': np.array([np.max(w[oracle_weights == 0]) if np.sum(oracle_weights == 0) > 0 else np.nan for w in weights_history]),
        },
        'p_diff': p_diff_history
    }

    # Save aggregated weight statistics
    np.save(f"{p_w_dir}/weight_stats.npy", weight_stats)

    # Save oracle weights for reference
    np.save(f"{p_w_dir}/oracle_weights.npy", oracle_weights)

    # Save weight snapshots at key steps
    target_steps = [int(v) for v in np.linspace(0, training_result['steps']-1, num=8)]
    snapshot_indices = []

    for target_step in target_steps:
        # Find the closest recorded step to the target
        if target_step <= steps[-1]:
            idx = np.argmin(np.abs(steps - target_step))
            snapshot_indices.append(idx)

    snapshot_indices = sorted(list(set(snapshot_indices)))  # Remove duplicates

    for idx in snapshot_indices:
        step = steps[idx]
        np.save(f"{p_w_dir}/weights_step_{step}.npy", weights_history[idx])

    # Save metadata
    with open(f"{p_w_dir}/metadata.txt", 'w') as f:
        f.write(f"Study F: Weight distribution during training\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"N = {n_inputs}\n")
        f.write(f"n_outputs = {n_outputs}\n")
        f.write(f"alpha = {learning_rate}\n")
        f.write(f"p_e = {p_e}\n")
        f.write(f"p_w = {p_w}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Max steps: {max_steps}\n")
        f.write(f"Convergence threshold: {convergence_threshold}\n")
        f.write(f"Converged: {training_result['converged']}\n")
        f.write(f"Final step: {training_result['steps']}\n")
        f.write(f"Final p_diff: {training_result['final_p_diff']:.6f}\n")
        f.write(f"Final loss: {training_result['final_loss']:.6f}\n")
        f.write(f"Recorded steps: {len(weights_history)}\n")

    print(f"  Converged: {training_result['converged']}")
    print(f"  Steps: {training_result['steps']}")
    print(f"  Recorded {len(weights_history)} weight snapshots")
    print(f"  Saved {len(snapshot_indices)} full weight arrays at key steps")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")


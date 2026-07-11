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
Study H: Product Node vs MLP — Sparse vs Full-Table Training
=============================================================
Compares a product node and a standard MLP under two training regimes:

  (A) SPARSE: inputs generated via a Bernouilli sampling (p_e=1/N) — very few
      bits are active per example.  The product node should generalise
      perfectly thanks to its multiplicative inductive bias; 

  (B) FULL: inputs sampled uniformly from the complete truth table
      (2^N rows).  

Both models use the same oracle, the same optimizer (SGD) with the
same learning rate, and are evaluated on the full truth table.
We also track the fraction of unique codewords seen during training.

Fixed parameters:
    N               = 16            # input size (not too large to allow full-table evaluation; 2^16=65 536 rows)
    n_outputs       = 250           # number of parallel XOR/MLP outputs
    p_e             = 1.0 / N       # (sparse inputs prob.)
    p_w             = 0.5           # oracle weight density
    batch_size      = 100
    max_steps       = 1_000_000
    points_per_decade = 5           # log-spaced eval schedule: ~N points per decade (e.g. 3 -> 1, 4, 7, 10, 40, 70, …)
    n_runs          = 1             # independent seeds / oracles
    lr_sparse       = 0.02          # SGD learning rate — same for both models
    lr_full         = lr_sparse             # SGD learning rate — same for both models
    mlp_hidden      = (512, 512, 64, 1)     # MLP hidden layer sizes (From Abbe et al. 2023)
    chunk_size      = 4096          # chunk size for processing the full truth table

Outputs (saved to studies/results/H_product_vs_mlp_<timestamp>/):
    sparse_product_train_loss.npy   – (n_runs, max_steps) MSE loss
    sparse_mlp_train_loss.npy
    full_product_train_loss.npy
    full_mlp_train_loss.npy
    sparse_product_train_acc.npy    – (n_runs, n_evals) % correct on seen codewords
    sparse_mlp_train_acc.npy
    full_product_train_acc.npy
    full_mlp_train_acc.npy
    sparse_product_val_acc.npy      – (n_runs, n_evals) % correct on full truth table
    sparse_mlp_val_acc.npy
    full_product_val_acc.npy
    full_mlp_val_acc.npy
    sparse_coverage.npy             – (n_runs, n_evals) raw fraction of 2^N inputs seen
    full_coverage.npy
    oracle_weights_all.npy          – (n_runs, N, n_outputs) oracle weight matrix per run
    eval_steps.npy                  – (n_evals,)
    metadata.txt
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.layers.product import MultiBinaryProductLayer
from models.mlp import ParallelMLP

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
results_dir = f"studies/results/H_product_vs_mlp_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory: {results_dir}/")

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
N               = 16         # input size (not too large to allow full-table evaluation; 2^16=65 536 rows)
n_outputs       = 250        # number of parallel XOR/MLP outputs
p_e             = 1.0 / N    # (sparse inputs prob.)
p_w             = 0.5        # oracle weight density
batch_size      = 100
max_steps       = 1_000_000
points_per_decade = 5        # log-spaced eval schedule: ~N points per decade (e.g. 3 -> 1, 4, 7, 10, 40, 70, …)
n_runs          = 1         # independent seeds / oracles
lr_sparse       = 0.02      # SGD learning rate — same for both models
lr_full         = lr_sparse      # SGD learning rate — same for both models
mlp_hidden      = (512, 512, 64, 1)  # MLP hidden layer sizes (From Abbe et al. 2023)
chunk_size      = 4096  # chunk size for processing the full truth table

truth_table_size = 2 ** N
# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print(f"STUDY H: Product Node vs MLP — Sparse vs Full-Table Training")
print(f"{'='*80}")
print(f"N={N}  truth_table_size={truth_table_size}  n_outputs={n_outputs}")
print(f"p_e=1/N={p_e:.4f}  p_w={p_w}")
print(f"batch_size={batch_size}  max_steps={max_steps}  eval=log-spaced (~{points_per_decade} pts/decade)")
print(f"n_runs={n_runs}  sparse model: SGD lr={lr_sparse}  full model: SGD lr={lr_full}  MLP hidden={mlp_hidden}")
print(f"{'='*80}\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_truth_table(n_bits, dev):
    """Return all 2^n_bits binary vectors as a float tensor."""
    indices = torch.arange(2 ** n_bits, dtype=torch.int64)
    bits = ((indices.unsqueeze(1) >> torch.arange(n_bits - 1, -1, -1)) & 1).float()
    return bits.to(dev)

def generate_sparse_batch(bs, n_bits, pe, dev):
    """Each bit is 1 with probability pe (BSC on the all-zeros vector)."""
    return (torch.rand(bs, n_bits, device=dev) < pe).float()

def make_log_eval_steps(max_steps, points_per_decade):
    """Generate log-spaced evaluation steps with ~points_per_decade steps per decade.
    
    E.g. points_per_decade=3 -> steps like 1, 4, 7, 10, 40, 70, 100, 400, 700, …
    """
    n_points = int(np.ceil(np.log10(max_steps) * points_per_decade)) + 1
    steps = np.unique(np.round(np.logspace(0, np.log10(max_steps), n_points)).astype(int))
    return steps[(steps >= 1) & (steps <= max_steps)]

@torch.no_grad()
def count_errors(model, tt, tt_labels, n_outputs, chunk=4096):
    """Mean binary errors per XOR output unit on the full truth table.
    
    Returns:
        errors: average number of mismatches per independent output
        accuracy: percentage of correctly classified codewords (0-100)
    """
    preds = torch.cat([model(tt[i:i+chunk]) for i in range(0, len(tt), chunk)])
    preds = preds.reshape(preds.shape[0], n_outputs)  # reshape if needed (e.g. MLP with output shape [B, n_outputs, 1])
    total_errors = ((preds > 0.5).float() != tt_labels).float().sum().item()
    errors = total_errors / tt_labels.shape[1]   # average over n_outputs
    accuracy = 100.0 * (1.0 - errors / len(tt))
    return errors, accuracy


def make_product_model(dev):
    m = MultiBinaryProductLayer(
        n_outputs=n_outputs,  
        hard_step=False,
    ).to(dev)
    _ = m(torch.zeros(1, N, device=dev))   # trigger lazy weight init
    return m


def make_mlp_model(dev):
    m = ParallelMLP(
        n_inputs=N, 
        n_outputs=n_outputs,
        hidden_sizes=mlp_hidden,
        activation=nn.ReLU(),
        output_activation=None,
    ).to(dev)
    _ = m(torch.zeros(1, N, device=dev))   # trigger lazy weight init
    return m

def make_oracle_model(dev):
    m = MultiBinaryProductLayer(
            n_outputs=n_outputs,
            hard_step=True
        ).to(dev)
    _ = m(torch.zeros(1, N, device=dev))   # trigger lazy weight init
    return m
# ---------------------------------------------------------------------------
# Pre-compute truth table (shared across all runs)
# ---------------------------------------------------------------------------
truth_table = generate_truth_table(N, device)
print(f"Truth table shape: {truth_table.shape}")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
eval_steps     = make_log_eval_steps(max_steps, points_per_decade)
eval_steps_set = set(eval_steps.tolist())
n_evals        = len(eval_steps)

# Training loss  – shape (n_runs, max_steps)
sparse_product_train_loss = np.zeros((n_runs, max_steps))
sparse_mlp_train_loss     = np.zeros((n_runs, max_steps))
full_product_train_loss   = np.zeros((n_runs, max_steps))
full_mlp_train_loss       = np.zeros((n_runs, max_steps))

# Training accuracy on seen codewords – shape (n_runs, n_evals)
sparse_product_train_acc = np.zeros((n_runs, n_evals))
sparse_mlp_train_acc     = np.zeros((n_runs, n_evals))
full_product_train_acc   = np.zeros((n_runs, n_evals))
full_mlp_train_acc       = np.zeros((n_runs, n_evals))

# Validation accuracy on full truth table – shape (n_runs, n_evals)
sparse_product_val_acc = np.zeros((n_runs, n_evals))
sparse_mlp_val_acc     = np.zeros((n_runs, n_evals))
full_product_val_acc   = np.zeros((n_runs, n_evals))
full_mlp_val_acc       = np.zeros((n_runs, n_evals))

# Coverage: fraction of unique codewords seen – (n_runs, n_evals)
sparse_coverage = np.zeros((n_runs, n_evals))
full_coverage   = np.zeros((n_runs, n_evals))

# Pre-computed bit-weights for binary-vector -> integer index conversion
powers_np = (2 ** np.arange(N - 1, -1, -1)).astype(np.int64)

# Loss: Sum over outputs (independent XOR nodes), mean over batch.
# This ensures training dynamics are independent of n_outputs P:
# standard MSELoss averages over B×P, which would scale gradients by 1/P.
loss_fn = lambda pred, target: ((pred - target) ** 2).sum(dim=1).mean()

# Oracle weight matrix for each run (allows any post-hoc parity analysis)
oracle_weights_all     = np.zeros((n_runs, N, n_outputs))

# Track number of completed runs (for proper checkpoint handling)
completed_runs = 0

# ---------------------------------------------------------------------------
# Checkpoint saving function (incremental save after each run)
# ---------------------------------------------------------------------------
def save_checkpoint(completed_count):
    """Save all results arrays to disk. Ensures data is not lost on interruption.
    
    Args:
        completed_count: Number of runs that have been fully completed.
                        We save only the relevant rows to avoid zero-padding issues.
    """
    try:
        # Save only the completed runs (first completed_count rows)
        # Training loss (MSE)
        np.save(f"{results_dir}/sparse_product_train_loss.npy", sparse_product_train_loss[:completed_count])
        np.save(f"{results_dir}/sparse_mlp_train_loss.npy",     sparse_mlp_train_loss[:completed_count])
        np.save(f"{results_dir}/full_product_train_loss.npy",   full_product_train_loss[:completed_count])
        np.save(f"{results_dir}/full_mlp_train_loss.npy",       full_mlp_train_loss[:completed_count])

        # Training accuracy on seen codewords
        np.save(f"{results_dir}/sparse_product_train_acc.npy", sparse_product_train_acc[:completed_count])
        np.save(f"{results_dir}/sparse_mlp_train_acc.npy",     sparse_mlp_train_acc[:completed_count])
        np.save(f"{results_dir}/full_product_train_acc.npy",   full_product_train_acc[:completed_count])
        np.save(f"{results_dir}/full_mlp_train_acc.npy",       full_mlp_train_acc[:completed_count])

        # Validation accuracy on full truth table
        np.save(f"{results_dir}/sparse_product_val_acc.npy", sparse_product_val_acc[:completed_count])
        np.save(f"{results_dir}/sparse_mlp_val_acc.npy",     sparse_mlp_val_acc[:completed_count])
        np.save(f"{results_dir}/full_product_val_acc.npy",   full_product_val_acc[:completed_count])
        np.save(f"{results_dir}/full_mlp_val_acc.npy",       full_mlp_val_acc[:completed_count])

        # Raw coverage
        np.save(f"{results_dir}/sparse_coverage.npy", sparse_coverage[:completed_count])
        np.save(f"{results_dir}/full_coverage.npy",   full_coverage[:completed_count])

        np.save(f"{results_dir}/oracle_weights_all.npy",   oracle_weights_all[:completed_count])
        np.save(f"{results_dir}/eval_steps.npy",           eval_steps)
        
        # Save metadata about completion
        np.save(f"{results_dir}/completed_runs.npy", np.array(completed_count))
    except Exception as e:
        print(f"\n ERROR during checkpoint save: {e}")
        print(f"Results may be incomplete. Check {results_dir}/ for any partial saves.")
        raise

# ---------------------------------------------------------------------------
# Main loop over runs
# ---------------------------------------------------------------------------
study_start_time = time.time()
total_steps = n_runs * max_steps

for run in range(n_runs):
    seed = run * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Run {run+1}/{n_runs}  (seed={seed}) ---")

    # ---- Oracle (same for both training regimes) -------------------------
    oracle_w = (torch.rand(N, n_outputs, device=device) < p_w).float()
    oracle_model = make_oracle_model(device)
    oracle_model.product_weights.data.copy_(oracle_w)
    tt_labels = torch.cat([                              # pre-computed labels (chunked)
        oracle_model(truth_table[i:i+chunk_size])
        for i in range(0, len(truth_table), chunk_size)
    ], dim=0)
    oracle_weights_all[run] = oracle_w.cpu().numpy()
    n_ones = tt_labels.sum(dim=0).tolist()
    print(f"  Oracle Weights: {oracle_w.shape} - Truth Table Verification: min/max ones = ({min(n_ones):.0f};{max(n_ones):.0f}) / {truth_table_size}")

    # ---- Models: 4 independent instances ---------------------------------
    prod_sp = make_product_model(device)
    mlp_sp  = make_mlp_model(device)
    prod_fu = make_product_model(device)
    mlp_fu  = make_mlp_model(device)

    opt_prod_sp = torch.optim.SGD(prod_sp.parameters(), lr=lr_sparse)
    opt_mlp_sp  = torch.optim.SGD(mlp_sp.parameters(),  lr=lr_sparse)
    opt_prod_fu = torch.optim.SGD(prod_fu.parameters(), lr=lr_full)
    opt_mlp_fu  = torch.optim.SGD(mlp_fu.parameters(),  lr=lr_full)

    # ---- Coverage trackers (boolean arrays of 2^N entries, CPU) ----------
    sparse_seen = np.zeros(truth_table_size, dtype=bool)
    full_seen   = np.zeros(truth_table_size, dtype=bool)

    # ---- Training loop ---------------------------------------------------
    eval_idx = 0
    sp_cov = fu_cov = 0.0   # updated at each eval; used in FINAL summary

    for step in range(1, max_steps + 1):

        # -- Sparse batch --
        x_sp = generate_sparse_batch(batch_size, N, p_e, device)
        y_sp = oracle_model(x_sp).detach()
        idx_sp_np = (x_sp.cpu().numpy() @ powers_np).astype(int)
        sparse_seen[idx_sp_np] = True   # update sparse coverage

        # -- Full batch (uniform sample from truth table) --
        idx_fu    = torch.randint(0, truth_table_size, (batch_size,), device=device)
        idx_fu_np = idx_fu.cpu().numpy()          # cache once for all coverage tracking
        x_fu      = truth_table[idx_fu]
        y_fu      = tt_labels[idx_fu].detach()
        full_seen[idx_fu_np] = True               # update raw full coverage

        # -- Gradient steps for all 4 models --
        def sgd_step(model, opt, x, y):
            opt.zero_grad()
            
            # Model output - Reshaped to (B, n_outputs) 
            # if needed (e.g. for MLP with output shape (B, n_outputs, 1)
            x = model(x)
            x = x.reshape(x.shape[0], n_outputs)
            loss = loss_fn(x, y)
            loss.backward()
            opt.step()
            return loss.item()

        sparse_product_train_loss[run, step-1] = sgd_step(prod_sp, opt_prod_sp, x_sp, y_sp)
        sparse_mlp_train_loss[run, step-1]     = sgd_step(mlp_sp,  opt_mlp_sp,  x_sp, y_sp)
        full_product_train_loss[run, step-1]   = sgd_step(prod_fu, opt_prod_fu,  x_fu, y_fu)
        full_mlp_train_loss[run, step-1]       = sgd_step(mlp_fu,  opt_mlp_fu,   x_fu, y_fu)

        # -- Periodic evaluation --
        if step in eval_steps_set:
            # Validation accuracy on full truth table
            _, sparse_product_val_acc[run, eval_idx] = count_errors(prod_sp, truth_table, tt_labels, n_outputs, chunk_size)
            _, sparse_mlp_val_acc[run, eval_idx]     = count_errors(mlp_sp,  truth_table, tt_labels, n_outputs, chunk_size)
            _, full_product_val_acc[run, eval_idx]   = count_errors(prod_fu, truth_table, tt_labels, n_outputs, chunk_size)
            _, full_mlp_val_acc[run, eval_idx]       = count_errors(mlp_fu,  truth_table, tt_labels, n_outputs, chunk_size)
            
            # Training accuracy on seen codewords
            sp_idx = np.where(sparse_seen)[0]
            fu_idx = np.where(full_seen)[0]
            
            if len(sp_idx) > 0:
                sp_tt = truth_table[sp_idx]
                sp_labels = tt_labels[sp_idx]
                _, sparse_product_train_acc[run, eval_idx] = count_errors(prod_sp, sp_tt, sp_labels, n_outputs, chunk_size)
                _, sparse_mlp_train_acc[run, eval_idx]     = count_errors(mlp_sp,  sp_tt, sp_labels, n_outputs, chunk_size)
            
            if len(fu_idx) > 0:
                fu_tt = truth_table[fu_idx]
                fu_labels = tt_labels[fu_idx]
                _, full_product_train_acc[run, eval_idx] = count_errors(prod_fu, fu_tt, fu_labels, n_outputs, chunk_size)
                _, full_mlp_train_acc[run, eval_idx]     = count_errors(mlp_fu,  fu_tt, fu_labels, n_outputs, chunk_size)
            
            # Raw coverage
            sp_cov = sparse_seen.sum() / truth_table_size
            fu_cov = full_seen.sum()   / truth_table_size
            sparse_coverage[run, eval_idx] = sp_cov
            full_coverage[run, eval_idx]   = fu_cov

            eval_idx += 1
            ei = eval_idx - 1

            elapsed    = time.time() - study_start_time
            steps_done = run * max_steps + step
            steps_left = total_steps - steps_done
            eta_sec    = elapsed / steps_done * steps_left if steps_done > 0 else 0
            eta_str    = str(timedelta(seconds=int(eta_sec)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(
                f"  step {step:>6d}  [run {run+1}/{n_runs}]  elapsed={elapsed_str}  ETA={eta_str}\n"
                f"sparse[prod=(loss: {sparse_product_train_loss[run,step-1]:.6f} - train/val.: {sparse_product_train_acc[run,ei]:5.1f}/{sparse_product_val_acc[run,ei]:5.1f}%) "
                f"mlp=(loss: {sparse_mlp_train_loss[run,step-1]:.6f} - train/val.: {sparse_mlp_train_acc[run,ei]:5.1f}/{sparse_mlp_val_acc[run,ei]:5.1f}%) cov={sp_cov:.4f}\n  "
                f"full[prod=(loss: {full_product_train_loss[run,step-1]:.6f} - train/val.: {full_product_train_acc[run,ei]:5.1f}/{full_product_val_acc[run,ei]:5.1f}%) "
                f"mlp=(loss: {full_mlp_train_loss[run,step-1]:.6f} - train/val.: {full_mlp_train_acc[run,ei]:5.1f}/{full_mlp_val_acc[run,ei]:5.1f}%) cov={fu_cov:.3f}\n" 
            )

    # Final summary
    print(
        f"  FINAL: sparse[prod=(loss: {sparse_product_train_loss[run,-1]:.6f} - train/val.: {sparse_product_train_acc[run,-1]:.1f}/{sparse_product_val_acc[run,-1]:.1f}%) "
        f"mlp=(loss: {sparse_mlp_train_loss[run,-1]:.6f} - train/val.: {sparse_mlp_train_acc[run,-1]:.1f}/{sparse_mlp_val_acc[run,-1]:.1f}%) cov={sp_cov:.4f}\n  "
        f"full[prod=(loss: {full_product_train_loss[run,-1]:.6f} - train/val.: {full_product_train_acc[run,-1]:.1f}/{full_product_val_acc[run,-1]:.1f}%) "
        f"mlp=(loss: {full_mlp_train_loss[run,-1]:.6f} - train/val.: {full_mlp_train_acc[run,-1]:.1f}/{full_mlp_val_acc[run,-1]:.1f}%) cov={fu_cov:.3f}\n"
    )

    del prod_sp, mlp_sp, prod_fu, mlp_fu
    del opt_prod_sp, opt_mlp_sp, opt_prod_fu, opt_mlp_fu
    
    # Increment completed runs counter and save checkpoint
    completed_runs = run + 1
    
    # Save checkpoint BEFORE cache cleanup (safe against device errors)
    try:
        save_checkpoint(completed_runs)
        print(f"  Checkpoint saved after run {run+1}/{n_runs}")
    except Exception as e:
        print(f"  WARNING: Checkpoint save failed after run {run+1}: {e}")
    
    # Clean cache (might fail on some devices, but checkpoint is already safe)
    try:
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  WARNING: Cache cleanup failed (non-critical): {e}")

# ---------------------------------------------------------------------------
# Final metadata
# ---------------------------------------------------------------------------
with open(f"{results_dir}/metadata.txt", "w", encoding="utf-8") as f:
    f.write("Study H: Product Node vs MLP — Sparse vs Full-Table Training\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"N={N}, n_outputs={n_outputs}, p_e=1/N={p_e:.6f}, p_w={p_w}\n")
    f.write(f"batch_size={batch_size}, max_steps={max_steps}, n_evals={n_evals} (log-spaced, {points_per_decade} pts/decade)\n")
    f.write(f"n_runs={n_runs}, lr_sparse={lr_sparse}, lr_full={lr_full}, mlp_hidden={mlp_hidden}\n")

print(f"\n{'='*80}")
print(f"Study completed!")
print(f"Results saved to: {results_dir}/")
print(f"{'='*80}")

#!/usr/bin/env python3
"""
Evaluation script for paper 5683: NNEinFact on Uber dataset.
Produces heldout (alpha,beta)-divergence and runtime metrics.

Usage: python3 eval.py
Output: heldout_loss and runtime printed to stdout.
"""
import numpy as np
import torch
import time
import json
import os
import sys

os.chdir('/repo')
from einfact import NNEinFact

# Configuration
MODEL_STR = 'wr,dr,hr,irk,jkr->wdhij'
K = 5
R = 60
ALPHA = 0.7
BETA = 0.0
N_SPLITS = 10
TRAIN_SPLIT = 0.9
MAX_ITER = 5000
DEVICE = 'cuda:0'
BASE_SEED = 100

# Load data
Y = np.load('data/Y.npz')['Y']
shape_dict = {**dict(zip(MODEL_STR.split('->')[-1], Y.shape)), 'k': K, 'r': R}

heldout_losses = []
runtimes = []

for split_idx in range(N_SPLITS):
    np.random.seed(BASE_SEED + split_idx)
    train_mask = np.random.random(Y.shape) < TRAIN_SPLIT
    
    model = NNEinFact(MODEL_STR, shape_dict=shape_dict, device=DEVICE, alpha=ALPHA, beta=BETA)
    t0 = time.time()
    history = model.fit(Y, max_iter=MAX_ITER, verbose=False, mask=train_mask, early_stopping=True)
    elapsed = time.time() - t0
    
    heldout_losses.append(float(history['heldout_loss'][-1]))
    runtimes.append(elapsed)
    del model
    torch.cuda.empty_cache()

mean_heldout = np.mean(heldout_losses)
mean_runtime = np.mean(runtimes)
std_heldout = np.std(heldout_losses)
std_runtime = np.std(runtimes)

results = {
    'heldout_loss_mean': float(mean_heldout),
    'heldout_loss_std': float(std_heldout),
    'heldout_loss_se': float(std_heldout / np.sqrt(N_SPLITS)),
    'runtime_mean': float(mean_runtime),
    'runtime_std': float(std_runtime),
    'runtime_se': float(std_runtime / np.sqrt(N_SPLITS)),
    'heldout_losses': [float(x) for x in heldout_losses],
    'runtimes': runtimes,
    'settings': {
        'model_str': MODEL_STR, 'k': K, 'r': R,
        'alpha': ALPHA, 'beta': BETA,
        'n_splits': N_SPLITS, 'train_split': TRAIN_SPLIT,
        'max_iter': MAX_ITER, 'device': DEVICE, 'base_seed': BASE_SEED
    }
}

# Print key metrics for parsing
print(f"HELDOUT_LOSS={mean_heldout:.8f}")
print(f"RUNTIME={mean_runtime:.4f}")
print(f"HELDOUT_LOSS_STD={std_heldout:.8f}")
print(f"RUNTIME_STD={std_runtime:.4f}")

# Save full results
with open('eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Full results saved to eval_results.json")

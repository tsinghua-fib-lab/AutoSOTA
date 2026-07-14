#!/usr/bin/env python3
"""Focused grid search around best regions: k=2,3 and r=20-30."""

import numpy as np
import torch
import time
import json
import os

os.chdir('/repo')
from einfact import NNEinFact

MODEL_STR = 'wr,dr,hr,irk,jkr->wdhij'
ALPHA = 0.7
BETA = 0.0
N_SCREEN_SPLITS = 5  # more splits for better estimates
TRAIN_SPLIT = 0.9
MAX_ITER = 5000
DEVICE = 'cuda:0'
BASE_SEED = 42

Y = np.load('data/Y.npz')['Y']

# Focused grid
configs = [
    (2, 20), (2, 22), (2, 25), (2, 28), (2, 30),
    (3, 18), (3, 20), (3, 22), (3, 25), (3, 28),
    (4, 20), (4, 25),
]

results = []

for k, r in configs:
    heldout_losses = []
    runtimes = []

    for split_idx in range(N_SCREEN_SPLITS):
        np.random.seed(BASE_SEED + split_idx)
        train_mask = np.random.random(Y.shape) < TRAIN_SPLIT

        shape_dict = {**dict(zip(MODEL_STR.split('->')[-1], Y.shape)), 'k': k, 'r': r}

        model = NNEinFact(MODEL_STR, shape_dict=shape_dict, device=DEVICE, alpha=ALPHA, beta=BETA)
        t0 = time.time()
        history = model.fit(Y, max_iter=MAX_ITER, verbose=False, mask=train_mask, early_stopping=True)
        elapsed = time.time() - t0

        heldout_losses.append(float(history['heldout_loss'][-1]))
        runtimes.append(elapsed)
        del model
        torch.cuda.empty_cache()

    mean_heldout = np.mean(heldout_losses)
    std_heldout = np.std(heldout_losses)
    mean_runtime = np.mean(runtimes)

    results.append({
        'k': k, 'r': r,
        'heldout_loss': float(mean_heldout),
        'heldout_std': float(std_heldout),
        'runtime': float(mean_runtime),
    })

    print(f"k={k}, r={r}: heldout={mean_heldout:.6f}±{std_heldout:.6f}, runtime={mean_runtime:.2f}s", flush=True)

results.sort(key=lambda x: x['heldout_loss'])
print("\n=== ALL CONFIGS (sorted) ===")
for i, r in enumerate(results):
    marker = " *** BEST" if i == 0 else ""
    print(f"{i+1}. k={r['k']}, r={r['r']}: heldout={r['heldout_loss']:.6f}, runtime={r['runtime']:.2f}s{marker}")

with open('grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

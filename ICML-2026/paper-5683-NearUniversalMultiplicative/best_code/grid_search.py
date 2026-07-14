#!/usr/bin/env python3
"""Grid search over (k, r) for paper 5683. Screen with 3 splits per config."""

import numpy as np
import torch
import time
import json
import os
import sys

os.chdir('/repo')
from einfact import NNEinFact

MODEL_STR = 'wr,dr,hr,irk,jkr->wdhij'
ALPHA = 0.7
BETA = 0.0
N_SCREEN_SPLITS = 3
TRAIN_SPLIT = 0.9
MAX_ITER = 5000
DEVICE = 'cuda:0'
BASE_SEED = 42

Y = np.load('data/Y.npz')['Y']

# Grid
k_values = [2, 3, 4, 5, 6, 8, 10]
r_values = [5, 8, 10, 12, 15, 20, 25]

results = []

for k in k_values:
    for r in r_values:
        heldout_losses = []
        runtimes = []

        for split_idx in range(N_SCREEN_SPLITS):
            np.random.seed(BASE_SEED + split_idx)
            train_mask = np.random.random(Y.shape) < TRAIN_SPLIT

            shape_dict = {
                **dict(zip(MODEL_STR.split('->')[-1], Y.shape)),
                'k': k, 'r': r
            }

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

        results.append({
            'k': k, 'r': r,
            'total_params': k * r * 3 + 27*r + 24*r + 7*r + 100*r*k + 100*k*r,
            'heldout_loss': float(mean_heldout),
            'runtime': float(mean_runtime),
        })

        print(f"k={k}, r={r}: heldout={mean_heldout:.6f}, runtime={mean_runtime:.2f}s", flush=True)

# Sort and show top 5
results.sort(key=lambda x: x['heldout_loss'])
print("\n=== TOP 5 CONFIGS ===")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. k={r['k']}, r={r['r']}: heldout={r['heldout_loss']:.6f}, runtime={r['runtime']:.2f}s")

# Save results
with open('grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nGrid search results saved to grid_search_results.json")

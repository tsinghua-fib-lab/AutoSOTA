#!/usr/bin/env python3
"""Clean screening of (k,r) configs with 3 splits each on CLEAN baseline code."""

import numpy as np
import torch
import time
import os
import sys

os.chdir('/repo')
from einfact import NNEinFact

MODEL_STR = 'wr,dr,hr,irk,jkr->wdhij'
ALPHA = 0.7
BETA = 0.0
N_SPLITS = 3
TRAIN_SPLIT = 0.9
MAX_ITER = 5000
DEVICE = 'cuda:0'
BASE_SEED = 42

Y = np.load('data/Y.npz')['Y']

configs = [
    (6, 10, "baseline"),
    (2, 25, "k2r25"),
    (2, 30, "k2r30"),
    (3, 20, "k3r20"),
    (2, 20, "k2r20"),
    (3, 25, "k3r25"),
    (2, 35, "k2r35"),
]

for k, r, label in configs:
    losses = []
    rts = []

    for split_idx in range(N_SPLITS):
        np.random.seed(BASE_SEED + split_idx)
        train_mask = np.random.random(Y.shape) < TRAIN_SPLIT

        shape_dict = {**dict(zip(MODEL_STR.split('->')[-1], Y.shape)), 'k': k, 'r': r}

        model = NNEinFact(MODEL_STR, shape_dict=shape_dict, device=DEVICE, alpha=ALPHA, beta=BETA)
        t0 = time.time()
        history = model.fit(Y, max_iter=MAX_ITER, verbose=False, mask=train_mask, early_stopping=True)
        elapsed = time.time() - t0

        losses.append(float(history['heldout_loss'][-1]))
        rts.append(elapsed)
        del model
        torch.cuda.empty_cache()

    print(f"{label} (k={k},r={r}): heldout={np.mean(losses):.6f}±{np.std(losses):.6f}, runtime={np.mean(rts):.2f}s", flush=True)

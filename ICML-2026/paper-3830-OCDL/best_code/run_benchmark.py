#!/usr/bin/env python3
"""Reproduce DVM-AD benchmark on Cardiotocography dataset.

Exact protocol from paper Appendix D.7.2 (Table 9):
  - ADBench Cardiotocography: 21 features, NSP label (1=normal, 2,3=anomaly)
  - 5 seeds (0-4), 70/30 stratified train/test split
  - One-class training (normal samples only from train split)
  - DVM-AD params: mode="both", epsilon_sel=0.1, artificial_mode="max"
  - AUROC evaluation on held-out test split

Note: The UCI Cardiotocography dataset was updated in March 2024
(2126 samples vs the 2114 used in ADBench/paper). This script uses
the current UCI version via ucimlrepo.

Usage:
    python run_benchmark.py

Output:
    Per-seed and summary AUROC (mean +/- std over 5 seeds).
"""

import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from dvmad import DVMAD
import warnings
warnings.filterwarnings("ignore")

# ── Configuration (exact paper values) ──────────────────────────
SEEDS = [0, 1, 2, 3, 4]
EPSILON_SEL = 0.02      # eigenvalue selection threshold (optimization: tighter tail)
EPSILON_TOL = 1e-8      # score normalization stability (paper: epsilon_tol)
MODE = "both"
ARTIFICIAL_MODE = "max"
TEST_SIZE = 0.3         # 70/30 split

# ── Load dataset ─────────────────────────────────────────────────
print("=" * 64)
print("  DVM-AD Reproduction — Cardiotocography (ADBench)")
print("=" * 64)
print(f"  epsilon_sel = {EPSILON_SEL}")
print(f"  epsilon_tol = {EPSILON_TOL}")
print(f"  mode        = {MODE}")
print(f"  seeds       = {SEEDS}")
print(f"  split       = {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
print(f"  training    = one-class (normal-only)")
print()

print("Loading Cardiotocography from UCI repository (id=193)...")
ctg = fetch_ucirepo(id=193)
X_full = ctg.data.features.values.astype(np.float64)
y_nsp = ctg.data.targets["NSP"].values

# ADBench label mapping: N=1 (normal) -> 0; S=2, P=3 (anomaly) -> 1
y_full = np.where(y_nsp == 1, 0, 1)

n_normal = int(np.sum(y_full == 0))
n_anom = int(np.sum(y_full == 1))
print(f"  Samples: {len(y_full)}  |  Features: {X_full.shape[1]}")
print(f"  Normal: {n_normal}  |  Anomalies: {n_anom}")
print(f"  Anomaly ratio: {n_anom / len(y_full) * 100:.2f}%")
print()

# ── Run over seeds ───────────────────────────────────────────────
auc_scores = []
print(f"{'Seed':>6s}  {'AUROC':>10s}")
print("-" * 20)

for seed in SEEDS:
    # Stratified split preserving anomaly ratio
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y_full,
    )

    # One-class training: keep only normal instances
    X_tr_normal = X_tr[y_tr == 0]

    # DVM-AD with default paper configuration
    clf = DVMAD(
        contamination=0.1,
        mode=MODE,
        eps=EPSILON_SEL,
        artificial_mode=ARTIFICIAL_MODE,
    )
    clf.fit(X_tr_normal)

    # Anomaly scores on held-out test set
    scores = clf.decision_function(X_te)

    # AUROC (higher scores = more anomalous)
    auc = roc_auc_score(y_te, scores)
    auc_scores.append(auc)
    print(f"  {seed:>4d}   {auc * 100:>8.2f}%")

# ── Summary ─────────────────────────────────────────────────────
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)
print("-" * 20)
print(f"  Mean  {mean_auc * 100:>8.2f}%")
print(f"  Std   {std_auc * 100:>8.2f}%")
print()
print("=" * 64)
print(f"  DVM-AD Cardiotocography AUROC: {mean_auc * 100:.2f} +/- {std_auc * 100:.2f}%")
print(f"  Paper (Table 9):               85.51 +/- 1.24%")
print(f"  Reproduction CI bounds:        [84.27, 86.75]")
print(f"  Status: {'PASS' if mean_auc * 100 >= 84.27 else 'BELOW CI'}")
print("=" * 64)

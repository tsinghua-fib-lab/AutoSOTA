"""Reproduce DVM-AD benchmark on Cardiotocography dataset.

Protocol from Appendix D.7.2:
- 5 seeds (0-4), 70/30 stratified train/test split
- One-class training (normal samples only from train split)
- AUROC evaluation on held-out test split
- DVM-AD with default configuration (mode="both", eps=0.1, artificial_mode="max")
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from dvmad import DVMAD

import warnings
warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
EPSILON_SEL = 0.1
EPSILON_TOL = 1e-8

print("=" * 60)
print("DVM-AD Cardiotocography Benchmark")
print("=" * 60)
print(f"Parameters: epsilon_sel={EPSILON_SEL}, epsilon_tol={EPSILON_TOL}")
print(f"Seeds: {SEEDS}")
print(f"Split: 70/30, training: one-class")
print()

# Load dataset
print("Loading Cardiotocography dataset from UCI...")
cardiotocography = fetch_ucirepo(id=193)
X_full = cardiotocography.data.features.values.astype(np.float64)
y_nsp = cardiotocography.data.targets['NSP'].values

# ADBench mapping: NSP=1 (Normal) -> 0, NSP=2,3 (Suspect/Pathologic) -> 1
y_full = np.where(y_nsp == 1, 0, 1)

print(f"Dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
print(f"Normal: {np.sum(y_full == 0)}, Anomaly: {np.sum(y_full == 1)}")
print(f"Anomaly ratio: {np.sum(y_full == 1) / len(y_full) * 100:.2f}%")
print()

# Run over seeds
auc_scores = []
for seed in SEEDS:
    # Stratified split (maintain anomaly ratio in each split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.3,
        random_state=seed,
        stratify=y_full
    )

    # One-class training: use only normal samples (y==0) from training set
    X_train_normal = X_train[y_train == 0]

    # Fit DVM-AD
    clf = DVMAD(
        contamination=0.1,  # not used for AUROC computation
        mode="both",
        eps=EPSILON_SEL,
        artificial_mode="max",
    )
    clf.fit(X_train_normal)

    # Score test set
    scores = clf.decision_function(X_test)

    # Compute AUROC
    auc = roc_auc_score(y_test, scores)
    auc_scores.append(auc)
    print(f"  Seed {seed}: AUROC = {auc * 100:.2f}%")

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)  # sample std

print()
print("=" * 60)
print(f"DVM-AD Cardiotocography AUROC: {mean_auc * 100:.2f} +/- {std_auc * 100:.2f}%")
print(f"Paper reported: 85.51 +/- 1.24%")
print(f"CI bounds: [84.27, 86.75]")
print("=" * 60)

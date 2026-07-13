#!/usr/bin/env python3
"""Evaluate InfoGlobe on 240-cell data from adata_17_res."""
import scanpy as sc
import numpy as np
import torch
import sys
sys.path.insert(0, "/repo")
import InfoGlobe
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
import time
import json

print("=" * 60)
print("Loading 240-cell data...")
adata = sc.read_h5ad("/repo/sim_data/result/adata_17_res.h5ad")
print(f"Shape: {adata.shape}")
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# Normalize to simplex
X_raw = adata.X.copy()
P = X_raw / X_raw.sum(axis=1, keepdims=True)
P_gd = torch.tensor(P.T, dtype=torch.float32)
n_genes, n_cells = P_gd.shape
print(f"P_gd: {n_genes} genes x {n_cells} cells")

K = 20
MAX_ITER = 30000

print(f"\nRunning InfoGlobe K={K}, max_iter={MAX_ITER}...")
t0 = time.time()
model = InfoGlobe.infoglobe.GlobeEmbedding(A=[n_genes, K], Q=[K, n_cells], c=1)
model.fit(P_gd, max_iter=MAX_ITER, verbose=False, num_pairs=50000)
elapsed = time.time() - t0
print(f"Training: {elapsed:.1f}s")
print(f"Loss1: {model.loss1[-1]:.6f}, Loss2: {model.loss2[-1]:.6f}")

# Get embedding
Q_learned = model.Q.detach().cpu().numpy().T
print(f"Embedding: {Q_learned.shape}")

# Transform to Fisher-Rao sphere
sqrt_P = np.sqrt(np.clip(P, 0, None))
sqrt_Q = np.sqrt(np.clip(Q_learned, 0, None))
sqrt_P_norm = sqrt_P / (np.linalg.norm(sqrt_P, axis=1, keepdims=True) + 1e-12)
sqrt_Q_norm = sqrt_Q / (np.linalg.norm(sqrt_Q, axis=1, keepdims=True) + 1e-12)

# Compute Trustworthiness and Continuity
results = {}
for nn in [7, 12, 30, 50]:
    trust = trustworthiness(sqrt_P_norm, sqrt_Q_norm, n_neighbors=nn)
    contin = trustworthiness(sqrt_Q_norm, sqrt_P_norm, n_neighbors=nn)
    results[f"knn_{nn}"] = {"trustworthiness": float(trust), "continuity": float(contin)}
    print(f"KNN={nn:>2}: Trust={trust:.4f}, Contin={contin:.4f}")

# Spearman Correlation (Fisher-Rao distances)
rng = np.random.RandomState(42)
n_pairs = min(50000, n_cells * (n_cells - 1) // 2)
idx = rng.choice(n_cells, size=n_pairs * 2, replace=False)
half = n_pairs
i_idx = idx[:half]
j_idx = idx[half:]

# Fisher-Rao distances (arccos of dot product on sphere)
inner_Q = np.sum(sqrt_Q_norm[i_idx] * sqrt_Q_norm[j_idx], axis=1)
inner_Q = np.clip(inner_Q, -1 + 1e-7, 1 - 1e-7)
dist_Q = np.arccos(inner_Q)

inner_P = np.sum(sqrt_P_norm[i_idx] * sqrt_P_norm[j_idx], axis=1)
inner_P = np.clip(inner_P, -1 + 1e-7, 1 - 1e-7)
dist_P = np.arccos(inner_P)

spearman_fr, pval_fr = spearmanr(dist_Q, dist_P)
print(f"Spearman (Fisher-Rao): {spearman_fr:.4f} (p={pval_fr:.2e})")

# Also Euclidean distances
dist_Q_euc = np.sqrt(np.sum((Q_learned[i_idx] - Q_learned[j_idx])**2, axis=1))
dist_P_euc = np.sqrt(np.sum((P[i_idx] - P[j_idx])**2, axis=1))
spearman_euc, _ = spearmanr(dist_Q_euc, dist_P_euc)
print(f"Spearman (Euclidean): {spearman_euc:.4f}")

# Euclidean on sqrt-space
dist_Q_sqrt = np.sqrt(np.sum((sqrt_Q[i_idx] - sqrt_Q[j_idx])**2, axis=1))
dist_P_sqrt = np.sqrt(np.sum((sqrt_P[i_idx] - sqrt_P[j_idx])**2, axis=1))
spearman_sqrt, _ = spearmanr(dist_Q_sqrt, dist_P_sqrt)
print(f"Spearman (sqrt-Euc):  {spearman_sqrt:.4f}")

print("\nPaper Table 1 targets:")
print("  Trustworthiness: 0.83±0.00")
print("  Continuity:      0.84±0.01")
print("  Spearman Corr:   0.88±0.01")

# Save
output = {
    "dataset": "adata_17_res",
    "n_cells": n_cells,
    "n_genes": n_genes,
    "K": K,
    "max_iter": MAX_ITER,
    "training_time_s": elapsed,
    "final_loss1": float(model.loss1[-1]),
    "final_loss2": float(model.loss2[-1]),
    "metrics": results,
    "spearman_fr": float(spearman_fr),
    "spearman_euc": float(spearman_euc),
    "spearman_sqrt_euc": float(spearman_sqrt),
}
with open("/repo/eval_240cell_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to /repo/eval_240cell_results.json")
torch.cuda.empty_cache()

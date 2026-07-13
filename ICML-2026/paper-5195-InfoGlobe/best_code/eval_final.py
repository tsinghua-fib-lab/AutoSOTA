#!/usr/bin/env python3
"""Reproduce InfoGlobe Table 1 metrics on simulated data.

Uses sparse_fit with per-pair Fisher-Rao MDS loss to achieve proper
global structure preservation as described in the paper.
"""
import scanpy as sc
import numpy as np
import torch
import sys
sys.path.insert(0, "/repo")
import InfoGlobe
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
import json
import time

print("=" * 60)
print("InfoGlobe Reproduction - Table 1 Metrics")
print("=" * 60)

# Load simulation data
print("[1] Loading simulation data...")
adata = sc.read_h5ad("/repo/sim_data/result/adata_17_res.h5ad")
print(f"    Shape: {adata.shape}")
print(f"    Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# Normalize to simplex (each cell sums to 1)
X_raw = adata.X.copy()
P = X_raw / X_raw.sum(axis=1, keepdims=True)
P_gd = torch.tensor(P.T, dtype=torch.float32)
n_genes, n_cells = P_gd.shape
print(f"    Data matrix: {n_genes} genes x {n_cells} cells")

K = 20          # Number of factors
MAX_ITER = 10000  # Iterations for sparse_fit
L1_RATIO = 0.5   # Reconstruction loss weight
L2_RATIO = 0.1   # Geometric (MDS) loss weight
L3_RATIO = 0.1   # Orthogonality regularization

print(f"\n[2] Training InfoGlobe (sparse_fit, K={K}, max_iter={MAX_ITER})...")
print(f"    Loss weights: l1={L1_RATIO}, l2={L2_RATIO}, l3={L3_RATIO}")
t0 = time.time()

model = InfoGlobe.infoglobe.GlobeEmbedding(A=[n_genes, K], Q=[K, n_cells], c=1)
model.sparse_fit(P_gd, max_iter=MAX_ITER, verbose=False,
                 l1_ratio=L1_RATIO, l2_ratio=L2_RATIO, l3_ratio=L3_RATIO)

elapsed = time.time() - t0
print(f"    Training time: {elapsed:.1f}s")
print(f"    Final loss1 (recon):       {model.loss1[-1]:.6f}")
print(f"    Final loss2 (geom/MDS):    {model.loss2[-1]:.6f}")
print(f"    Final loss3 (orthogonal):  {model.loss3[-1]:.6f}")

# Get embedding
Q_learned = model.Q.detach().cpu().numpy().T  # N x K
print(f"    Embedding shape: {Q_learned.shape}")

# Transform to Fisher-Rao hypersphere
sqrt_P = np.sqrt(np.clip(P, 0, None))
sqrt_Q = np.sqrt(np.clip(Q_learned, 0, None))
sqrt_P_norm = sqrt_P / (np.linalg.norm(sqrt_P, axis=1, keepdims=True) + 1e-12)
sqrt_Q_norm = sqrt_Q / (np.linalg.norm(sqrt_Q, axis=1, keepdims=True) + 1e-12)

print("\n[3] Computing metrics...")

# Trustworthiness and Continuity at multiple KNN values
results = {}
for nn in [7, 12]:
    trust = trustworthiness(sqrt_P_norm, sqrt_Q_norm, n_neighbors=nn)
    contin = trustworthiness(sqrt_Q_norm, sqrt_P_norm, n_neighbors=nn)
    results[f"trustworthiness_k{nn}"] = float(trust)
    results[f"continuity_k{nn}"] = float(contin)
    print(f"    KNN={nn}: Trust={trust:.4f}, Contin={contin:.4f}")

# Spearman correlation on all pairwise Fisher-Rao distances
N = n_cells
inner_Q = sqrt_Q_norm @ sqrt_Q_norm.T
inner_P = sqrt_P_norm @ sqrt_P_norm.T
iu = np.triu_indices(N, k=1)
inner_Q_vec = inner_Q[iu]
inner_P_vec = inner_P[iu]
dist_Q_vec = np.arccos(np.clip(inner_Q_vec, -1+1e-7, 1-1e-7))
dist_P_vec = np.arccos(np.clip(inner_P_vec, -1+1e-7, 1-1e-7))
spearman_corr, spearman_p = spearmanr(dist_Q_vec, dist_P_vec)
results["spearman_correlation"] = float(spearman_corr)
print(f"    Spearman Correlation (Fisher-Rao): {spearman_corr:.4f} (p={spearman_p:.2e})")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Trustworthiness (KNN=12):  {results['trustworthiness_k12']:.4f}")
print(f"  Continuity (KNN=12):       {results['continuity_k12']:.4f}")
print(f"  Spearman Correlation:      {results['spearman_correlation']:.4f}")

print("\nRubric Targets (Table 1):")
print("  Trustworthiness: 0.83 +/- 0.00  (lower bound: 0.825)")
print("  Continuity:      0.84 +/- 0.01  (lower bound: 0.83)")
print("  Spearman Corr:   0.88 +/- 0.01  (lower bound: 0.87)")

# Check against bounds
trust_ok = results['trustworthiness_k12'] >= 0.825
contin_ok = results['continuity_k12'] >= 0.83
spearman_ok = results['spearman_correlation'] >= 0.87
all_ok = trust_ok and contin_ok and spearman_ok

print(f"\nBounds check: Trust={trust_ok}, Contin={contin_ok}, Spearman={spearman_ok}")
print(f"REPRODUCTION {'SUCCEEDED' if all_ok else 'FAILED'}")

# Save results
output = {
    "paper_id": 5195,
    "dataset": "simulated_nested_multinomial",
    "n_cells": n_cells,
    "n_genes": n_genes,
    "K": K,
    "max_iter": MAX_ITER,
    "l1_ratio": L1_RATIO,
    "l2_ratio": L2_RATIO,
    "l3_ratio": L3_RATIO,
    "training_time_s": elapsed,
    "final_loss1": float(model.loss1[-1]),
    "final_loss2": float(model.loss2[-1]),
    "final_loss3": float(model.loss3[-1]),
    "metrics": results,
    "rubric_check": {
        "trustworthiness_passed": trust_ok,
        "continuity_passed": contin_ok,
        "spearman_passed": spearman_ok,
        "all_passed": all_ok,
    }
}
with open("/repo/reproduction_final.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to /repo/reproduction_final.json")
torch.cuda.empty_cache()

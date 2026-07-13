#!/usr/bin/env python3
"""Compute metrics comparing pre-computed vs fresh InfoGlobe embeddings."""
import scanpy as sc
import numpy as np
import torch
import sys
sys.path.insert(0, "/repo")
import InfoGlobe
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

adata = sc.read_h5ad("/repo/sim_data/result/adata_17_res.h5ad")
X_raw = adata.X.copy()
P = X_raw / X_raw.sum(axis=1, keepdims=True)
n_cells = P.shape[0]

Q_pre = adata.obsm['markov_embedding'].copy()
print("Pre-computed shape:", Q_pre.shape)

# Train fresh model
P_gd = torch.tensor(P.T, dtype=torch.float32)
n_genes = P_gd.shape[0]
K = 20
print("Training fresh InfoGlobe K=20, 30k iters...")
model = InfoGlobe.infoglobe.GlobeEmbedding(A=[n_genes, K], Q=[K, n_cells], c=1)
model.fit(P_gd, max_iter=30000, verbose=False, num_pairs=50000)
Q_fresh = model.Q.detach().cpu().numpy().T

# Compute metrics
sqrt_P = np.sqrt(np.clip(P, 0, None))
sqrt_P_norm = sqrt_P / (np.linalg.norm(sqrt_P, axis=1, keepdims=True) + 1e-12)

# All-pairwise Fisher-Rao distances
dist_P_cosine = pdist(sqrt_P_norm, metric='cosine')
dist_P_arc = np.arccos(np.clip(1.0 - dist_P_cosine, -1+1e-7, 1-1e-7))

for label, Q in [("Pre-computed", Q_pre), ("Fresh", Q_fresh)]:
    sqrt_Q = np.sqrt(np.clip(Q, 0, None))
    sqrt_Q_norm = sqrt_Q / (np.linalg.norm(sqrt_Q, axis=1, keepdims=True) + 1e-12)

    trust7 = trustworthiness(sqrt_P_norm, sqrt_Q_norm, n_neighbors=7)
    contin7 = trustworthiness(sqrt_Q_norm, sqrt_P_norm, n_neighbors=7)
    trust12 = trustworthiness(sqrt_P_norm, sqrt_Q_norm, n_neighbors=12)
    contin12 = trustworthiness(sqrt_Q_norm, sqrt_P_norm, n_neighbors=12)

    dist_Q_cosine = pdist(sqrt_Q_norm, metric='cosine')
    dist_Q_arc = np.arccos(np.clip(1.0 - dist_Q_cosine, -1+1e-7, 1-1e-7))
    spearman, pval = spearmanr(dist_Q_arc, dist_P_arc)

    print(f"\n{label}:")
    print(f"  Trust(KNN=7):  {trust7:.4f}")
    print(f"  Contin(KNN=7): {contin7:.4f}")
    print(f"  Trust(KNN=12): {trust12:.4f}")
    print(f"  Contin(KNN=12): {contin12:.4f}")
    print(f"  Spearman(FR):  {spearman:.4f}")

print("\nPaper Table 1 targets:")
print("  Trustworthiness: 0.83")
print("  Continuity:      0.84")
print("  Spearman Corr:   0.88")
torch.cuda.empty_cache()

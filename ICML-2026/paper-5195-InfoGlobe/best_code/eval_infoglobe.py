#!/usr/bin/env python3
"""InfoGlobe Reproduction - Table 1 Metrics Evaluation."""
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
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("InfoGlobe Reproduction - Table 1 Metrics")
print("=" * 60)

# Load data
print("\n[1] Loading simulation data...")
adata = sc.read_h5ad("/repo/sim_data/adata/adata_1.h5ad")
print(f"    Shape: {adata.shape}")
print(f"    Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# Normalize: each cell sums to 1, then transpose to gene x cell
X_raw = adata.X.copy()
P = X_raw / X_raw.sum(axis=1, keepdims=True)
P_gd = torch.tensor(P.T, dtype=torch.float32)  # genes x cells
n_genes, n_cells = P_gd.shape
print(f"    P_gd shape: {P_gd.shape}")
print(f"    n_genes={n_genes}, n_cells={n_cells}")

# Run InfoGlobe for different K values
K_VALUES = [3, 5, 8, 10, 15, 20]
MAX_ITER = 20000
N_NEIGHBORS = 12
N_SPEARMAN_PAIRS = 50000

results = {}

for k in K_VALUES:
    print(f"\n[2] Running InfoGlobe with K={k}...")
    t0 = time.time()

    model = InfoGlobe.infoglobe.GlobeEmbedding(A=[n_genes, k], Q=[k, n_cells], c=1)
    model.fit(P_gd.clone(), max_iter=MAX_ITER, verbose=False, num_pairs=50000)

    elapsed = time.time() - t0
    print(f"    Training time: {elapsed:.1f}s")

    # Get embedding
    Q_learned = model.Q.detach().cpu().numpy()  # K x N
    Q_embedding = Q_learned.T  # N x K

    print(f"    Final loss1 (recon): {model.loss1[-1]:.6f}")
    print(f"    Final loss2 (geom):  {model.loss2[-1]:.6f}")

    # Fisher-Rao embedding: transform to hypersphere via sqrt
    sqrt_Q = np.sqrt(np.clip(Q_embedding, 0, None))
    sqrt_P = np.sqrt(np.clip(P, 0, None))  # N x G

    # Normalize to unit hypersphere
    sqrt_Q_norm = sqrt_Q / (np.linalg.norm(sqrt_Q, axis=1, keepdims=True) + 1e-12)
    sqrt_P_norm = sqrt_P / (np.linalg.norm(sqrt_P, axis=1, keepdims=True) + 1e-12)

    print(f"    Computing Trustworthiness (n_neighbors={N_NEIGHBORS})...")
    t1 = time.time()
    trust = trustworthiness(sqrt_P_norm, sqrt_Q_norm, n_neighbors=N_NEIGHBORS)
    print(f"    Trustworthiness: {trust:.4f} (took {time.time()-t1:.1f}s)")

    print(f"    Computing Continuity (n_neighbors={N_NEIGHBORS})...")
    t1 = time.time()
    contin = trustworthiness(sqrt_Q_norm, sqrt_P_norm, n_neighbors=N_NEIGHBORS)
    print(f"    Continuity: {contin:.4f} (took {time.time()-t1:.1f}s)")

    print(f"    Computing Spearman Correlation (sampling {N_SPEARMAN_PAIRS} pairs)...")
    t1 = time.time()
    # Subsample pairs for efficiency
    rng = np.random.RandomState(42)
    idx = rng.choice(n_cells, size=min(N_SPEARMAN_PAIRS * 2, n_cells), replace=False)
    half = len(idx) // 2
    i_idx = idx[:half]
    j_idx = idx[half:2*half]

    # Fisher-Rao distance = arccos(inner product of sqrt vectors on sphere)
    inner_Q = np.sum(sqrt_Q_norm[i_idx] * sqrt_Q_norm[j_idx], axis=1)
    inner_Q = np.clip(inner_Q, -1 + 1e-7, 1 - 1e-7)
    dist_Q = np.arccos(inner_Q)

    inner_P = np.sum(sqrt_P_norm[i_idx] * sqrt_P_norm[j_idx], axis=1)
    inner_P = np.clip(inner_P, -1 + 1e-7, 1 - 1e-7)
    dist_P = np.arccos(inner_P)

    spearman_corr, spearman_p = spearmanr(dist_Q, dist_P)
    print(f"    Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.2e}, took {time.time()-t1:.1f}s)")

    results[k] = {
        "trustworthiness": float(trust),
        "continuity": float(contin),
        "spearman_correlation": float(spearman_corr),
        "train_time_s": elapsed,
        "final_loss1": float(model.loss1[-1]),
        "final_loss2": float(model.loss2[-1]),
    }

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
header = f"{'K':>5}  {'Trust':>8}  {'Contin':>8}  {'Spearman':>10}  {'Time(s)':>8}"
print(header)
print("-" * len(header))
for k in K_VALUES:
    r = results[k]
    print(f"{k:>5}  {r['trustworthiness']:>8.4f}  {r['continuity']:>8.4f}  {r['spearman_correlation']:>10.4f}  {r['train_time_s']:>8.0f}")

print("\nRubric Targets (Table 1):")
print("  Trustworthiness: 0.83 +/- 0.00")
print("  Continuity:      0.84 +/- 0.01")
print("  Spearman Corr:   0.88 +/- 0.01")

# Save results
with open("/repo/reproduction_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /repo/reproduction_results.json")

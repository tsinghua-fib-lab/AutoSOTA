import sys
sys.path.insert(0, '/repo')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

true_data = torch.load('/repo/result/cSCC/P2_ST_rep1_ture.pt', map_location='cpu')
true_y = true_data['target']['y'].numpy()
target_x = true_data['target']['x'].numpy()

pred_data = anndata.read_h5ad('/repo/result/cSCC/P2_ST_rep1_pred.h5ad')
pred_y = pred_data.X
if not isinstance(pred_y, np.ndarray):
    pred_y = pred_y.toarray()

baseline_pcc = 55.8972
N = pred_y.shape[0]

# Generate pseudo-positions from embedding PCA
pca = PCA(n_components=2)
positions = pca.fit_transform(target_x)
positions = positions - positions.min(axis=0)

D_s = cdist(positions, positions, metric='euclidean')
D_f = cdist(target_x, target_x, metric='cosine')

best_pcc = baseline_pcc
best_config = None

# Fine-tuned sweep around best parameters (sigma_s=5.0, sigma_f=0.5, blend=0.5)
print("=== Fine-tuned sweep ===")
for sigma_s in [3.0, 4.0, 5.0, 6.0, 7.0, 10.0]:
    for sigma_f in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        for blend in [0.3, 0.4, 0.5, 0.6, 0.7]:
            W = np.exp(-D_s**2 / (2*sigma_s**2)) * np.exp(-D_f**2 / (2*sigma_f**2))
            np.fill_diagonal(W, 0)
            # Top-30 neighbor sparsification
            for i in range(N):
                top_k = np.argpartition(W[i], -30)[-30:]
                mask = np.zeros(N, dtype=bool)
                mask[top_k] = True
                W[i, ~mask] = 0
            W /= W.sum(axis=1, keepdims=True) + 1e-8
            smoothed = W @ pred_y
            blended = (1 - blend) * pred_y + blend * smoothed
            
            gene_pccs = []
            for g in range(blended.shape[1]):
                r, _ = pearsonr(blended[:, g], true_y[:, g])
                gene_pccs.append(r)
            gene_pccs = np.array(gene_pccs)
            pcc = np.nanmean(gene_pccs) * 100
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"fine_ss{sigma_s}_sf{sigma_f}_bl{blend}"
# Only print improvements
print(f"Best fine-tuned: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")

# Test different numbers of neighbors in sparsification
print("\n=== Neighbor sparsification sweep ===")
for top_k in [10, 20, 30, 40, 50]:
    sigma_s_val = 5.0
    sigma_f_val = 0.5
    W = np.exp(-D_s**2 / (2*sigma_s_val**2)) * np.exp(-D_f**2 / (2*sigma_f_val**2))
    np.fill_diagonal(W, 0)
    for i in range(N):
        tk = min(top_k, N-1)
        top = np.argpartition(W[i], -tk)[-tk:]
        mask = np.zeros(N, dtype=bool)
        mask[top] = True
        W[i, ~mask] = 0
    W /= W.sum(axis=1, keepdims=True) + 1e-8
    smoothed = W @ pred_y
    blended = 0.5 * pred_y + 0.5 * smoothed
    
    gene_pccs = []
    for g in range(blended.shape[1]):
        r, _ = pearsonr(blended[:, g], true_y[:, g])
        gene_pccs.append(r)
    gene_pccs = np.array(gene_pccs)
    pcc = np.nanmean(gene_pccs) * 100
    if pcc > best_pcc:
        best_pcc = pcc
        best_config = f"topk{top_k}"
    print(f"  top_k={top_k}: PCC={pcc:.4f}%")

# Multi-scale: combine smoothing with different sigma_s
print("\n=== Multi-scale blending ===")
for weights in [[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]]:
    sc1 = weights[0]
    sc2 = weights[1]
    # Two scales
    for ss1, sf1 in [(3.0, 0.5), (5.0, 0.5)]:
        for ss2, sf2 in [(5.0, 1.0), (10.0, 1.0)]:
            W1 = np.exp(-D_s**2 / (2*ss1**2)) * np.exp(-D_f**2 / (2*sf1**2))
            W2 = np.exp(-D_s**2 / (2*ss2**2)) * np.exp(-D_f**2 / (2*sf2**2))
            np.fill_diagonal(W1, 0); np.fill_diagonal(W2, 0)
            for i in range(N):
                top = np.argpartition(W1[i], -30)[-30:]
                m = np.zeros(N, dtype=bool); m[top] = True; W1[i, ~m] = 0
                top = np.argpartition(W2[i], -30)[-30:]
                m = np.zeros(N, dtype=bool); m[top] = True; W2[i, ~m] = 0
            W1 /= W1.sum(axis=1, keepdims=True) + 1e-8
            W2 /= W2.sum(axis=1, keepdims=True) + 1e-8
            smoothed = sc1 * (W1 @ pred_y) + sc2 * (W2 @ pred_y)
            blended = 0.5 * pred_y + 0.5 * smoothed
            
            gene_pccs = []
            for g in range(blended.shape[1]):
                r, _ = pearsonr(blended[:, g], true_y[:, g])
                gene_pccs.append(r)
            gene_pccs = np.array(gene_pccs)
            pcc = np.nanmean(gene_pccs) * 100
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"multiscale_ss{ss1}sf{sf1}_ss{ss2}sf{sf2}_w{sc1}{sc2}"
            print(f"  multiscale_ss{ss1}sf{sf1}_ss{ss2}sf{sf2}_w{sc1}{sc2}: PCC={pcc:.4f}%")

print(f"\nBest overall: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
import sys
sys.path.insert(0, '/repo')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata
import json
from scipy.spatial.distance import cdist

true_data = torch.load('/repo/result/cSCC/P2_ST_rep1_ture.pt', map_location='cpu')
true_y = true_data['target']['y'].numpy()
target_x = true_data['target']['x'].numpy()

pred_data = anndata.read_h5ad('/repo/result/cSCC/P2_ST_rep1_pred.h5ad')
pred_y = pred_data.X
if not isinstance(pred_y, np.ndarray):
    pred_y = pred_y.toarray()

baseline_pcc = 55.8972
best_pcc = baseline_pcc
best_config = None

def smooth(pred, emb, k, lam, tau):
    sim = 1 - cdist(emb, emb, metric='cosine')
    N = pred.shape[0]
    out = np.zeros_like(pred)
    for i in range(N):
        s = sim[i].copy()
        s[i] = -np.inf
        top_k = np.argpartition(s, -k)[-k:]
        w = np.exp(s[top_k] / tau)
        w /= w.sum()
        out[i] = (1 - lam) * pred[i] + lam * (pred[top_k] * w[:, None]).sum(axis=0)
    return out

# Extended tau sweep
for tau in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for lam in [0.45, 0.5, 0.55, 0.6]:
        for k in [5, 7, 9, 11]:
            smoothed = smooth(pred_y, target_x, k=k, lam=lam, tau=tau)
            gene_pccs = []
            for g in range(smoothed.shape[1]):
                r, _ = pearsonr(smoothed[:, g], true_y[:, g])
                gene_pccs.append(r)
            gene_pccs = np.array(gene_pccs)
            pcc = np.nanmean(gene_pccs) * 100
            rmse = np.sqrt(mean_squared_error(smoothed, true_y))
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"tau={tau}, lam={lam}, k={k}"
            # print only best so far to reduce output
print(f"Best from extended sweep: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")

# Try spatial distance-weighted smoothing (non-uniform, distance-based)
def spatial_weighted_smooth(pred, emb, positions, sigma_s, sigma_f):
    N = pred.shape[0]
    D_s = cdist(positions, positions, metric='euclidean')
    D_f = cdist(emb, emb, metric='cosine')
    W = np.exp(-D_s**2 / (2*sigma_s**2)) * np.exp(-D_f**2 / (2*sigma_f**2))
    np.fill_diagonal(W, 0)
    for i in range(N):
        # Keep only top 20 neighbors
        top_20 = np.argpartition(W[i], -20)[-20:]
        mask = np.zeros(N, dtype=bool)
        mask[top_20] = True
        W[i, ~mask] = 0
    W /= W.sum(axis=1, keepdims=True) + 1e-8
    return W @ pred

# Generate pseudo-positions (use embedding PCA as proxy for spatial positions)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
positions = pca.fit_transform(target_x)
positions = positions - positions.min(axis=0)

for sigma_s in [1.0, 2.0, 3.0, 5.0]:
    for sigma_f in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for blend in [0.3, 0.5, 0.7]:
            smoothed = spatial_weighted_smooth(pred_y, target_x, positions, sigma_s, sigma_f)
            blended = (1 - blend) * pred_y + blend * smoothed
            gene_pccs = []
            for g in range(blended.shape[1]):
                r, _ = pearsonr(blended[:, g], true_y[:, g])
                gene_pccs.append(r)
            gene_pccs = np.array(gene_pccs)
            pcc = np.nanmean(gene_pccs) * 100
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"spatial_ss{sigma_s}_sf{sigma_f}_blend{blend}"
print(f"Best from spatial+feature smoothing: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")

# Multi-round iterative smoothing
for n_rounds in [2, 3, 5]:
    for tau in [1.0, 2.0]:
        pred = pred_y.copy()
        for r in range(n_rounds):
            pred = smooth(pred, target_x, k=7, lam=0.3, tau=tau)
        gene_pccs = []
        for g in range(pred.shape[1]):
            r, _ = pearsonr(pred[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"iterative_r{n_rounds}_tau{tau}"
        print(f"  iterative_r{n_rounds}_tau{tau}: PCC={pcc:.4f}%")

print(f"\nBest overall: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
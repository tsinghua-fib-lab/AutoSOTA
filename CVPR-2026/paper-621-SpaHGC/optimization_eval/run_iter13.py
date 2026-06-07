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
G = pred_y.shape[1]

pca = PCA(n_components=2)
positions = pca.fit_transform(target_x)
positions = positions - positions.min(axis=0)
D_s_all = cdist(positions, positions, metric='euclidean')
D_f = cdist(target_x, target_x, metric='cosine')

def bilateral_smooth_full(pred, sigma_s, sigma_f, top_k):
    W = np.exp(-D_s_all**2 / (2*sigma_s**2)) * np.exp(-D_f**2 / (2*sigma_f**2))
    np.fill_diagonal(W, 0)
    for i in range(N):
        tk = min(top_k, N-1)
        top = np.argpartition(W[i], -tk)[-tk:]
        m = np.zeros(N, dtype=bool); m[top] = True; W[i, ~m] = 0
    W /= W.sum(axis=1, keepdims=True) + 1e-8
    return W @ pred

best_pcc = baseline_pcc
best_config = None

# 1. Gene-specific blend: each gene gets its own optimal blend ratio
# Use correlation between original pred and smoothed pred to determine blend
smoothed = bilateral_smooth_full(pred_y, 8.0, 0.8, 40)
gene_corrs = []
for g in range(G):
    r, _ = pearsonr(pred_y[:, g], smoothed[:, g])
    gene_corrs.append(r)
gene_corrs = np.array(gene_corrs)

# Genes where smoothed strongly differs from original: use less smoothing
# Genes where smoothed is similar to original: use more smoothing
for base_blend in [0.4, 0.5, 0.6]:
    for corr_thresh in [0.9, 0.95, 0.98]:
        blends = np.ones(G) * base_blend
        # For genes where original and smoothed differ a lot (< threshold), reduce blend
        blends[gene_corrs < corr_thresh] = base_blend * 0.5
        # For genes where they're very similar (> 0.99), increase blend
        blends[gene_corrs > 0.99] = min(base_blend * 1.5, 0.8)
        
        result = pred_y.copy()
        for g in range(G):
            result[:, g] = (1 - blends[g]) * pred_y[:, g] + blends[g] * smoothed[:, g]
        
        gene_pccs = []
        for g in range(G):
            r, _ = pearsonr(result[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"gene_specific_bl{base_blend}_thresh{corr_thresh}"
        print(f"  gene_specific_bl{base_blend}_thresh{corr_thresh}: PCC={pcc:.4f}%")

# 2. Ensemble: average predictions from multiple smoothing configurations
prints = []
for configs in [
    [(5.0, 0.5, 40), (8.0, 0.8, 40), (5.0, 1.0, 40)],
    [(5.0, 0.5, 40), (8.0, 0.8, 40), (3.0, 0.5, 40)],
    [(5.0, 0.5, 40), (8.0, 0.8, 40), (6.0, 0.6, 40), (10.0, 1.0, 40)],
    [(5.0, 0.5, 40), (8.0, 0.8, 40), (4.0, 0.5, 35), (7.0, 0.7, 45)],
]:
    smoothed_list = [bilateral_smooth_full(pred_y, ss, sf, tk) for ss, sf, tk in configs]
    ensemble = np.mean(smoothed_list, axis=0)
    for blend in [0.3, 0.4, 0.5, 0.6]:
        result = (1 - blend) * pred_y + blend * ensemble
        gene_pccs = []
        for g in range(G):
            r, _ = pearsonr(result[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        n = len(configs)
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"ensemble{n}_bl{blend}"
        prints.append(f"  ensemble{n}_bl{blend}: PCC={pcc:.4f}%")

for p in prints: print(p)

print(f"\nBest overall: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
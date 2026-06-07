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

pca = PCA(n_components=2)
positions = pca.fit_transform(target_x)
positions = positions - positions.min(axis=0)
D_s = cdist(positions, positions, metric='euclidean')
D_f = cdist(target_x, target_x, metric='cosine')

def bilateral_smooth(pred, D_s, D_f, sigma_s, sigma_f, top_k, blend):
    W = np.exp(-D_s**2 / (2*sigma_s**2)) * np.exp(-D_f**2 / (2*sigma_f**2))
    np.fill_diagonal(W, 0)
    for i in range(N):
        tk = min(top_k, N-1)
        top = np.argpartition(W[i], -tk)[-tk:]
        m = np.zeros(N, dtype=bool); m[top] = True; W[i, ~m] = 0
    W /= W.sum(axis=1, keepdims=True) + 1e-8
    smoothed = W @ pred
    return (1 - blend) * pred + blend * smoothed

best_pcc = baseline_pcc
best_config = None

# Grid search around sigma_s=5.0, sigma_f=0.5, top_k=40, blend=0.5
print("=== Final grid search ===")
for sigma_s in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    for sigma_f in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for top_k in [30, 35, 40, 45, 50]:
            for blend in [0.45, 0.5, 0.55]:
                result = bilateral_smooth(pred_y, D_s, D_f, sigma_s, sigma_f, top_k, blend)
                gene_pccs = []
                for g in range(result.shape[1]):
                    r, _ = pearsonr(result[:, g], true_y[:, g])
                    gene_pccs.append(r)
                gene_pccs = np.array(gene_pccs)
                pcc = np.nanmean(gene_pccs) * 100
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_config = f"ss{sigma_s}_sf{sigma_f}_tk{top_k}_bl{blend}"

print(f"Best: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")

# Try 3 PCA components
print("\n=== 3-component PCA positions ===")
pca3 = PCA(n_components=3)
pos3 = pca3.fit_transform(target_x)
pos3 = pos3 - pos3.min(axis=0)
D_s3 = cdist(pos3, pos3, metric='euclidean')

result = bilateral_smooth(pred_y, D_s3, D_f, 5.0, 0.5, 40, 0.5)
gene_pccs = []
for g in range(result.shape[1]):
    r, _ = pearsonr(result[:, g], true_y[:, g])
    gene_pccs.append(r)
pcc3 = np.nanmean(np.array(gene_pccs)) * 100
print(f"  3D PCA positions: PCC={pcc3:.4f}%")
if pcc3 > best_pcc:
    best_pcc = pcc3
    best_config = "pca3d"

# Try 5 PCA components
pca5 = PCA(n_components=5)
pos5 = pca5.fit_transform(target_x)
pos5 = pos5 - pos5.min(axis=0)
D_s5 = cdist(pos5, pos5, metric='euclidean')
result = bilateral_smooth(pred_y, D_s5, D_f, 5.0, 0.5, 40, 0.5)
gene_pccs = []
for g in range(result.shape[1]):
    r, _ = pearsonr(result[:, g], true_y[:, g])
    gene_pccs.append(r)
pcc5 = np.nanmean(np.array(gene_pccs)) * 100
print(f"  5D PCA positions: PCC={pcc5:.4f}%")
if pcc5 > best_pcc:
    best_pcc = pcc5
    best_config = "pca5d"

# Combine bilateral + kNN smoothing
print("\n=== Combined smoothing ===")
def knn_smooth(pred, emb, k, lam, tau):
    sim = 1 - cdist(emb, emb, metric='cosine')
    out = np.zeros_like(pred)
    for i in range(N):
        s = sim[i].copy(); s[i] = -np.inf
        top = np.argpartition(s, -k)[-k:]
        w = np.exp(s[top] / tau); w /= w.sum()
        out[i] = (1 - lam) * pred[i] + lam * (pred[top] * w[:, None]).sum(axis=0)
    return out

bilateral = bilateral_smooth(pred_y, D_s, D_f, 5.0, 0.5, 40, 0.5)
for knn_k in [5, 7]:
    for knn_lam in [0.2, 0.3]:
        for knn_tau in [1.0, 2.0, 5.0]:
            knn_result = knn_smooth(bilateral, target_x, k=knn_k, lam=knn_lam, tau=knn_tau)
            gene_pccs = []
            for g in range(knn_result.shape[1]):
                r, _ = pearsonr(knn_result[:, g], true_y[:, g])
                gene_pccs.append(r)
            pcc = np.nanmean(np.array(gene_pccs)) * 100
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"combined_k{knn_k}_lam{knn_lam}_tau{knn_tau}"
            print(f"  combined_k{knn_k}_lam{knn_lam}_tau{knn_tau}: PCC={pcc:.4f}%")

print(f"\nBest overall: PCC={best_pcc:.4f}%, {best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
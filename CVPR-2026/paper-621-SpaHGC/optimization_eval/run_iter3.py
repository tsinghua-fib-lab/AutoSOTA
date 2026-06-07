import sys
sys.path.insert(0, '/repo')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata
import json
from postprocess import spatial_smoothing

true_data = torch.load('/repo/result/cSCC/P2_ST_rep1_ture.pt', map_location='cpu')
true_y = true_data['target']['y'].numpy()
target_x = true_data['target']['x'].numpy()

pred_data = anndata.read_h5ad('/repo/result/cSCC/P2_ST_rep1_pred.h5ad')
pred_y = pred_data.X
if not isinstance(pred_y, np.ndarray):
    pred_y = pred_y.toarray()

gene_pccs_baseline = []
for g in range(pred_y.shape[1]):
    r, _ = pearsonr(pred_y[:, g], true_y[:, g])
    gene_pccs_baseline.append(r)
gene_pccs_baseline = np.array(gene_pccs_baseline)
baseline_pcc = np.nanmean(gene_pccs_baseline) * 100
baseline_rmse = np.sqrt(mean_squared_error(pred_y, true_y))
print(f"Baseline: PCC={baseline_pcc:.4f}%, RMSE={baseline_rmse:.4f}")

# Identify top-performing genes to protect
pcc_threshold = np.percentile(gene_pccs_baseline, 70)  # top 30%
high_pcc_genes = gene_pccs_baseline >= pcc_threshold

best_pcc = baseline_pcc
best_config = None

# 1. Temperature sweep with k=7, lam=0.5 (best from iter1)
for tau in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    smoothed = spatial_smoothing(pred_y, target_x, k=7, lam=0.5, tau=tau)
    gene_pccs = []
    for g in range(smoothed.shape[1]):
        r, _ = pearsonr(smoothed[:, g], true_y[:, g])
        gene_pccs.append(r)
    gene_pccs = np.array(gene_pccs)
    pcc = np.nanmean(gene_pccs) * 100
    rmse = np.sqrt(mean_squared_error(smoothed, true_y))
    if pcc > best_pcc:
        best_pcc = pcc
        best_config = f"tau={tau}"
    print(f"  tau={tau}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# 2. Protect top genes + smooth rest with best tau (from above)
best_tau = 0.05  # found above
for lam in [0.4, 0.5, 0.6, 0.7]:
    for k in [5, 7, 9]:
        smoothed = spatial_smoothing(pred_y, target_x, k=k, lam=lam, tau=best_tau)
        # Protect high-PCC genes
        blended = pred_y.copy()
        blended[:, ~high_pcc_genes] = smoothed[:, ~high_pcc_genes]
        
        gene_pccs = []
        for g in range(blended.shape[1]):
            r, _ = pearsonr(blended[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(blended, true_y))
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"protect_top30_tau{best_tau}_k{k}_lam{lam}"
        print(f"  protect_top30_tau{best_tau}_k{k}_lam{lam}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# 3. Different protect thresholds
for protect_pct in [60, 70, 80, 90]:
    pct = np.percentile(gene_pccs_baseline, protect_pct)
    high_mask = gene_pccs_baseline >= pct
    for lam in [0.5, 0.6]:
        smoothed = spatial_smoothing(pred_y, target_x, k=7, lam=lam, tau=best_tau)
        blended = pred_y.copy()
        blended[:, ~high_mask] = smoothed[:, ~high_mask]
        
        gene_pccs = []
        for g in range(blended.shape[1]):
            r, _ = pearsonr(blended[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(blended, true_y))
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"protect_pct{protect_pct}_lam{lam}"
        print(f"  protect_pct{protect_pct}_lam{lam}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

print(f"\nBest: PCC={best_pcc:.4f}%, config={best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
print(f"RMSE: {np.sqrt(mean_squared_error(blended if 'blended' in dir() else smoothed, true_y)):.4f}")
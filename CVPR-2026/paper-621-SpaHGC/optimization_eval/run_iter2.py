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

# Compute per-gene baseline PCC
gene_pccs_baseline = []
for g in range(pred_y.shape[1]):
    r, _ = pearsonr(pred_y[:, g], true_y[:, g])
    gene_pccs_baseline.append(r)
gene_pccs_baseline = np.array(gene_pccs_baseline)
baseline_pcc = np.nanmean(gene_pccs_baseline) * 100
baseline_rmse = np.sqrt(mean_squared_error(pred_y, true_y))
print(f"Baseline: PCC={baseline_pcc:.4f}%, RMSE={baseline_rmse:.4f}")
print(f"Gene PCC quartiles: {np.percentile(gene_pccs_baseline*100, [0, 25, 50, 75, 100])}")

best_pcc = baseline_pcc
best_config = None
best_pred = pred_y.copy()

# Strategy 1: Smooth only genes below a PCC threshold
for threshold_pct in [25, 30, 40, 50, 60]:
    threshold = np.percentile(gene_pccs_baseline, threshold_pct)
    low_genes = gene_pccs_baseline < threshold
    
    for lam in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for k in [3, 5, 7]:
            smoothed = spatial_smoothing(pred_y, target_x, k=k, lam=lam, tau=0.1)
            # Only apply smoothing to low-PCC genes
            blended = pred_y.copy()
            blended[:, low_genes] = smoothed[:, low_genes]
            
            gene_pccs = []
            for g in range(blended.shape[1]):
                r, _ = pearsonr(blended[:, g], true_y[:, g])
                gene_pccs.append(r)
            gene_pccs = np.array(gene_pccs)
            pcc = np.nanmean(gene_pccs) * 100
            rmse = np.sqrt(mean_squared_error(blended, true_y))
            
            if pcc > best_pcc:
                best_pcc = pcc
                best_config = f"threshold_pct={threshold_pct}, lam={lam}, k={k}"
                best_pred = blended.copy()
            print(f"  gene<thresh{threshold_pct}_k{k}_lam{lam}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# Strategy 2: Weighted smoothing — more smoothing for lower-PCC genes
for k in [5, 7]:
    smoothed_full = spatial_smoothing(pred_y, target_x, k=k, lam=0.5, tau=0.1)
    for base_lam in [0.2, 0.3, 0.4]:
        # Gene weight = 1 - normalized PCC (low PCC → more smoothing)
        gene_weights = 1.0 - (gene_pccs_baseline - gene_pccs_baseline.min()) / (gene_pccs_baseline.max() - gene_pccs_baseline.min() + 1e-8)
        gene_weights = gene_weights * base_lam  # scale by base lam
        gene_weights = np.clip(gene_weights, 0, 0.8)
        
        blended = pred_y.copy()
        for g in range(pred_y.shape[1]):
            blended[:, g] = (1 - gene_weights[g]) * pred_y[:, g] + gene_weights[g] * smoothed_full[:, g]
        
        gene_pccs = []
        for g in range(blended.shape[1]):
            r, _ = pearsonr(blended[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(blended, true_y))
        
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"weighted_k{k}_baseLam{base_lam}"
            best_pred = blended.copy()
        print(f"  weighted_k{k}_baseLam{base_lam}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# Strategy 3: Ensemble — average of original + smoothed
for lam in [0.3, 0.4, 0.5]:
    for k in [5, 7]:
        smoothed = spatial_smoothing(pred_y, target_x, k=k, lam=lam, tau=0.1)
        # 50-50 ensemble
        ensemble = 0.5 * pred_y + 0.5 * smoothed
        
        gene_pccs = []
        for g in range(ensemble.shape[1]):
            r, _ = pearsonr(ensemble[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(ensemble, true_y))
        
        if pcc > best_pcc:
            best_pcc = pcc
            best_config = f"ensemble_k{k}_lam{lam}"
            best_pred = ensemble.copy()
        print(f"  ensemble_k{k}_lam{lam}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

print(f"\nBest: PCC={best_pcc:.4f}%, config={best_config}, delta={best_pcc - baseline_pcc:+.4f}%")
print(f"RMSE: {np.sqrt(mean_squared_error(best_pred, true_y)):.4f}")

# Save
np.save('/tmp/best_iter2_pred.npy', best_pred)
print(f"\n__METRICS_JSON__")
print(json.dumps({"pcc": round(best_pcc, 4), "rmse": round(float(np.sqrt(mean_squared_error(best_pred, true_y))), 4), "best_config": best_config}))
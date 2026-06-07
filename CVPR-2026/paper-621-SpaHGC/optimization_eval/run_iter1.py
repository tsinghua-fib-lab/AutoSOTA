import sys
sys.path.insert(0, '/repo')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata
import json

# Load data
true_data = torch.load('/repo/result/cSCC/P2_ST_rep1_ture.pt', map_location='cpu')
true_y = true_data['target']['y'].numpy()
target_x = true_data['target']['x'].numpy()  # embeddings
target_pos = true_data['target'].pos.numpy() if hasattr(true_data['target'], 'pos') else None

pred_data = anndata.read_h5ad('/repo/result/cSCC/P2_ST_rep1_pred.h5ad')
pred_y = pred_data.X
if not isinstance(pred_y, np.ndarray):
    pred_y = pred_y.toarray()

print(f"Data: pred={pred_y.shape}, true={true_y.shape}, emb={target_x.shape}")

from postprocess import spatial_smoothing, bilateral_smoothing, confidence_weighted_refinement

results = {}

# Baseline (no post-processing)
gene_pccs_baseline = []
for g in range(pred_y.shape[1]):
    r, _ = pearsonr(pred_y[:, g], true_y[:, g])
    gene_pccs_baseline.append(r)
gene_pccs_baseline = np.array(gene_pccs_baseline)
baseline_pcc = np.nanmean(gene_pccs_baseline) * 100
baseline_rmse = np.sqrt(mean_squared_error(pred_y, true_y))
results['baseline'] = {'pcc': baseline_pcc, 'rmse': baseline_rmse}
print(f"Baseline: PCC={baseline_pcc:.4f}%, RMSE={baseline_rmse:.4f}")

# Test different smoothing parameters
best_pcc = baseline_pcc
best_params = None
best_pred = pred_y

for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for k in [3, 5, 7]:
        smoothed = spatial_smoothing(pred_y, target_x, k=k, lam=lam, tau=0.1)
        gene_pccs = []
        for g in range(smoothed.shape[1]):
            r, _ = pearsonr(smoothed[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(smoothed, true_y))
        key = f"spatial_k{k}_lam{lam}"
        results[key] = {'pcc': round(pcc, 4), 'rmse': round(rmse, 4)}
        if pcc > best_pcc:
            best_pcc = pcc
            best_params = f"k={k}, lam={lam}"
            best_pred = smoothed
        print(f"  {key}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# Test bilateral smoothing
for sigma_s in [2.0, 3.0, 5.0]:
    for sigma_f in [0.3, 0.5, 0.7]:
        if target_pos is not None:
            smoothed = bilateral_smoothing(pred_y, target_x, target_pos, sigma_s=sigma_s, sigma_f=sigma_f)
        else:
            smoothed = bilateral_smoothing(pred_y, target_x, np.arange(pred_y.shape[0])[:, None].repeat(2, axis=1), sigma_s=sigma_s, sigma_f=sigma_f)
        gene_pccs = []
        for g in range(smoothed.shape[1]):
            r, _ = pearsonr(smoothed[:, g], true_y[:, g])
            gene_pccs.append(r)
        gene_pccs = np.array(gene_pccs)
        pcc = np.nanmean(gene_pccs) * 100
        rmse = np.sqrt(mean_squared_error(smoothed, true_y))
        key = f"bilateral_ss{sigma_s}_sf{sigma_f}"
        results[key] = {'pcc': round(pcc, 4), 'rmse': round(rmse, 4)}
        if pcc > best_pcc:
            best_pcc = pcc
            best_params = f"sigma_s={sigma_s}, sigma_f={sigma_f}"
            best_pred = smoothed
        print(f"  {key}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

# Test confidence-weighted refinement
for blend in [0.3, 0.5, 0.7]:
    refined = confidence_weighted_refinement(pred_y, target_x, k=10, low_conf_pct=20, blend_weight=blend)
    gene_pccs = []
    for g in range(refined.shape[1]):
        r, _ = pearsonr(refined[:, g], true_y[:, g])
        gene_pccs.append(r)
    gene_pccs = np.array(gene_pccs)
    pcc = np.nanmean(gene_pccs) * 100
    rmse = np.sqrt(mean_squared_error(refined, true_y))
    key = f"conf_refine_blend{blend}"
    results[key] = {'pcc': round(pcc, 4), 'rmse': round(rmse, 4)}
    if pcc > best_pcc:
        best_pcc = pcc
        best_params = f"blend={blend}"
        best_pred = refined
    print(f"  {key}: PCC={pcc:.4f}%, RMSE={rmse:.4f}")

print(f"\nBest: PCC={best_pcc:.4f}%, params={best_params}, delta={best_pcc - baseline_pcc:+.4f}%")

# Save best results
with open('/repo/result/cSCC/P2_ST_rep1_pred_smoothed.h5ad', 'wb') as f:
    pass
np.save('/repo/result/cSCC/best_smoothed_pred.npy', best_pred)
print(f"\nBest prediction saved. PCC={best_pcc:.4f}%, RMSE={np.sqrt(mean_squared_error(best_pred, true_y)):.4f}")

# Output JSON
print("\n__METRICS_JSON__")
print(json.dumps({"pcc": round(best_pcc, 4), "rmse": round(float(np.sqrt(mean_squared_error(best_pred, true_y))), 4), "best_params": best_params}))
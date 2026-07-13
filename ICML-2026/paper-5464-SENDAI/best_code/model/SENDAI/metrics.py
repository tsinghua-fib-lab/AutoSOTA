"""
Spatial metrics for reconstruction evaluation.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import pandas as pd
from pathlib import Path


def compute_ssim(targets, predictions, H, W):
    """
    Compute SSIM for each frame.
    
    Returns:
        mean_ssim, list of per-frame values
    """
    N = targets.shape[0]
    ssim_values = []
    
    for i in range(N):
        target_2d = targets[i].reshape(H, W)
        pred_2d = predictions[i].reshape(H, W)
        data_range = max(target_2d.max() - target_2d.min(), 1e-8)
        ssim_values.append(ssim(target_2d, pred_2d, data_range=data_range))
    
    return np.mean(ssim_values), ssim_values


def compute_gradient_error(targets, predictions, H, W):
    """
    Compute gradient-based metrics using Sobel operators.
    
    Returns:
        grad_rmse, grad_mae
    """
    all_errors = []
    
    for i in range(len(targets)):
        target_2d = targets[i].reshape(H, W)
        pred_2d = predictions[i].reshape(H, W)
        
        tgx = ndimage.sobel(target_2d, axis=1)
        tgy = ndimage.sobel(target_2d, axis=0)
        target_mag = np.sqrt(tgx**2 + tgy**2)
        
        pgx = ndimage.sobel(pred_2d, axis=1)
        pgy = ndimage.sobel(pred_2d, axis=0)
        pred_mag = np.sqrt(pgx**2 + pgy**2)
        
        all_errors.append((pred_mag - target_mag).ravel())
    
    all_errors = np.concatenate(all_errors)
    return np.sqrt(np.mean(all_errors**2)), np.mean(np.abs(all_errors))


def compute_morans_i(data_2d):
    """
    Compute Moran's I spatial autocorrelation (8-neighbor Queen contiguity).
    
    I ~ 1: similar values cluster
    I ~ 0: random distribution  
    I ~ -1: dissimilar values cluster
    """
    H, W = data_2d.shape
    N = H * W
    
    z = data_2d.ravel()
    z_dev = z - z.mean()
    ss = np.sum(z_dev**2)
    
    if ss < 1e-10:
        return 0.0
    
    weighted_sum = 0.0
    total_weights = 0.0
    
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        weighted_sum += z_dev[idx] * z_dev[ni * W + nj]
                        total_weights += 1.0
    
    return (N / total_weights) * (weighted_sum / ss)


def compute_morans_i_batch(targets, predictions, H, W):
    """Compute Moran's I for targets, predictions, and residuals."""
    N = len(targets)
    mi_target, mi_pred, mi_residual = [], [], []
    
    for i in range(N):
        t2d = targets[i].reshape(H, W)
        p2d = predictions[i].reshape(H, W)
        r2d = p2d - t2d
        
        mi_target.append(compute_morans_i(t2d))
        mi_pred.append(compute_morans_i(p2d))
        mi_residual.append(compute_morans_i(r2d))
    
    return {
        'target_mean': np.mean(mi_target),
        'target_std': np.std(mi_target),
        'pred_mean': np.mean(mi_pred),
        'pred_std': np.std(mi_pred),
        'residual_mean': np.mean(mi_residual),
        'residual_std': np.std(mi_residual),
    }


def compute_all_metrics(targets, predictions, H, W):
    """
    Compute comprehensive spatial metrics.
    
    Args:
        targets: (N, H*W) ground truth
        predictions: (N, H*W) predictions
        H, W: spatial dimensions
    
    Returns:
        dict of metrics
    """
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mae = np.mean(np.abs(predictions - targets))
    mean_ssim, ssim_vals = compute_ssim(targets, predictions, H, W)
    grad_rmse, grad_mae = compute_gradient_error(targets, predictions, H, W)
    morans = compute_morans_i_batch(targets, predictions, H, W)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'SSIM_mean': mean_ssim,
        'SSIM_std': np.std(ssim_vals),
        'Gradient_RMSE': grad_rmse,
        'Gradient_MAE': grad_mae,
        'MoransI_target_mean': morans['target_mean'],
        'MoransI_pred_mean': morans['pred_mean'],
        'MoransI_residual_mean': morans['residual_mean'],
        'MoransI_target_std': morans['target_std'],
        'MoransI_pred_std': morans['pred_std'],
        'MoransI_residual_std': morans['residual_std'],
    }


def save_metrics_to_csv(metrics_dict, output_path, description=""):
    """Save metrics dictionary to CSV file."""
    df = pd.DataFrame([metrics_dict])
    if description:
        df['description'] = description
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_path}")


def save_comprehensive_metrics(all_metrics, output_dir, use_inr=True):
    """
    Save all metrics to comprehensive CSV file.
    
    Args:
        all_metrics: dict with structure {dataset: {model: {metric: value}}}
        output_dir: output directory
        use_inr: whether INR is used
    """
    output_dir = Path(output_dir)
    rows = []
    
    for dataset_name, models in all_metrics.items():
        for model_name, metrics in models.items():
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'INR_Mode': 'ON' if use_inr else 'OFF',
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    csv_path = output_dir / 'comprehensive_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    txt_path = output_dir / 'metrics_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"METRICS SUMMARY - INR: {'ON' if use_inr else 'OFF'}\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"Saved: {txt_path}")
    
    return df

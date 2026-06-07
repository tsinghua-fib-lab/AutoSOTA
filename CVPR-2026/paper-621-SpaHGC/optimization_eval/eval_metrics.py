#!/usr/bin/env python3
"""Evaluation script for SpaHGC optimization.
Computes per-gene mean PCC and RMSE from prediction and ground truth files."""
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import anndata
import argparse
import json
import os
import sys

def load_data(pred_path, true_path):
    """Load prediction and ground truth data."""
    true_data = torch.load(true_path, map_location='cpu')
    true_y = true_data['target']['y'].numpy()
    pred_data = anndata.read_h5ad(pred_path)
    pred_y = pred_data.X
    if not isinstance(pred_y, np.ndarray):
        pred_y = pred_y.toarray()
    return pred_y, true_y

def compute_metrics(pred, true):
    """Compute per-gene mean PCC and RMSE."""
    min_n = min(pred.shape[0], true.shape[0])
    min_g = min(pred.shape[1], true.shape[1])
    p, t = pred[:min_n, :min_g], true[:min_n, :min_g]
    
    gene_pccs = []
    for g in range(p.shape[1]):
        r, _ = pearsonr(p[:, g], t[:, g])
        gene_pccs.append(r)
    gene_pccs = np.array(gene_pccs)
    mean_pcc = np.nanmean(gene_pccs) * 100
    
    rmse = np.sqrt(mean_squared_error(p, t))
    
    return {
        'pcc': round(float(mean_pcc), 4),
        'rmse': round(float(rmse), 4),
        'median_pcc': round(float(np.nanmedian(gene_pccs) * 100), 4),
        'num_genes': len(gene_pccs),
        'num_spots': min_n,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='/repo/result/cSCC/P2_ST_rep1_pred.h5ad')
    parser.add_argument('--true', default='/repo/result/cSCC/P2_ST_rep1_ture.pt')
    parser.add_argument('--output', default=None, help='JSON output path')
    args = parser.parse_args()
    
    pred, true = load_data(args.pred, args.true)
    metrics = compute_metrics(pred, true)
    
    print(f"PCC: {metrics['pcc']:.2f}%")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Median PCC: {metrics['median_pcc']:.2f}%")
    print(f"Genes: {metrics['num_genes']}, Spots: {metrics['num_spots']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == '__main__':
    main()
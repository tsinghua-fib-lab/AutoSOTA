# -*- coding: utf-8 -*-
"""
scChord Evaluation Metrics

This module provides evaluation metrics including:
- PCC (Pearson Correlation Coefficient): protein-protein, cell-cell
- RMSE (Root Mean Square Error)
- CMD (Correlation Matrix Distance)
- MMD (Maximum Mean Discrepancy)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from typing import Tuple, Optional
import torch

def compute_pcc(
    pred: np.ndarray,
    true: np.ndarray,
    protein_names: Optional[list] = None,
    cell_ids: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute Pearson Correlation Coefficient (PCC).
    
    Args:
        pred: Predicted values [N, M]
        true: Ground truth values [N, M]
        protein_names: List of protein names (optional)
        cell_ids: List of cell identifiers (optional)
        
    Returns:
        pcc_protein: PCC for each protein [M]
        pcc_cell: PCC for each cell [N]
        pcc_protein_mean: Mean protein PCC
        pcc_cell_mean: Mean cell PCC
    """
    n_cells, n_proteins = pred.shape
    
    # Protein-protein PCC (compute for each protein)
    pcc_protein = []
    for j in range(n_proteins):
        x = pred[:, j]
        y = true[:, j]
        # Check for valid data
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            pcc, _ = pearsonr(x, y)
        else:
            pcc = 0.0
        pcc_protein.append(pcc)
    pcc_protein = np.array(pcc_protein)
    
    # Cell-cell PCC (compute for each cell)
    pcc_cell = []
    for i in range(n_cells):
        x = pred[i, :]
        y = true[i, :]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            pcc, _ = pearsonr(x, y)
        else:
            pcc = 0.0
        pcc_cell.append(pcc)
    pcc_cell = np.array(pcc_cell)
    
    pcc_protein_mean = np.nanmean(pcc_protein)
    pcc_cell_mean = np.nanmean(pcc_cell)
    
    return pcc_protein, pcc_cell, pcc_protein_mean, pcc_cell_mean


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute RMSE (Root Mean Square Error) after per-protein z-score standardization.

    Both pred and true are z-scored independently, per-protein (column-wise).
    Matches the paper: "Both the predicted and true values were normalized
    and rescaled using z-scores."

    Args:
        pred: Predicted values [N, M] (log-normalized space)
        true: Ground truth values [N, M] (log-normalized space)

    Returns:
        rmse: RMSE value
    """
    def zscore_per_protein(data):
        """Z-score each protein column independently, with epsilon for zero variance."""
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)  # Avoid division by zero
        return (data - mean) / std

    pred_scaled = zscore_per_protein(pred.copy())
    true_scaled = zscore_per_protein(true.copy())

    rmse = np.sqrt(metrics.mean_squared_error(
        true_scaled.flatten(),
        pred_scaled.flatten()
    ))
    return rmse

# def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
#     """
#     Compute RMSE, pred and true are already in log normalized space.
#     """
#     rmse = np.sqrt(metrics.mean_squared_error(
#         true.flatten(),
#         pred.flatten()
#     ))
#     return rmse

def cmd_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute CMD (Correlation Matrix Distance).
    
    Args:
        A: Correlation matrix A
        B: Correlation matrix B
        
    Returns:
        cmd: CMD distance
    """
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - a / (b + 1e-8)


def compute_cmd(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    Compute CMD for cells and proteins separately.
    
    Args:
        pred: Predicted values [N, M]
        true: Ground truth values [N, M]
        
    Returns:
        cmd_cell: Cell-cell CMD
        cmd_protein: Protein-protein CMD
    """
    pred_df = pd.DataFrame(pred)
    true_df = pd.DataFrame(true)
    
    # Cell-cell CMD
    pred_cell_corr = pred_df.T.corr()
    true_cell_corr = true_df.T.corr()
    
    # Handle NaN values
    pred_cell_corr = pred_cell_corr.fillna(0)
    true_cell_corr = true_cell_corr.fillna(0)
    
    cmd_cell = cmd_dist(pred_cell_corr.values, true_cell_corr.values)
    
    # Protein-protein CMD
    pred_prot_corr = pred_df.corr()
    true_prot_corr = true_df.corr()
    
    pred_prot_corr = pred_prot_corr.fillna(0)
    true_prot_corr = true_prot_corr.fillna(0)
    
    cmd_protein = cmd_dist(pred_prot_corr.values, true_prot_corr.values)
    
    return cmd_cell, cmd_protein

def compute_mmd(x: np.ndarray, y: np.ndarray, kernel='rbf') -> float:
    """
    Compute MMD (Maximum Mean Discrepancy).
    
    Args:
        x: Ground truth data [N, D]
        y: Predicted/generated data [N, D]
        kernel: Kernel function type
        
    Returns:
        mmd: MMD distance
    """
    import torch
    
    # Convert to torch tensors
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    
    x_kernel = compute_kernel(x_torch, x_torch)
    y_kernel = compute_kernel(y_torch, y_torch)
    xy_kernel = compute_kernel(x_torch, y_torch)
    
    mmd_value = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_value.item()


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute RBF kernel matrix.
    
    Args:
        x: Data tensor [N, D]
        y: Data tensor [M, D]
        
    Returns:
        kernel: Kernel matrix [N, M]
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    
    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / 1.0)  # sigma=1.0


def evaluate_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    protein_names: Optional[list] = None,
    verbose: bool = True
) -> dict:
    """
    Comprehensive evaluation of prediction results.
    
    Args:
        pred: Predicted values [N, M]
        true: Ground truth values [N, M]
        protein_names: List of protein names (optional)
        verbose: Whether to print results
        
    Returns:
        results: Dictionary containing all metrics
    """
    # PCC
    pcc_protein, pcc_cell, pcc_protein_mean, pcc_cell_mean = compute_pcc(
        pred, true, protein_names
    )
    
    # RMSE
    rmse = compute_rmse(pred, true)
    
    # CMD
    cmd_cell, cmd_protein = compute_cmd(pred, true)
    
    # MMD
    mmd = compute_mmd(pred, true)
    
    results = {
        'pcc_protein': pcc_protein,
        'pcc_cell': pcc_cell,
        'pcc_protein_mean': pcc_protein_mean,
        'pcc_protein_median': np.nanmedian(pcc_protein),
        'pcc_cell_mean': pcc_cell_mean,
        'pcc_cell_median': np.nanmedian(pcc_cell),
        'rmse': rmse,
        'cmd_cell': cmd_cell,
        'cmd_protein': cmd_protein,
        'mmd': mmd
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"PCC (Protein-protein): mean = {pcc_protein_mean:.4f}, median = {np.nanmedian(pcc_protein):.4f}")
        print(f"PCC (Cell-cell):       mean = {pcc_cell_mean:.4f}, median = {np.nanmedian(pcc_cell):.4f}")
        print(f"CMD (Cell-cell):       {cmd_cell:.4f}")
        print(f"CMD (Protein-protein): {cmd_protein:.4f}")
        print(f"RMSE:                  {rmse:.4f}")
        print(f"MMD:                   {mmd:.4f}")
        print("=" * 50 + "\n")
    
    return results


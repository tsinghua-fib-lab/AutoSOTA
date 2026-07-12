# -*- coding: utf-8 -*-
"""
scBridge-Flow evaluation metrics
Contains: PCC (protein-protein, cell-cell), RMSE
"""

import numpy as np
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
    Compute PCC (Pearson Correlation Coefficient)
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values [N, M]
    true : np.ndarray
        Ground truth values [N, M]
    protein_names : list, optional
        List of protein names
    cell_ids : list, optional
        List of cell IDs
        
    Returns
    ----------
    pcc_protein : np.ndarray
        PCC for each protein [M]
    pcc_cell : np.ndarray
        PCC for each cell [N]
    pcc_protein_mean : float
        Mean PCC across proteins
    pcc_cell_mean : float
        Mean PCC across cells
    """
    n_cells, n_proteins = pred.shape
    
    # Protein-protein PCC (compute for each protein)
    pcc_protein = []
    for j in range(n_proteins):
        x = pred[:, j]
        y = true[:, j]
        # Check if valid data exists
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


# def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
#     """
#     Compute RMSE (Root Mean Squared Error), calculated after sc.pp.scale normalization

#     Parameters
#     ----------
#     pred : np.ndarray
#         Predicted values [N, M]
#     true : np.ndarray
#         Ground truth values [N, M]

#     Returns
#     ----------
#     rmse : float
#         RMSE value
#     """
#     import scanpy as sc
#     import anndata

#     # Wrap data as AnnData for scanpy scale
#     pred_adata = anndata.AnnData(pred.copy())
#     true_adata = anndata.AnnData(true.copy())

#     sc.pp.scale(pred_adata, zero_center=True, copy=False)
#     sc.pp.scale(true_adata, zero_center=True, copy=False)

#     pred_scaled = pred_adata.X
#     true_scaled = true_adata.X

#     rmse = np.sqrt(metrics.mean_squared_error(
#         true_scaled.flatten(),
#         pred_scaled.flatten()
#     ))
#     return rmse

def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute RMSE. pred and true are already in log-normalized space
    """
    rmse = np.sqrt(metrics.mean_squared_error(
        true.flatten(),
        pred.flatten()
    ))
    return rmse

def cmd_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute CMD (Correlation Matrix Distance)
    
    Parameters
    ----------
    A : np.ndarray
        Correlation matrix A
    B : np.ndarray
        Correlation matrix B
        
    Returns
    ----------
    cmd : float
        CMD distance
    """
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - a / (b + 1e-8)


def compute_cmd(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    Compute CMD (compute separately for cells and proteins)
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values [N, M]
    true : np.ndarray
        Ground truth values [N, M]
        
    Returns
    ----------
    cmd_cell : float
        Cell CMD distance
    cmd_protein : float
        Protein CMD distance
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have the same shape, got {pred.shape} vs {true.shape}")
    if pred.ndim != 2:
        raise ValueError(f"pred and true must be 2D arrays, got ndim={pred.ndim}")

    n_cells, n_proteins = pred.shape

    # Cell-cell CMD: avoid materializing N x N correlation matrices.
    # If Z is row-standardized data scaled by sqrt(M-1), corr_cell = Z Z^T.
    # Then <A, B> = ||Z_pred^T Z_true||_F^2 and ||A||_F = ||Z_pred^T Z_pred||_F.
    if n_proteins > 1:
        pred_centered = pred - pred.mean(axis=1, keepdims=True)
        true_centered = true - true.mean(axis=1, keepdims=True)

        pred_std = pred.std(axis=1, ddof=1, keepdims=True)
        true_std = true.std(axis=1, ddof=1, keepdims=True)

        pred_z = np.divide(
            pred_centered,
            pred_std,
            out=np.zeros_like(pred_centered),
            where=pred_std > 1e-12
        ) / np.sqrt(n_proteins - 1)
        true_z = np.divide(
            true_centered,
            true_std,
            out=np.zeros_like(true_centered),
            where=true_std > 1e-12
        ) / np.sqrt(n_proteins - 1)

        cross = pred_z.T @ true_z
        pred_self = pred_z.T @ pred_z
        true_self = true_z.T @ true_z

        numerator = np.sum(cross * cross)
        denominator = np.linalg.norm(pred_self, 'fro') * np.linalg.norm(true_self, 'fro')
        cmd_cell = 1 - numerator / (denominator + 1e-8)
    else:
        cmd_cell = 0.0

    # Protein-protein CMD
    # M x M matrix is usually manageable, and this path is numerically stable.
    pred_prot_corr = np.corrcoef(pred, rowvar=False)
    true_prot_corr = np.corrcoef(true, rowvar=False)

    pred_prot_corr = np.nan_to_num(pred_prot_corr, nan=0.0)
    true_prot_corr = np.nan_to_num(true_prot_corr, nan=0.0)

    cmd_protein = cmd_dist(pred_prot_corr, true_prot_corr)
    
    return cmd_cell, cmd_protein

def compute_mmd(x: np.ndarray, y: np.ndarray, kernel='rbf') -> float:
    """
    Compute MMD (Maximum Mean Discrepancy)
    
    Parameters
    ----------
    x : np.ndarray
        Ground truth data [N, D]
    y : np.ndarray
        Predicted/generated data [N, D]
    kernel : str
        Kernel function type
        
    Returns
    ----------
    mmd : float
        MMD distance
    """
    import torch
    
    # Convert to torch tensor
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    
    x_kernel = compute_kernel(x_torch, x_torch)
    y_kernel = compute_kernel(y_torch, y_torch)
    xy_kernel = compute_kernel(x_torch, y_torch)
    
    mmd_value = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_value.item()

def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute RBF kernel matrix
    
    Parameters
    ----------
    x : torch.Tensor
        Data [N, D]
    y : torch.Tensor
        Data [M, D]
        
    Returns
    ----------
    kernel : torch.Tensor
        Kernel matrix [N, M]
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
    Comprehensive evaluation of prediction results
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values [N, M]
    true : np.ndarray
        Ground truth values [N, M]
    protein_names : list, optional
        List of protein names
    verbose : bool
        Whether to print results
        
    Returns
    ----------
    results : dict
        Dictionary containing all metrics
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

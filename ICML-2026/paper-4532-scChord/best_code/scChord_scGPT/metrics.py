# -*- coding: utf-8 -*-
"""
scBridge-Flow evaluation metrics: PCC (protein-wise and cell-wise), RMSE, CMD, MMD.
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
    Pearson correlation: per-protein and per-cell PCC.

    Parameters
    ----------
    pred : np.ndarray
        Predictions [N, M].
    true : np.ndarray
        Ground truth [N, M].
    protein_names : list, optional
        Protein names (unused in computation; kept for API compatibility).
    cell_ids : list, optional
        Cell IDs (unused in computation; kept for API compatibility).

    Returns
    -------
    pcc_protein : np.ndarray
        PCC per protein [M].
    pcc_cell : np.ndarray
        PCC per cell [N].
    pcc_protein_mean : float
        Mean of per-protein PCC.
    pcc_cell_mean : float
        Mean of per-cell PCC.
    """
    n_cells, n_proteins = pred.shape

    # Protein-wise PCC
    pcc_protein = []
    for j in range(n_proteins):
        x = pred[:, j]
        y = true[:, j]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            pcc, _ = pearsonr(x, y)
        else:
            pcc = 0.0
        pcc_protein.append(pcc)
    pcc_protein = np.array(pcc_protein)

    # Cell-wise PCC
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
#     RMSE after scanpy scale (zero-center; legacy experiment).
#
#     Parameters
#     ----------
#     pred : np.ndarray
#         Predictions [N, M].
#     true : np.ndarray
#         Ground truth [N, M].
#
#     Returns
#     -------
#     rmse : float
#     """
#     import scanpy as sc
#     import anndata
#
#     pred_adata = anndata.AnnData(pred.copy())
#     true_adata = anndata.AnnData(true.copy())
#
#     sc.pp.scale(pred_adata, zero_center=True, copy=False)
#     sc.pp.scale(true_adata, zero_center=True, copy=False)
#
#     pred_scaled = pred_adata.X
#     true_scaled = true_adata.X
#
#     rmse = np.sqrt(metrics.mean_squared_error(
#         true_scaled.flatten(),
#         pred_scaled.flatten()
#     ))
#     return rmse

def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    RMSE in log-normalized space (pred and true already aligned).
    """
    rmse = np.sqrt(metrics.mean_squared_error(
        true.flatten(),
        pred.flatten()
    ))
    return rmse

def cmd_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Correlation matrix distance between two correlation matrices.

    Parameters
    ----------
    A : np.ndarray
        Correlation matrix A.
    B : np.ndarray
        Correlation matrix B.

    Returns
    -------
    cmd : float
        CMD value.
    """
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - a / (b + 1e-8)


def compute_cmd(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    CMD on cell-cell and protein-protein correlation matrices.

    Parameters
    ----------
    pred : np.ndarray
        Predictions [N, M].
    true : np.ndarray
        Ground truth [N, M].

    Returns
    -------
    cmd_cell : float
        Cell-level CMD.
    cmd_protein : float
        Protein-level CMD.
    """
    pred_df = pd.DataFrame(pred)
    true_df = pd.DataFrame(true)

    # Cell-cell CMD
    pred_cell_corr = pred_df.T.corr()
    true_cell_corr = true_df.T.corr()

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
    Maximum Mean Discrepancy (RBF kernel).

    Parameters
    ----------
    x : np.ndarray
        Reference samples [N, D].
    y : np.ndarray
        Generated samples [N, D].
    kernel : str
        Kernel type (only 'rbf' used in implementation).

    Returns
    -------
    mmd : float
        MMD value.
    """
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()

    x_kernel = compute_kernel(x_torch, x_torch)
    y_kernel = compute_kernel(y_torch, y_torch)
    xy_kernel = compute_kernel(x_torch, y_torch)

    mmd_value = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_value.item()

def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    RBF kernel matrix K(x, y).

    Parameters
    ----------
    x : torch.Tensor
        [N, D].
    y : torch.Tensor
        [M, D].

    Returns
    -------
    kernel : torch.Tensor
        [N, M].
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
    Full metric bundle: PCC, RMSE, CMD, MMD.

    Parameters
    ----------
    pred : np.ndarray
        Predictions [N, M].
    true : np.ndarray
        Ground truth [N, M].
    protein_names : list, optional
        Protein names (passed to compute_pcc).
    verbose : bool
        Print summary table.

    Returns
    -------
    results : dict
        All metric arrays and scalars.
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

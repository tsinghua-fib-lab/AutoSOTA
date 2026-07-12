# -*- coding: utf-8 -*-
"""
scBridge-Flow data utilities:

1. Load single-cell RNA–protein multi-omic AnnData
2. HVG selection and preprocessing
3. PyTorch Dataset construction
"""

import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict


def log_normalize(data: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """
    Log-normalize counts: log(count / total * target_sum + 1).

    Parameters
    ----------
    data : np.ndarray
        Raw count matrix [N, D].
    target_sum : float
        Target sum for normalization (default 1e4).

    Returns
    -------
    np.ndarray
        Log-normalized values.
    """
    row_sums = np.sum(data, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8  # avoid division by zero
    return np.log(data / row_sums * target_sum + 1.)

def preprocess_protein(prot_counts: np.ndarray) -> np.ndarray:
    """
    Protein preprocessing: log-normalization.

    Parameters
    ----------
    prot_counts : np.ndarray
        Raw protein counts [N, M].

    Returns
    -------
    prot_norm : np.ndarray
        Preprocessed protein matrix [N, M].
    """
    return log_normalize(prot_counts, target_sum=1e4).astype(np.float32)

def preprocess_rna(
    adata,
    n_top_genes: int = 1000,
    target_sum: float = 1e4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RNA preprocessing: HVG selection and log-normalization.

    Parameters
    ----------
    adata : AnnData
        Input object.
    n_top_genes : int
        Number of highly variable genes.
    target_sum : float
        Normalization target sum.

    Returns
    -------
    rna_norm : np.ndarray
        Preprocessed RNA [N, G].
    hvg_mask : np.ndarray
        Boolean mask of HVGs on full adata.var.
    """
    sc.pp.filter_genes(adata, min_cells=1)

    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=target_sum)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes)

    hvg_mask = adata_temp.var.highly_variable
    adata_hvg = adata[:, hvg_mask]

    X = adata_hvg.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    rna_norm = log_normalize(X, target_sum=target_sum)

    return rna_norm.astype(np.float32), hvg_mask




class SingleCellDataset(Dataset):
    """
    Single-cell RNA–protein dataset.

    Each item is a dict with:
        - rna_norm: [G] preprocessed RNA
        - prot_norm: [M] preprocessed protein (for VAE training)
        - prot_raw: [M] raw protein counts
        - batch_id: int batch index
    """

    def __init__(
        self,
        rna_norm: np.ndarray,
        prot_norm: np.ndarray,
        prot_raw: np.ndarray,
        batch_ids: np.ndarray
    ):
        """
        Parameters
        ----------
        rna_norm : np.ndarray
            Preprocessed RNA [N, G].
        prot_norm : np.ndarray
            Preprocessed protein [N, M].
        prot_raw : np.ndarray
            Raw protein counts [N, M].
        batch_ids : np.ndarray
            Batch indices [N].
        """
        self.rna_norm = torch.from_numpy(rna_norm).float()
        self.prot_norm = torch.from_numpy(prot_norm).float()
        self.prot_raw = torch.from_numpy(prot_raw).float()
        self.batch_ids = torch.from_numpy(batch_ids).long()

    def __len__(self):
        return len(self.rna_norm)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'rna_norm': self.rna_norm[idx],
            'prot_norm': self.prot_norm[idx],
            'prot_raw': self.prot_raw[idx],
            'batch_id': self.batch_ids[idx]
        }


def load_data(
    data_path: str,
    n_top_genes: int = 1000,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[SingleCellDataset, SingleCellDataset, Dict]:
    """
    Load one H5AD, preprocess, and split train/test randomly.

    Parameters
    ----------
    data_path : str
        Path to .h5ad file.
    n_top_genes : int
        Number of HVGs.
    train_ratio : float
        Fraction of cells for training.
    random_state : int
        RNG seed.

    Returns
    -------
    train_dataset : SingleCellDataset
    test_dataset : SingleCellDataset
    data_info : dict
        Gene/protein names, indices, batch mapping, etc.
    """
    print(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)

    if 'batch_id' in adata.obs.columns:
        batch_col = adata.obs['batch_id']

        if pd.api.types.is_numeric_dtype(batch_col):
            batch_ids = batch_col.to_numpy(dtype=np.int64, copy=False)
            batch_mapping = None
        else:
            if pd.api.types.is_categorical_dtype(batch_col):
                batch_col = batch_col.astype(object).where(batch_col.notna(), 'unknown').astype(str)
            else:
                batch_col = batch_col.fillna('unknown').astype(str)
            batch_cat = pd.Categorical(batch_col)
            batch_ids = batch_cat.codes.astype(np.int64)
            batch_mapping = dict(enumerate(batch_cat.categories.astype(str)))
    else:
        batch_ids = np.zeros(adata.n_obs, dtype=np.int64)
        batch_mapping = None

    if 'protein_expression' in adata.obsm:
        prot_df = adata.obsm['protein_expression']
        prot_raw = prot_df.values.astype(np.float32)
        protein_names = list(prot_df.columns)
    else:
        raise ValueError("No protein expression found in adata.obsm['protein_expression']")

    np.random.seed(random_state)
    n_cells = adata.n_obs
    indices = np.random.permutation(n_cells)
    n_train = int(n_cells * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_adata = adata[train_idx].copy()

    sc.pp.filter_genes(train_adata, min_cells=1)
    adata_temp = train_adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes)
    hvg_genes = adata_temp.var_names[adata_temp.var.highly_variable]

    common_genes = [g for g in hvg_genes if g in adata.var_names]
    adata_hvg = adata[:, common_genes]

    X = adata_hvg.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    rna_norm = log_normalize(X, target_sum=1e4).astype(np.float32)

    prot_norm = preprocess_protein(prot_raw)

    train_dataset = SingleCellDataset(
        rna_norm=rna_norm[train_idx],
        prot_norm=prot_norm[train_idx],
        prot_raw=prot_raw[train_idx],
        batch_ids=batch_ids[train_idx]
    )

    test_dataset = SingleCellDataset(
        rna_norm=rna_norm[test_idx],
        prot_norm=prot_norm[test_idx],
        prot_raw=prot_raw[test_idx],
        batch_ids=batch_ids[test_idx]
    )

    data_info = {
        'n_genes': len(common_genes),
        'n_proteins': len(protein_names),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'gene_names': list(common_genes),
        'protein_names': protein_names,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'batch_mapping': batch_mapping,
    }

    print(f"Data loaded: {data_info['n_train']} train, {data_info['n_test']} test")
    print(f"Features: {data_info['n_genes']} genes (HVG), {data_info['n_proteins']} proteins")

    return train_dataset, test_dataset, data_info


def load_data_cross_dataset(
    train_path: str,
    test_path: str,
    n_top_genes: int = 1000
) -> Tuple[SingleCellDataset, SingleCellDataset, Dict]:
    """
    Cross-dataset split: all cells from train_path for training, test_path for testing.

    Parameters
    ----------
    train_path : str
        Training .h5ad.
    test_path : str
        Test .h5ad.
    n_top_genes : int
        HVG count (selected on training data).

    Returns
    -------
    train_dataset : SingleCellDataset
    test_dataset : SingleCellDataset
    data_info : dict
    """
    print(f"Loading training data from {train_path}...")
    train_adata = sc.read_h5ad(train_path)

    print(f"Loading test data from {test_path}...")
    test_adata = sc.read_h5ad(test_path)

    train_batch_ids = np.zeros(train_adata.n_obs, dtype=np.int64)
    test_batch_ids = np.zeros(test_adata.n_obs, dtype=np.int64)

    if 'protein_expression' not in train_adata.obsm:
        raise ValueError("No protein expression found in train adata.obsm['protein_expression']")
    if 'protein_expression' not in test_adata.obsm:
        raise ValueError("No protein expression found in test adata.obsm['protein_expression']")

    train_prot_df = train_adata.obsm['protein_expression']
    test_prot_df = test_adata.obsm['protein_expression']

    train_prot_df.columns = train_prot_df.columns.str.replace("-", "_").str.replace(".", "_")
    test_prot_df.columns = test_prot_df.columns.str.replace("-", "_").str.replace(".", "_")

    common_proteins = list(train_prot_df.columns.intersection(test_prot_df.columns))
    print(f"Common proteins: {len(common_proteins)}")

    train_prot_raw = train_prot_df[common_proteins].values.astype(np.float32)
    test_prot_raw = test_prot_df[common_proteins].values.astype(np.float32)

    sc.pp.filter_genes(train_adata, min_cells=1)
    adata_temp = train_adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes)
    hvg_genes = list(adata_temp.var_names[adata_temp.var.highly_variable])

    common_genes = [g for g in hvg_genes if g in test_adata.var_names]
    print(f"Common HVG genes: {len(common_genes)}")

    train_adata_hvg = train_adata[:, common_genes]
    test_adata_hvg = test_adata[:, common_genes]

    train_X = train_adata_hvg.X
    if scipy.sparse.issparse(train_X):
        train_X = train_X.toarray()
    train_rna_norm = log_normalize(train_X, target_sum=1e4).astype(np.float32)

    test_X = test_adata_hvg.X
    if scipy.sparse.issparse(test_X):
        test_X = test_X.toarray()
    test_rna_norm = log_normalize(test_X, target_sum=1e4).astype(np.float32)

    train_prot_norm = preprocess_protein(train_prot_raw)
    test_prot_norm = preprocess_protein(test_prot_raw)

    train_dataset = SingleCellDataset(
        rna_norm=train_rna_norm,
        prot_norm=train_prot_norm,
        prot_raw=train_prot_raw,
        batch_ids=train_batch_ids
    )

    test_dataset = SingleCellDataset(
        rna_norm=test_rna_norm,
        prot_norm=test_prot_norm,
        prot_raw=test_prot_raw,
        batch_ids=test_batch_ids
    )

    data_info = {
        'n_genes': len(common_genes),
        'n_proteins': len(common_proteins),
        'n_train': train_adata.n_obs,
        'n_test': test_adata.n_obs,
        'gene_names': common_genes,
        'protein_names': common_proteins,
        'train_path': train_path,
        'test_path': test_path,
        'train_obs_names': list(train_adata.obs_names),
        'test_obs_names': list(test_adata.obs_names),
    }

    print(f"Data loaded: {data_info['n_train']} train, {data_info['n_test']} test")
    print(f"Features: {data_info['n_genes']} genes (HVG), {data_info['n_proteins']} proteins")

    return train_dataset, test_dataset, data_info


def get_dataloader(
    dataset: SingleCellDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True
) -> DataLoader:
    """Build a DataLoader for SingleCellDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

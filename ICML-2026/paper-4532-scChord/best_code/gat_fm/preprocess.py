"""
Data preprocessing helpers for GAT-FM.

This module provides utilities to:
1. Validate required AnnData fields
2. Normalize field names for training/inference
3. Merge datasets with different protein panels
4. Create placeholder RNA embeddings for quick testing
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def validate_adata(adata: AnnData) -> Dict[str, bool]:
    """
    Validate whether required fields exist for GAT-FM.

    Required fields:
    - adata.obsm['protein_expression']: (N, P) expression matrix
    - adata.obsm['protein_mask']: (N, P) observation mask
    - adata.obs['dataset_id']: (N,) dataset labels
    - adata.uns['protein_names']: list of protein names
    """
    results = {
        'has_protein_expression': 'protein_expression' in adata.obsm,
        'has_protein_mask': 'protein_mask' in adata.obsm,
        'has_dataset_id': 'dataset_id' in adata.obs,
        'has_protein_names': 'protein_names' in adata.uns,
    }

    if results['has_protein_expression'] and results['has_protein_mask']:
        expr_shape = adata.obsm['protein_expression'].shape
        mask_shape = adata.obsm['protein_mask'].shape
        results['shapes_match'] = expr_shape == mask_shape
    else:
        results['shapes_match'] = False

    return results


def prepare_adata_for_training(
    adata: AnnData,
    protein_layer_key: str = None,
    mask_layer_key: str = None,
    dataset_id: Union[str, int] = 0,
    protein_names: List[str] = None,
) -> AnnData:
    """
    Normalize AnnData structure to GAT-FM expected fields.
    """
    adata = adata.copy()

    if protein_layer_key is not None and protein_layer_key in adata.obsm:
        adata.obsm['protein_expression'] = adata.obsm[protein_layer_key].astype(np.float32)
    elif 'protein_expression' not in adata.obsm:
        raise ValueError("Protein expression not found. Specify protein_layer_key.")

    if mask_layer_key is not None and mask_layer_key in adata.obsm:
        adata.obsm['protein_mask'] = adata.obsm[mask_layer_key].astype(np.float32)
    elif 'protein_mask' not in adata.obsm:
        adata.obsm['protein_mask'] = np.ones_like(
            adata.obsm['protein_expression'], dtype=np.float32
        )

    if 'dataset_id' not in adata.obs:
        adata.obs['dataset_id'] = pd.Categorical([dataset_id] * len(adata))

    if protein_names is not None:
        adata.uns['protein_names'] = protein_names
    elif 'protein_names' not in adata.uns:
        n_proteins = adata.obsm['protein_expression'].shape[1]
        adata.uns['protein_names'] = [f'protein_{i}' for i in range(n_proteins)]

    return adata


def merge_datasets(
    adatas: List[AnnData],
    dataset_ids: Optional[List[Union[str, int]]] = None,
) -> AnnData:
    """
    Merge multiple AnnData objects with different protein panels.

    The merged output contains the union of proteins and per-cell masks.
    """
    if dataset_ids is None:
        dataset_ids = list(range(len(adatas)))

    all_proteins = set()
    protein_sets = []
    for adata in adatas:
        if 'protein_names' in adata.uns:
            proteins = set(adata.uns['protein_names'])
        else:
            n = adata.obsm['protein_expression'].shape[1]
            proteins = set(f'protein_{i}' for i in range(n))
        protein_sets.append(proteins)
        all_proteins.update(proteins)

    union_proteins = sorted(list(all_proteins))
    protein_to_idx = {p: i for i, p in enumerate(union_proteins)}
    n_union = len(union_proteins)

    merged_expr = []
    merged_mask = []
    merged_dataset_id = []

    for adata, proteins, did in zip(adatas, protein_sets, dataset_ids):
        n_cells = len(adata)
        expr = np.zeros((n_cells, n_union), dtype=np.float32)
        mask = np.zeros((n_cells, n_union), dtype=np.float32)

        original_proteins = list(proteins)
        if 'protein_names' in adata.uns:
            original_proteins = list(adata.uns['protein_names'])

        for j, pname in enumerate(original_proteins):
            union_idx = protein_to_idx[pname]
            expr[:, union_idx] = adata.obsm['protein_expression'][:, j]
            if 'protein_mask' in adata.obsm:
                mask[:, union_idx] = adata.obsm['protein_mask'][:, j]
            else:
                mask[:, union_idx] = 1.0

        merged_expr.append(expr)
        merged_mask.append(mask)
        merged_dataset_id.extend([did] * n_cells)

    merged_expr = np.concatenate(merged_expr, axis=0)
    merged_mask = np.concatenate(merged_mask, axis=0)
    merged_dataset_id = pd.Categorical(merged_dataset_id)

    merged = AnnData(
        X=np.zeros((len(merged_dataset_id), 1)),
        obs=pd.DataFrame({'dataset_id': merged_dataset_id}),
    )
    merged.obsm['protein_expression'] = merged_expr
    merged.obsm['protein_mask'] = merged_mask
    merged.uns['protein_names'] = union_proteins

    return merged


def create_protein_to_ppi_mapping(
    protein_names: List[str],
    ppi_df: pd.DataFrame,
    protein_col1: str = 'protein1',
    protein_col2: str = 'protein2',
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Build protein index mapping and adjacency matrix from a PPI dataframe.
    """
    protein_to_idx = {name: idx for idx, name in enumerate(protein_names)}
    n = len(protein_names)

    adjacency = np.zeros((n, n), dtype=np.float32)

    for _, row in ppi_df.iterrows():
        p1, p2 = row[protein_col1], row[protein_col2]
        if p1 in protein_to_idx and p2 in protein_to_idx:
            i, j = protein_to_idx[p1], protein_to_idx[p2]
            adjacency[i, j] = 1.0

    return protein_to_idx, adjacency


def extract_rna_embeddings_placeholder(
    adata: AnnData,
    output_path: str,
    embedding_dim: int = 512,
) -> np.ndarray:
    """
    Create random RNA embeddings for testing only.

    Replace this with a real RNA encoder (for example scGPT) in production.
    """
    n_cells = len(adata)

    embeddings = np.random.randn(n_cells, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(output_path, embeddings)

    print(f"Saved placeholder RNA embeddings: {output_path}")
    print(f"Shape: {embeddings.shape}")

    return embeddings


def preprocess_for_training(
    adata_path: str,
    rna_embed_path: str,
    output_path: str,
    ppi_path: Optional[str] = None,
    dataset_id: Union[str, int] = 0,
):
    """
    End-to-end preprocessing pipeline for one dataset.
    """
    del ppi_path  # API compatibility

    print(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    adata = prepare_adata_for_training(adata, dataset_id=dataset_id)

    validation = validate_adata(adata)
    print(f"Validation result: {validation}")

    if not all(validation.values()):
        raise ValueError(f"AnnData validation failed: {validation}")

    rna_embed_path = Path(rna_embed_path)
    if not rna_embed_path.exists():
        print("RNA embeddings not found. Generating placeholder embeddings...")
        extract_rna_embeddings_placeholder(adata, str(rna_embed_path))

    adata.write_h5ad(output_path)
    print(f"Saved preprocessed AnnData: {output_path}")

    print("\nDataset summary:")
    print(f"  Cells: {len(adata)}")
    print(f"  Proteins: {adata.obsm['protein_expression'].shape[1]}")
    print(f"  Protein observation ratio: {adata.obsm['protein_mask'].mean():.2%}")


def split_adata(
    adata: AnnData,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[AnnData, AnnData, AnnData]:
    """
    Split AnnData into train/validation/test sets.
    """
    n = len(adata)
    indices = np.random.RandomState(seed).permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return adata[train_idx].copy(), adata[val_idx].copy(), adata[test_idx].copy()

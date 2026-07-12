"""
Data Loading and Processing for GAT-FM.

This module handles:
1. Loading AnnData (.h5ad) files
2. Z-score normalization with mean/std caching
3. PyTorch Dataset and DataLoader creation
4. Multi-dataset handling with mosaic masks

Data format conventions (from GAT-Diffusion-Protein-Prediction.md):
- Protein expression: adata.obsm["protein_expression"] (N, P_union)
- Protein mask: adata.obsm["protein_mask"] (N, P_union), 1=observed
- Dataset ID: adata.obs["dataset_id"] (N,) categorical
- RNA embeddings: separate .npy files (N, 512)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NormalizationStats:
    """Statistics for z-score normalization."""
    mean: np.ndarray  # (P,)
    std: np.ndarray   # (P,)
    
    def save(self, path: str):
        """Save normalization stats to file."""
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> 'NormalizationStats':
        """Load normalization stats from file."""
        data = np.load(path)
        return cls(mean=data['mean'], std=data['std'])


def compute_normalization_stats(
    protein_expr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> NormalizationStats:
    """
    Compute z-score normalization statistics.
    
    Args:
        protein_expr: (N, P) protein expression matrix
        mask: (N, P) observation mask, 1=observed (optional)
        
    Returns:
        NormalizationStats with mean and std per protein
    """
    if mask is not None:
        # Compute stats only on observed values
        masked_expr = np.ma.array(protein_expr, mask=(mask == 0))
        mean = np.ma.mean(masked_expr, axis=0).filled(0)
        std = np.ma.std(masked_expr, axis=0).filled(1)
    else:
        mean = np.mean(protein_expr, axis=0)
        std = np.std(protein_expr, axis=0)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    
    return NormalizationStats(mean=mean, std=std)


def normalize_protein_expression(
    protein_expr: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """
    Apply z-score normalization.
    
    Args:
        protein_expr: (N, P) or (P,) protein expression
        stats: Normalization statistics
        
    Returns:
        Normalized protein expression
    """
    return (protein_expr - stats.mean) / stats.std


def denormalize_protein_expression(
    normalized_expr: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """
    Reverse z-score normalization.
    
    Args:
        normalized_expr: (N, P) or (P,) normalized expression
        stats: Normalization statistics
        
    Returns:
        Original scale protein expression
    """
    return normalized_expr * stats.std + stats.mean


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein prediction task.
    
    Loads data from AnnData and pre-computed RNA embeddings.
    """
    
    def __init__(
        self,
        adata_path: str,
        rna_embed_path: str,
        normalize: bool = True,
        norm_stats: Optional[NormalizationStats] = None,
        compute_stats: bool = False,
    ):
        """
        Args:
            adata_path: Path to .h5ad file
            rna_embed_path: Path to .npy file with RNA embeddings
            normalize: Whether to normalize protein expression
            norm_stats: Pre-computed normalization stats (optional)
            compute_stats: Whether to compute stats from this dataset
        """
        super().__init__()
        
        # Load AnnData
        self.adata = sc.read_h5ad(adata_path)
        
        # Extract protein expression
        if 'protein_expression' in self.adata.obsm:
            protein_expr_data = self.adata.obsm['protein_expression']
            # If this is a DataFrame, keep the original protein names from columns.
            if hasattr(protein_expr_data, 'columns'):
                self._protein_names_from_expr = list(protein_expr_data.columns)
            else:
                self._protein_names_from_expr = None
            # Convert to a numpy array for downstream tensor conversion.
            if hasattr(protein_expr_data, 'values'):
                self.protein_expr = protein_expr_data.values.astype(np.float32)
            else:
                self.protein_expr = np.asarray(protein_expr_data).astype(np.float32)
        else:
            raise ValueError(f"protein_expression not found in {adata_path}")
        
        # Extract protein mask
        if 'protein_mask' in self.adata.obsm:
            self.protein_mask = self.adata.obsm['protein_mask'].astype(np.float32)
        else:
            # Default: all proteins observed
            self.protein_mask = np.ones_like(self.protein_expr, dtype=np.float32)
        
        # Extract dataset_id
        if 'dataset_id' in self.adata.obs:
            dataset_ids = self.adata.obs['dataset_id']
            if hasattr(dataset_ids, 'cat'):
                self.dataset_id = dataset_ids.cat.codes.values.astype(np.int64)
            else:
                self.dataset_id = dataset_ids.values.astype(np.int64)
        else:
            # Default: single dataset
            self.dataset_id = np.zeros(len(self.adata), dtype=np.int64)
        
        # Load RNA embeddings
        self.rna_embed = np.load(rna_embed_path).astype(np.float32)
        
        # Validate shapes
        assert len(self.protein_expr) == len(self.rna_embed), \
            f"Mismatch: {len(self.protein_expr)} cells vs {len(self.rna_embed)} RNA embeddings"
        
        # Handle normalization
        self.normalize = normalize
        if normalize:
            if compute_stats or norm_stats is None:
                self.norm_stats = compute_normalization_stats(
                    self.protein_expr, self.protein_mask
                )
            else:
                self.norm_stats = norm_stats
            self.protein_expr_normalized = normalize_protein_expression(
                self.protein_expr, self.norm_stats
            )
        else:
            self.norm_stats = None
            self.protein_expr_normalized = self.protein_expr
        
        # Store metadata
        self.num_cells = len(self.adata)
        self.num_proteins = self.protein_expr.shape[1]
        self.rna_embed_dim = self.rna_embed.shape[1]
    
    def __len__(self) -> int:
        return self.num_cells
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict with keys:
                - protein_expr: (P,) normalized protein expression (target x1)
                - protein_mask: (P,) observation mask
                - rna_embed: (512,) RNA embedding
                - dataset_id: scalar dataset index
        """
        # Be robust to both numpy arrays and pandas objects.
        # NOTE: pandas.DataFrame.__getitem__(int) treats the int as a *column label*.
        # This can crash inside DataLoader workers with KeyError when idx is not a column.
        expr_src = self.protein_expr_normalized
        mask_src = self.protein_mask

        if hasattr(expr_src, "iloc"):
            # DataFrame/Series: always use row-based indexing
            protein_expr = expr_src.iloc[int(idx)]
            # DataFrame row -> Series; Series -> scalar/array
            if hasattr(protein_expr, "to_numpy"):
                protein_expr = protein_expr.to_numpy()
            else:
                protein_expr = np.asarray(protein_expr)
        else:
            protein_expr = np.asarray(expr_src[int(idx)])

        if hasattr(mask_src, "iloc"):
            protein_mask = mask_src.iloc[int(idx)]
            if hasattr(protein_mask, "to_numpy"):
                protein_mask = protein_mask.to_numpy()
            else:
                protein_mask = np.asarray(protein_mask)
        else:
            protein_mask = np.asarray(mask_src[int(idx)])
            
        return {
            'protein_expr': torch.from_numpy(protein_expr.astype('float32')),
            'protein_mask': torch.from_numpy(protein_mask.astype('float32')),
            'rna_embed': torch.from_numpy(self.rna_embed[idx]),
            'dataset_id': torch.tensor(self.dataset_id[idx], dtype=torch.long),
        }
    
    def get_protein_names(self) -> List[str]:
        """Get protein names from obsm['protein_expression'] column labels."""
        # Prefer names cached during dataset initialization.
        if hasattr(self, '_protein_names_from_expr') and self._protein_names_from_expr is not None:
            return self._protein_names_from_expr
        
        # Fallback to reading labels directly from obsm at call time.
        if 'protein_expression' in self.adata.obsm:
            protein_expr = self.adata.obsm['protein_expression']
            if hasattr(protein_expr, 'columns'):
                return list(protein_expr.columns)
        
        # Final fallback to deterministic placeholder names.
        return [f'protein_{i}' for i in range(self.num_proteins)]


class MultiDatasetLoader:
    """
    Handles loading and merging multiple datasets with mosaic panels.
    
    Creates a unified dataset with:
    - Union of all proteins across datasets
    - Per-cell masks indicating which proteins were measured
    - Dataset IDs for batch effect modeling
    """
    
    def __init__(
        self,
        dataset_configs: List[Dict[str, str]],
        normalize: bool = True,
    ):
        """
        Args:
            dataset_configs: List of dicts with 'adata_path' and 'rna_embed_path'
            normalize: Whether to normalize (uses combined stats)
        """
        self.dataset_configs = dataset_configs
        self.normalize = normalize
        self.datasets = []
        
        # Load all datasets
        for i, config in enumerate(dataset_configs):
            dataset = ProteinDataset(
                adata_path=config['adata_path'],
                rna_embed_path=config['rna_embed_path'],
                normalize=False,  # We'll normalize after merging
            )
            self.datasets.append(dataset)
        
        # For now, assume all datasets have same protein set
        # TODO: Implement union of protein panels with masking
        self.num_proteins = self.datasets[0].num_proteins
        self.rna_embed_dim = self.datasets[0].rna_embed_dim
        
        # Compute combined normalization stats if needed
        if normalize:
            all_expr = np.concatenate([d.protein_expr for d in self.datasets], axis=0)
            all_mask = np.concatenate([d.protein_mask for d in self.datasets], axis=0)
            self.norm_stats = compute_normalization_stats(all_expr, all_mask)
            
            # Apply normalization
            for dataset in self.datasets:
                dataset.protein_expr_normalized = normalize_protein_expression(
                    dataset.protein_expr, self.norm_stats
                )
                dataset.normalize = True
                dataset.norm_stats = self.norm_stats
        else:
            self.norm_stats = None
    
    def get_combined_dataset(self) -> Dataset:
        """Get combined dataset from all sources."""
        return torch.utils.data.ConcatDataset(self.datasets)
    
    def get_num_datasets(self) -> int:
        """Get number of distinct datasets."""
        return len(self.datasets)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the protein dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (for GPU)
        drop_last: Whether to drop incomplete last batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_protein_batch,
    )


def collate_protein_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for protein batches.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Batched dict with stacked tensors
    """
    return {
        'protein_expr': torch.stack([b['protein_expr'] for b in batch]),
        'protein_mask': torch.stack([b['protein_mask'] for b in batch]),
        'rna_embed': torch.stack([b['rna_embed'] for b in batch]),
        'dataset_id': torch.stack([b['dataset_id'] for b in batch]),
    }


def load_single_dataset(
    adata_path: str,
    rna_embed_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, NormalizationStats]:
    """
    Convenience function to load a single dataset with train/val/test splits.
    
    Args:
        adata_path: Path to .h5ad file
        rna_embed_path: Path to RNA embedding .npy file
        batch_size: Batch size for all loaders
        train_split: Fraction for training
        val_split: Fraction for validation (test = 1 - train - val)
        seed: Random seed for splitting
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    # Load full dataset
    full_dataset = ProteinDataset(
        adata_path=adata_path,
        rna_embed_path=rna_embed_path,
        normalize=True,
        compute_stats=True,
    )
    
    # Split indices
    n = len(full_dataset)
    indices = np.random.RandomState(seed).permutation(n)
    
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )
    test_loader = create_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )
    
    return train_loader, val_loader, test_loader, full_dataset.norm_stats

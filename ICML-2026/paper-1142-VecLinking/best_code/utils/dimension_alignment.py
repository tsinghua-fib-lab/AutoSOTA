"""
Dimension Alignment Utilities

Provides functions to align embedding dimensions when emb1 and emb2 have different dimensions.
Supports PCA dimensionality reduction and zero padding.
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger
from sklearn.decomposition import PCA


def align_dimensions_pca(
    emb1: np.ndarray,
    emb2: np.ndarray,
    target_dim: Optional[int] = None,
    method: str = "min"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align dimensions using PCA to reduce to a common dimension.

    Args:
        emb1: First embeddings (n1, d1)
        emb2: Second embeddings (n2, d2)
        target_dim: Target dimension (if None, uses method to determine)
        method: Method to determine target_dim if None:
                - "min": Use minimum of d1 and d2
                - "max": Use maximum of d1 and d2 (will pad smaller one)
                - "mean": Use mean of d1 and d2

    Returns:
        (emb1_aligned, emb2_aligned): Aligned embeddings with same dimension
    """
    d1 = emb1.shape[1]
    d2 = emb2.shape[1]

    if d1 == d2:
        logger.debug(f"Dimensions already aligned: {d1}")
        return emb1, emb2

    # Determine target dimension
    if target_dim is None:
        if method == "min":
            target_dim = min(d1, d2)
        elif method == "max":
            target_dim = max(d1, d2)
        elif method == "mean":
            target_dim = int((d1 + d2) / 2)
        else:
            raise ValueError(f"Unknown method: {method}")

    logger.debug(f"Aligning dimensions using PCA: {d1}, {d2} -> {target_dim}")

    # Apply PCA if needed
    if d1 > target_dim:
        pca1 = PCA(n_components=target_dim, random_state=42)
        emb1_aligned = pca1.fit_transform(emb1)
        explained_var = pca1.explained_variance_ratio_.sum()
        logger.debug(f"PCA on emb1: {d1} -> {target_dim} (explained variance: {explained_var:.4f})")
    elif d1 < target_dim:
        # Pad with zeros
        emb1_aligned = np.pad(emb1, ((0, 0), (0, target_dim - d1)), mode='constant')
        logger.debug(f"Zero-padded emb1: {d1} -> {target_dim}")
    else:
        emb1_aligned = emb1

    if d2 > target_dim:
        pca2 = PCA(n_components=target_dim, random_state=42)
        emb2_aligned = pca2.fit_transform(emb2)
        explained_var = pca2.explained_variance_ratio_.sum()
        logger.debug(f"PCA on emb2: {d2} -> {target_dim} (explained variance: {explained_var:.4f})")
    elif d2 < target_dim:
        # Pad with zeros
        emb2_aligned = np.pad(emb2, ((0, 0), (0, target_dim - d2)), mode='constant')
        logger.debug(f"Zero-padded emb2: {d2} -> {target_dim}")
    else:
        emb2_aligned = emb2

    return emb1_aligned, emb2_aligned


def align_dimensions_padding(
    emb1: np.ndarray,
    emb2: np.ndarray,
    padding_mode: str = "zero"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align dimensions by padding the smaller one to match the larger.

    Args:
        emb1: First embeddings (n1, d1)
        emb2: Second embeddings (n2, d2)
        padding_mode: Padding mode ('zero', 'mean', 'random')
                      - 'zero': Pad with zeros
                      - 'mean': Pad with mean of existing dimensions
                      - 'random': Pad with random values from normal distribution

    Returns:
        (emb1_aligned, emb2_aligned): Aligned embeddings with same dimension
    """
    d1 = emb1.shape[1]
    d2 = emb2.shape[1]

    if d1 == d2:
        logger.debug(f"Dimensions already aligned: {d1}")
        return emb1, emb2

    target_dim = max(d1, d2)
    logger.debug(f"Aligning dimensions using {padding_mode} padding: {d1}, {d2} -> {target_dim}")

    # Pad emb1 if needed
    if d1 < target_dim:
        pad_size = target_dim - d1
        if padding_mode == "zero":
            padding = np.zeros((emb1.shape[0], pad_size))
        elif padding_mode == "mean":
            mean_val = emb1.mean(axis=1, keepdims=True)
            padding = np.repeat(mean_val, pad_size, axis=1)
        elif padding_mode == "random":
            # Match the statistics of existing dimensions
            std = emb1.std()
            padding = np.random.normal(0, std, (emb1.shape[0], pad_size))
        else:
            raise ValueError(f"Unknown padding_mode: {padding_mode}")

        emb1_aligned = np.concatenate([emb1, padding], axis=1)
        logger.debug(f"Padded emb1: {d1} -> {target_dim}")
    else:
        emb1_aligned = emb1

    # Pad emb2 if needed
    if d2 < target_dim:
        pad_size = target_dim - d2
        if padding_mode == "zero":
            padding = np.zeros((emb2.shape[0], pad_size))
        elif padding_mode == "mean":
            mean_val = emb2.mean(axis=1, keepdims=True)
            padding = np.repeat(mean_val, pad_size, axis=1)
        elif padding_mode == "random":
            std = emb2.std()
            padding = np.random.normal(0, std, (emb2.shape[0], pad_size))
        else:
            raise ValueError(f"Unknown padding_mode: {padding_mode}")

        emb2_aligned = np.concatenate([emb2, padding], axis=1)
        logger.debug(f"Padded emb2: {d2} -> {target_dim}")
    else:
        emb2_aligned = emb2

    return emb1_aligned, emb2_aligned

def align_dimensions(
    emb1: np.ndarray,
    emb2: np.ndarray,
    method: str = "pca",
    target_dim: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal dimension alignment function that dispatches to specific methods.

    Args:
        emb1: First embeddings (n1, d1)
        emb2: Second embeddings (n2, d2)
        method: Alignment method:
                - "pca": Use PCA for reduction
                - "padding": Use zero/mean/random padding
                - "none": No alignment (return as-is)
        target_dim: Target dimension (method-specific)
        **kwargs: Additional arguments for specific methods

    Returns:
        (emb1_aligned, emb2_aligned): Aligned embeddings
    """
    if method == "none":
        logger.debug("No dimension alignment applied")
        return emb1, emb2
    elif method == "pca":
        return align_dimensions_pca(emb1, emb2, target_dim=target_dim, **kwargs)
    elif method == "padding":
        return align_dimensions_padding(emb1, emb2, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

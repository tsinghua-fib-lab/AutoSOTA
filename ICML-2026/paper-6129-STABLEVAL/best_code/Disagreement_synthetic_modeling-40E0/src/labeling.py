"""Observed label generation from annotators."""

import numpy as np
from typing import List, Tuple, Dict
from .annotators import AnnotatorPool


def generate_observed_labels(
    z: np.ndarray,
    d: np.ndarray,
    annotator_pool: AnnotatorPool,
    labels_per_item: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate observed labels from annotators.
    
    For each item (a, i):
        1. Sample m annotators without replacement
        2. For each sampled annotator r:
            - Compute adjusted confusion matrix
            - Sample observed label from categorical
    
    Args:
        z: True labels of shape (n_agents, n_items)
        d: Item ambiguity of shape (n_agents, n_items)
        annotator_pool: AnnotatorPool instance
        labels_per_item: Number of labels m per item
        rng: Random number generator
    
    Returns:
        y: Observed labels of shape (n_agents, n_items, labels_per_item)
        annotators: Annotator IDs of shape (n_agents, n_items, labels_per_item)
        label_records: List of dicts with (agent, item, annotator, label) for each observation
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_agents, n_items = z.shape
    m = labels_per_item
    
    # Initialize arrays
    y = np.zeros((n_agents, n_items, m), dtype=np.int32)
    annotators = np.zeros((n_agents, n_items, m), dtype=np.int32)
    label_records = []
    
    for a in range(n_agents):
        for i in range(n_items):
            # Sample annotators for this item
            sampled = rng.choice(
                annotator_pool.n_annotators,
                size=m,
                replace=False
            )
            annotators[a, i] = sampled
            
            # Get true label and ambiguity
            true_label = z[a, i]
            ambiguity = d[a, i]
            
            # Generate label from each annotator
            for j, r in enumerate(sampled):
                # Get adjusted confusion matrix
                adj_matrix = annotator_pool.get_adjusted_confusion_matrix(r, ambiguity)
                
                # Sample observed label
                probs = adj_matrix[true_label]
                observed = rng.choice([0, 1, 2], p=probs)
                
                y[a, i, j] = observed
                
                label_records.append({
                    "agent": a,
                    "item": i,
                    "annotator": r,
                    "label": observed,
                    "true_label": true_label,
                })
    
    return y, annotators, label_records


def subsample_labels(
    y: np.ndarray,
    annotators: np.ndarray,
    new_m: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement ranking stability test to check if ranking change by subsampling labels to fewer annotations per item.
    
    Args:
        y: Observed labels of shape (n_agents, n_items, m)
        annotators: Annotator IDs of shape (n_agents, n_items, m)
        new_m: New number of labels per item (must be <= m)
        rng: Random number generator
    
    Returns:
        y_sub: Subsampled labels of shape (n_agents, n_items, new_m)
        annotators_sub: Subsampled annotator IDs
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_agents, n_items, m = y.shape
    
    if new_m > m:
        raise ValueError(f"new_m ({new_m}) cannot exceed current m ({m})")
    
    y_sub = np.zeros((n_agents, n_items, new_m), dtype=np.int32)
    annotators_sub = np.zeros((n_agents, n_items, new_m), dtype=np.int32)
    
    for a in range(n_agents):
        for i in range(n_items):
            # Random indices to keep
            keep_idx = rng.choice(m, size=new_m, replace=False)
            keep_idx.sort()  # Keep order for reproducibility
            
            y_sub[a, i] = y[a, i, keep_idx]
            annotators_sub[a, i] = annotators[a, i, keep_idx]
    
    return y_sub, annotators_sub


def get_label_counts(y: np.ndarray) -> np.ndarray:
    """
    Get label counts per item.
    
    Args:
        y: Observed labels of shape (n_agents, n_items, m)
    
    Returns:
        counts: Array of shape (n_agents, n_items, 3) with counts for each class
    """
    n_agents, n_items, m = y.shape
    counts = np.zeros((n_agents, n_items, 3), dtype=np.int32)
    
    for c in range(3):
        counts[:, :, c] = (y == c).sum(axis=2)
    
    return counts
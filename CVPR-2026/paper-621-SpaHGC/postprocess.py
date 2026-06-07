"""Post-processing module for SpaHGC predictions."""
import numpy as np
from scipy.spatial.distance import cdist

def spatial_smoothing(predictions, embeddings, k=5, lam=0.3, tau=0.1):
    """
    Apply spatial smoothing to predictions using embedding-guided weights.
    
    Args:
        predictions: (N, G) array of predicted gene expression
        embeddings: (N, D) array of image embeddings (from UNI encoder)
        k: number of nearest neighbors
        lam: smoothing strength (0=no smoothing, 1=full smoothing)
        tau: temperature for similarity weights
    
    Returns:
        smoothed: (N, G) array of smoothed predictions
    """
    N = predictions.shape[0]
    # Compute pairwise cosine similarity of embeddings
    sim = 1 - cdist(embeddings, embeddings, metric='cosine')  # (N, N)
    
    # For each spot, find k nearest neighbors (excluding self)
    smoothed = np.zeros_like(predictions)
    for i in range(N):
        # Get similarities, set self-similarity to -inf
        s = sim[i].copy()
        s[i] = -np.inf
        # Get top k neighbors
        top_k = np.argpartition(s, -k)[-k:]
        # Compute weights
        weights = np.exp(s[top_k] / tau)
        weights /= weights.sum()
        # Weighted average of neighbor predictions
        neighbor_pred = (predictions[top_k] * weights[:, np.newaxis]).sum(axis=0)
        # Blend with original
        smoothed[i] = (1 - lam) * predictions[i] + lam * neighbor_pred
    
    return smoothed


def bilateral_smoothing(predictions, embeddings, positions, sigma_s=3.0, sigma_f=0.5):
    """
    Apply edge-preserving bilateral filtering to predictions.
    
    Args:
        predictions: (N, G) array of predicted gene expression
        embeddings: (N, D) array of image embeddings
        positions: (N, 2) array of spatial coordinates (x, y)
        sigma_s: spatial bandwidth (in spot units)
        sigma_f: feature bandwidth
    
    Returns:
        smoothed: (N, G) array of smoothed predictions
    """
    N = predictions.shape[0]
    # Spatial distances
    D_s = cdist(positions, positions, metric='euclidean')  # (N, N)
    # Feature distances (cosine distance between embeddings)
    D_f = cdist(embeddings, embeddings, metric='cosine')  # (N, N)
    
    # Bilateral weights
    W = np.exp(-D_s**2 / (2 * sigma_s**2)) * np.exp(-D_f**2 / (2 * sigma_f**2))
    # Zero out self-weight
    np.fill_diagonal(W, 0)
    # Normalize
    W /= W.sum(axis=1, keepdims=True) + 1e-8
    
    # Apply smoothing
    smoothed = W @ predictions
    return smoothed


def confidence_weighted_refinement(predictions, embeddings, k=10, low_conf_pct=20, blend_weight=0.5):
    """
    Identify low-confidence predictions and refine using high-confidence neighbors.
    
    Args:
        predictions: (N, G) array
        embeddings: (N, D) array
        k: neighbors for confidence computation
        low_conf_pct: bottom percentile to refine
        blend_weight: how much to blend with neighbor average
    
    Returns:
        refined: (N, G) array
    """
    N = predictions.shape[0]
    sim = 1 - cdist(embeddings, embeddings, metric='cosine')
    
    # Compute per-spot confidence as mean similarity to k neighbors
    confidences = np.zeros(N)
    for i in range(N):
        s = sim[i].copy()
        s[i] = -np.inf
        top_k = np.argpartition(s, -k)[-k:]
        confidences[i] = sim[i, top_k].mean()
    
    # Identify low-confidence threshold
    threshold = np.percentile(confidences, low_conf_pct)
    low_conf_mask = confidences < threshold
    
    # Refine low-confidence spots
    refined = predictions.copy()
    for i in np.where(low_conf_mask)[0]:
        s = sim[i].copy()
        s[i] = -np.inf
        # Only use high-confidence neighbors
        top_k = np.argpartition(s, -k*2)[-k*2:]
        high_conf_neighbors = [j for j in top_k if confidences[j] >= threshold]
        if len(high_conf_neighbors) > 0:
            neighbor_avg = predictions[high_conf_neighbors].mean(axis=0)
            refined[i] = (1 - blend_weight) * predictions[i] + blend_weight * neighbor_avg
    
    return refined
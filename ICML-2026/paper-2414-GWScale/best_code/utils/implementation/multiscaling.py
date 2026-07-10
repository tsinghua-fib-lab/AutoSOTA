"""
Multiscale Operations

K-means clustering and coarsening utilities for multiscale optimization.
"""

import numpy as np
import torch
from pykeops.torch import LazyTensor


def kmeans(X, M=10, centers=None, numIters=10):
    """
    K-means clustering with LazyTensor.
    
    Args:
        X: Input points, shape (N, D)
        M: Number of clusters
        centers: Initial cluster centers, shape (M, D). If None, randomly sampled from X
        numIters: Number of iterations
    
    Returns:
        torch.Tensor: Final cluster centers, shape (M, D)
        torch.Tensor: Cluster assignments for each point, shape (N,)
    """
    N, D = X.shape
    if centers is None:
        idx = torch.tensor(np.random.choice(X.shape[0], M, replace=False), dtype=torch.int64)
        centers = X[idx].clone()

    X_i = LazyTensor(X[:, None, :])  # (N, 1, D)
    centers_j = LazyTensor(centers[None, :, :])  # (1, K, D)

    clusters = None

    for i in range(numIters):
        D_ij = ((X_i - centers_j) ** 2).sum(dim=-1)
        clusters = D_ij.argmin(dim=1).long().view(-1)

        centers.zero_()
        centers.scatter_add_(0, clusters[:, None].repeat(1, D), X)

        Ncl = torch.bincount(clusters, minlength=M).type_as(clusters).view(M, 1)
        centers /= Ncl

    return centers, clusters

def coarsen(X, a, ratio):
    """
    Coarsen point cloud and distribution using k-means clustering.
    
    Reduces the number of points by clustering and aggregating weights. Empty clusters are removed.
    
    Args:
        X: Input points, shape (N, D)
        a: Distribution weights, shape (N,)
        ratio: Coarsening ratio (fraction of points to keep)
    
    Returns:
        torch.Tensor: Coarsened points, shape (M, D) where M ≈ N * ratio
        torch.Tensor: Coarsened distribution, shape (M,)
    """
    M = int(X.shape[0] * ratio)
    X_coarse, clusters = kmeans(X, M=M)
    a_coarse = torch.scatter_add(torch.zeros(M, dtype=torch.float32), 0, clusters, a)

    fltr = torch.nonzero(a_coarse).squeeze()
    X_coarse, a_coarse = X_coarse[fltr], a_coarse[fltr]
    return X_coarse, a_coarse

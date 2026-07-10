"""
Dimensionality Reduction Utilities

Functions for reducing dimensionality of point clouds and cost matrices using PCA and low-rank approximations.
"""

import numpy as np
import torch

from pykeops.numpy import LazyTensor as NumpyLazyTensor

from scipy.sparse.linalg import aslinearoperator, eigsh


def symmetric_pca(M, dim):
    """
    Compute top eigenvalues and eigenvectors of a symmetric matrix.
  
    Uses sparse eigenvalue solver for efficient computation.
    Compatible with lazy matrices.
    
    Args:
        M: Symmetric matrix (as numpy array or linear operator)
        dim: Number of eigenvalues/eigenvectors to compute
    
    Returns:
        numpy.ndarray: Eigenvalues, shape (dim,)
        numpy.ndarray: Eigenvectors, shape (N, dim)
    """
    M = aslinearoperator(M)

    eigenvalues, eigenvectors = eigsh(M, k=dim, which='LM')
    return eigenvalues, eigenvectors


def pca_factorization(X, cost, dim):
    """
    Factorize cost matrix using PCA for low-rank approximation.
    
    Computes C ≈ U @ V^T where U and V are low-rank factors.
    
    Args:
        X: Input points, numpy array of shape (N, D)
        cost: Cost function to compute pairwise distances
        dim: Rank of approximation
    
    Returns:
        numpy.ndarray: First factor U, shape (N, dim)
        numpy.ndarray: Second factor V, shape (N, dim)
    """
    C = cost(NumpyLazyTensor(X[:, None, :]), NumpyLazyTensor(X[None, :, :]))
    eigenvalues, eigenvectors = symmetric_pca(C, dim)

    U = eigenvectors * np.sqrt(np.abs(eigenvalues))
    V = U * np.sign(eigenvalues)

    return U, V


def low_rank_approximation(X, cost, dim): #TODO supprimer
    """
    Compute low-rank approximation of cost matrix as PyTorch tensors.
    
    Wrapper around pca_factorization that returns PyTorch tensors.
    
    Args:
        X: Input points (torch.Tensor), shape (N, D)
        cost: Cost function
        dim: Rank of approximation
    
    Returns:
        torch.Tensor: First factor U, shape (N, dim)
        torch.Tensor: Second factor V, shape (N, dim)
    """
    Xnp = X.cpu().numpy()
    U, V = pca_factorization(Xnp, cost, dim)

    return torch.tensor(U, dtype=torch.float32).contiguous(), torch.tensor(V, dtype=torch.float32).contiguous()

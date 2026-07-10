"""
Kernel Operations for Gromov-Wasserstein

Functions for converting between cost and kernel representations, and kernel-based dimensionality reduction.
"""

import numpy as np
import torch
from pykeops.numpy import LazyTensor as NumpyLazyTensor

from utils.math.dimension_reduction import symmetric_pca


def center_kernel(K, a, lazy=False):
    """
    Center a kernel matrix with respect to a distribution.
        
    Args:
        K: Kernel matrix, shape (N, N)
        a: Distribution weights, shape (N,)
        lazy: If True, use LazyTensor implementation
    
    Returns:
        Kernel matrix or LazyTensor: Centered kernel
    """

    if lazy:
        K_sum = (K * NumpyLazyTensor(a[:, None, None])).sum(dim=0)
        K_sum2 = (K_sum.squeeze() * a).sum()
        K_centered = K - NumpyLazyTensor(K_sum[:, None]) - NumpyLazyTensor(K_sum[None, :]) + K_sum2
    else:
        K_sum = (K * a[:, None]).sum(dim=0)
        K_sum2 = (a * K_sum.squeeze()).sum()
        K_centered = K - K_sum[:, None] - K_sum[None, :] + K_sum2

    return K_centered


###################################
###   Matrix-based operations   ###
###################################


def kernel_from_costmatrix(C, a=None, center=False):
    """
    Convert cost matrix to kernel matrix.
    
    Uses the formula: K[i,j] = (C[i,0] + C[0,j] - C[i,j]) / 2
    
    Args:
        C: Cost/distance matrix, shape (N, N)
        a: Distribution for centering (optional), shape (N,)
        center: If True, center the kernel using distribution a
    
    Returns:
        torch.Tensor: Kernel matrix, shape (N, N)
    """
    K = (C[:, 0].view(-1, 1) + C[0, :].view(1, -1) - C) / 2
    return center_kernel(K, a) if center else K


def cost_from_kernelmatrix(K):
    """
    Convert kernel matrix to cost matrix.
    
    Uses the formula: C[i,j] = K[i,i] + K[j,j] - 2*K[i,j]
    
    Args:
        K: Kernel matrix, shape (N, N)
    
    Returns:
        torch.Tensor: Cost/distance matrix, shape (N, N)
    """
    Kxx = K.diag()
    return Kxx[:, None] + Kxx[None, :] - 2 * K


#####################################
###   Function-based operations   ###
#####################################


def kernel_pca(X, a, kernel, dim=2, return_eigendecomp=False, tol=None):
    """
    Perform kernel PCA for dimensionality reduction.
    
    Projects data into a lower-dimensional space using kernel methods.
    
    Args:
        X: Input data, shape (N, D)
        a: Distribution weights, shape (N,)
        kernel: Kernel function taking (N, 1, D) and (1, N, D) arrays and outputs an (N,N) array. 
                It should be compatible with Numpy inputs
        dim: Target dimensionality
        return_eigendecomp: If True, also return eigenvalues and eigenvectors
        tol: If not None, truncate eigenvalues smaller than tol * eigenvalues.max(). This prevents numerical issues
             related to approximation errors for eigenvalues smaller than machine precision
    Returns:
        torch.Tensor: Transformed data, shape (N, dim)
        torch.Tensor (optional): Eigenvalues if return_eigenvalues=True
    """
    Xnp = X.cpu().numpy()
    anp = a.cpu().numpy() if a is not None else np.ones(Xnp.shape[0], dtype=np.float32) / Xnp.shape[0]

    K = center_kernel(kernel(NumpyLazyTensor(Xnp[:, None, :]), NumpyLazyTensor(Xnp[None, :, :])), a=anp, lazy=True)

    eigenvalues, eigenvectors = symmetric_pca(K, dim)

    if tol is not None:
        filter = eigenvalues > (eigenvalues.max() * tol)
        eigenvalues, eigenvectors = eigenvalues[filter], eigenvectors[:, filter]

    X_transformed = eigenvectors * np.sqrt(eigenvalues.clip(min=0))
    X_transformed = torch.tensor(X_transformed, dtype=X.dtype, device=X.device).contiguous()

    if return_eigendecomp:
        return X_transformed, torch.tensor(eigenvalues, dtype=X.dtype, device=X.device), torch.tensor(eigenvectors,
                                                                                                      dtype=X.dtype,
                                                                                                      device=X.device).contiguous()
    else:
        return X_transformed


def reduce_kernel(X, a, cost, approx_dim, return_eigendecomp=False, tol=None):
    """
    Reduce dimensionality using kernel method from a cost function.
    
    Creates a kernel from the cost function and applies kernel PCA.
    
    Args:
        X: Input data, shape (N, D)
        a: Distribution weights, shape (N,)
        cost: Cost/distance function
        approx_dim: Target dimensionality
        return_eigendecomp: If True, also return eigenvalues and eigenvectors
        tol: If not None, truncate eigenvalues smaller than tol * eigenvalues.max(). This prevents numerical issues
             related to approximation errors for eigenvalues smaller than machine precision
    
    Returns:
        torch.Tensor: Embedded data, shape (N, approx_dim)
        torch.Tensor (optional): Eigenvalues if return_eigenvalues=True
    """
    X_0 = X[0].cpu().numpy()
    kernel = lambda u, v: (cost(u, X_0) + cost(v, X_0) - cost(u, v)) / 2

    return kernel_pca(X, a, kernel, dim=approx_dim, return_eigendecomp=return_eigendecomp, tol=tol)

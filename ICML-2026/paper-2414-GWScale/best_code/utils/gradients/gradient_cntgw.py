"""
Differentiable Kernel-Reduced Gromov-Wasserstein

Gradient computation through kernel PCA dimensionality reduction combined with quadratic GW. 
"""

import torch
from pykeops.torch import LazyTensor

from utils.gradients.gradient_quadraticgw import diffloss_quadraticgw
from utils.gradients.helpers import get_gradient_from_diffloss
from utils.implementation.kernels import reduce_kernel


def _kernel_projection(X, eigvals, eigvects, a, cost):
    """
    Project points onto kernel PCA subspace differentiably.
    
    Reconstructs the embedding in a way that preserves gradient flow from the embedded coordinates back to the original coordinates X.
    
    Args:
        X: Original points, shape (N, D)
        X_red: Reduced embedding (computed non-differentiably), shape (N, d)
        eigvals: Eigenvalues from kernel PCA, shape (d,)
        a: Distribution weights, shape (N,)
        cost: Cost function
    
    Returns:
        torch.Tensor: Differentiable embedding, shape (N, d)
    """
    if a is None:
        a = torch.ones(X.shape[0], dtype=torch.float32) / X.shape[0]

    X_0 = X[0].detach()
    kernel = lambda u, v: (cost(u, X_0) + cost(v, X_0) - cost(u, v)) / 2

    K = kernel(LazyTensor(X[:, None, :]), LazyTensor(X[None, :, :].detach()))

    K_sum = (K * LazyTensor(a[:, None, None])).sum(dim=0).detach()
    K_sum2 = (K_sum.squeeze() * a).sum()
    K_centered = K - LazyTensor(K_sum[:, None]) - LazyTensor(K_sum[None, :]) + K_sum2  # (N, N)
    return (K_centered * LazyTensor(eigvects[None, :, :])).sum(dim=1) / eigvals[None, :].sqrt()  # (N, D)


def _differentiate_kernelpca(X, a, cost, approx_dim, tol=1e-6):
    """
    Create differentiable kernel PCA embedding.
    
    First computes embedding without gradients, then reconstructs it through differentiable operations to enable gradient propagation.
    
    Args:
        X: Input points, shape (N, D)
        a: Distribution weights, shape (N,)
        cost: Cost function
        approx_dim: Target dimension for embedding
        tol: If not None, truncate eigenvalues smaller than tol * eigenvalues.max() in the kernel PCA. This prevents
             numerical issues related to approximation errors for eigenvalues smaller than machine precision
    Returns:
        torch.Tensor: Differentiable embedding, shape (N, approx_dim)
    """
    with torch.no_grad():
        X_emb, eigvalsX, eigvectsX = reduce_kernel(X, a, cost, approx_dim, return_eigendecomp=True, tol=tol)
    return _kernel_projection(X, eigvalsX, eigvectsX, a, cost)


def diffloss_cntgw(X, Y, costs, a=None, b=None, approx_dims=10, f_init=None, g_init=None, match_init=None,
                   solver_kwargs=None, return_solution=False):
    """
    Compute differentiable GW loss in kernel-reduced space.
    
    Embeds both point clouds into lower-dimensional spaces using kernel PCA, then solves quadratic GW on the embeddings. The entire pipeline is differentiable,
    allowing gradients to flow back to original coordinates.
    
    This approach is implemented using lazy tensors, allowing memory-efficient computations on large inputs.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        costs: Cost function(s), single function or tuple (cost_x, cost_y)
        a: Source distribution
        b: Target distribution
        approx_dims: Embedding dimension(s), int for same dim, tuple for different
        f_init: Initial source dual
        g_init: Initial target dual
        match_init: Initial matching for potential initialization, torch.tensor of indices of size (N,) with values in 0,...,M-1
        solver_kwargs: QuadraticGW solver parameters
        return_solution: If True, return solution components
    
    Returns:
        torch.Tensor: Differentiable loss
        tuple (optional): (f, g, Z, matching) if return_solution=True
    """
    cost_x, cost_y = costs if isinstance(costs, tuple) else costs, costs
    approx_dim_x, approx_dim_y = approx_dims if isinstance(approx_dims, tuple) else approx_dims, approx_dims

    X_emb = _differentiate_kernelpca(X, a, cost=cost_x, approx_dim=approx_dim_x)
    Y_emb = _differentiate_kernelpca(Y, b, cost=cost_y, approx_dim=approx_dim_y)

    Z_init = None
    if match_init is not None:
        a_init = torch.ones((X.shape[0],), dtype=torch.float32) / X.shape[0] if a is None else a
        Z_init = ((X_emb[:, :, None] * Y_emb[match_init, None, :]) * a_init[:, None, None]).sum(0)

    return diffloss_quadraticgw(X_emb, Y_emb, a, b, f_init, g_init, Z_init, solver_kwargs, return_solution)


def gradient_cntgw(X, Y, costs, which='both', a=None, b=None, approx_dims=10, f_init=None, g_init=None,
                   match_init=None, solver_kwargs=None, return_solution=False):
    """
    Compute gradients of kernel GW distance with respect to input point clouds, using kernel PCA for dimension reduction.
    
    Efficient gradient computation for high-dimensional point clouds by working in reduced kernel PCA space. 
    Gradients flow back to original coordinates through the differentiable embedding.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        costs: Cost function(s), single function or tuple (cost_x, cost_y)
        which: Controls which gradients to compute ('x', 'y', or 'both')
        a: Source distribution
        b: Target distribution
        approx_dims: Embedding dimension(s), int for same dim, tuple for different
        f_init: Initial source dual for warm start
        g_init: Initial target dual for warm start
        match_init: Initial matching for potential initialization, torch.tensor of indices of size (N,) with values in 0,...,M-1
        solver_kwargs: Additional keyword arguments for QuadraticGW solver
        return_solution: If True, return solution components in addition to gradients
    
    Returns:
        torch.Tensor or tuple: Gradients in original coordinate space
        tuple (optional): (gradients, (f, g, Z, matching)) if return_solution=True
    """
    if which in ['x', 'both']:
        X = X.clone().requires_grad_(True)
    if which in ['y', 'both']:
        Y = Y.clone().requires_grad_(True)

    res = diffloss_cntgw(X, Y, costs, a, b, approx_dims, f_init, g_init, match_init, solver_kwargs, return_solution)
    loss = res[0] if return_solution else res

    grads = get_gradient_from_diffloss(X, Y, loss, which=which)

    return (grads, res[1:]) if return_solution else grads

"""
Entropic Optimal Transport Utilities

Utility functions for working with transport plans.
"""

from pykeops.torch import LazyTensor

def barycentric_projection(source, transport, source_dim=0, lazy=False):
    """
    Compute barycentric projection of source points through a transport plan.
    
    Projects source points to target space as weighted averages according to the
    transport plan: Y_proj[j] = sum_i P[i,j] * X[i] / sum_i P[i,j]
    
    Args:
        source: Source data, shape (N, D) if source_dim=0, (M, D) otherwise
        transport: Transport plan, shape (N, M)
        source_dim: Dimension to aggregate (0 for row-wise, 1 for column-wise)
        lazy: If True, use LazyTensor implementation
    
    Returns:
        torch.Tensor: Projected points, shape (M, D) if source_dim=0
    """
    if lazy:
        source_lazy = LazyTensor(source[:, None, :]) if source_dim == 0 else LazyTensor(source[None, :, :])
        proj = (transport * source_lazy).sum(source_dim) / transport.sum(source_dim)
    else:
        source_ext = source[:, None, :]  if source_dim == 0 else source[None, :, :]
        proj = (transport[:,:,None] * source_ext).sum(source_dim) / transport.sum(source_dim)[:,None]

    return proj

def matching(transport, target_dim=1):
    """
    Extract hard matching from soft transport plan via argmax.
    
    Converts probabilistic transport to discrete matching by taking the maximum
    correspondence for each point.
    
    Args:
        transport: Transport plan, shape (N, M)
        target_dim: Dimension to take argmax over
    
    Returns:
        torch.Tensor: Indices of matched points, shape (N,) if target_dim=1, (M,) otherwise
    """
    return transport.argmax(target_dim).squeeze()

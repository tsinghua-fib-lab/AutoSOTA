"""
Initialization Functions for Gromov-Wasserstein Solvers

Provides initialization strategies for potential matrices in various GW formulations. By default (initialization_mode=None),
solvers are initialized at the trivial transport plan P = a @ b.T
"""

import torch
from pykeops.torch import LazyTensor

from utils.math.functions import center, squared_norm


def initialize_entropicgw(Cx, Cy, a, b, initialization_mode=None):
    """
    Initialize potential for EntropicGW from cost matrices.
    
    Args:
        Cx: Source cost matrix, shape (N, N)
        Cy: Target cost matrix, shape (M, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        initialization_mode: 'random' for random initialization, else initialization on trivial plan P = a @ b.T
    
    Returns:
        torch.Tensor: Initial cost matrix Z, shape (N, M)
    """
    if initialization_mode == 'random':
        return - torch.rand((Cx.shape[0], Cy.shape[0]), dtype=Cx.dtype, device=Cx.device)
    else:
        CikAiAk = torch.einsum("ik,i->i", Cx, a)
        CjlBjBl = torch.einsum("jl,j->j", Cy, b)

        return - 4 * CikAiAk[:, None] * CjlBjBl[None, :]


def initialize_kernelgw(Kx, Ky, a, b, initialization_mode=None):
    """
    Initialize potential for KernelGW.
    
    Args:
        Kx: Source kernel matrix, shape (N, N)
        Ky: Target kernel matrix, shape (M, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        initialization_mode: 'random' for random initialization, else initialization on trivial plan P = a @ b.T
    
    Returns:
        torch.Tensor: Initial kernel-based potential Z, shape (N, M)
    """
    if initialization_mode == 'random':
        return - torch.rand((Kx.shape[0], Ky.shape[0]), dtype=Kx.dtype, device=Kx.device)
    else:
        KikAiAk = torch.einsum("ik,i->i", Kx, a)
        KjlBjBl = torch.einsum("jl,j->j", Ky, b)

        K = - 16 * KikAiAk[:, None] * KjlBjBl[None, :]
        K -= 4 * Kx.diag()[:, None] * Ky.diag()[None, :]

        return K


def initialize_quadraticgw(X, Y, a, b, initialization_mode=None):
    """
    Initialize potential for QuadraticGW from point clouds.
    
    Args:
        X: Source point cloud, shape (N, D_x)
        Y: Target point cloud, shape (M, D_y)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        initialization_mode: 'random'  for random initialization, 'identity' for initialization on P = diag(a), else initialization on trivial plan P = a @ b.T
    
    Returns:
        torch.Tensor: Initial potential matrix Z, shape (D_x, D_y)
    """
    X_centered = center(X, a)
    Y_centered = center(Y, b)

    X_sqnorm = squared_norm(X_centered, keepdim=True)  # (N,1)
    Y_sqnorm = squared_norm(Y_centered, keepdim=True)  # (M,1)
    if initialization_mode == 'random':
        scale = ((a[:, None] * X_sqnorm).sum() * (b[:, None] * Y_sqnorm).sum()).sqrt()
        return scale * (torch.rand((X.shape[1], Y.shape[1]), dtype=torch.float32) - 0.5)
    elif initialization_mode == 'identity':
        return torch.einsum("iu,iv->uv", X_centered, Y_centered) / X.shape[0]
    else:
        return torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float32)


def initialize_lowrankgw(Cx2, Cy1, a, b, initialization_mode=None):
    """
    Initialize potential for LowRankGW using cost matrix factors.
    
    Args:
        Cx2: Second factor of source cost, shape (N, D)
        Cy1: First factor of target cost, shape (M, D')
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        initialization_mode: 'random'  for random initialization, 'identity' for initialization on P = diag(a), else initialization on trivial plan P = a @ b.T
    
    Returns:
        torch.Tensor: Initial potential matrix Z, shape (D, D')
    """
    if initialization_mode == 'random':
        return torch.rand((Cx2.shape[1], Cy1.shape[1]), dtype=torch.float32)
    elif initialization_mode == 'identity':  # DEBUG
        return 4 * torch.einsum("iu,iv->uv", Cx2, Cy1) / Cx2.shape[0]
    elif initialization_mode == 'zero':
        return torch.zeros((Cx2.shape[1], Cy1.shape[1]), dtype=torch.float32)
    else:
        P_ij = LazyTensor(a[:, None, None]) * LazyTensor(b[None, :, None])
        Dx2_i = LazyTensor(Cx2[:, None, :])  # (N, 1, D)
        DxP = (P_ij * Dx2_i).sum(dim=0)  # (M, D)

        return 4 * torch.einsum("ju,jv->uv", DxP, Cy1)  # (D, D')

def initialize_sampledgw(X, Y, a, b, cost_x, cost_y, indices_k, indices_l):
    """
    Initialize sampled cost vectors for SampledGW.
    
    Computes cost vectors only for sampled indices rather than full matrices.
    
    Args:
        X: Source point cloud, shape (N, D_x)
        Y: Target point cloud, shape (M, D_y)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        cost_x: Cost function for source space
        cost_y: Cost function for target space
        indices_k: Sampled source indices, shape (S,)
        indices_l: Sampled target indices, shape (S,)
    
    Returns:
        tuple: (Dx_is, Dy_js) where Dx_is has shape (N, S) and Dy_js has shape (M, S)
    """
    Dx_is = cost_x(X[:, None, :], X[None, indices_k, :])  # (N, S)
    Dy_js = cost_y(Y[:, None, :], Y[None, indices_l, :])  # (M, S)

    return Dx_is, Dy_js

"""
Gromov-Wasserstein Loss Functions

Loss computation functions for various Gromov-Wasserstein formulations.
"""

import torch
from pykeops.torch import LazyTensor

from utils.math.functions import kl_divergence, center, squared_norm
from utils.implementation.kernels import cost_from_kernelmatrix


def gw_loss(Cx, Cy, P, a, b, eps=None, include_divergence=True):
    """
    Compute Gromov-Wasserstein loss from cost matrices.
    
    Computes GW(Cx, Cy, P) = sum_{i,j,k,l} (Cx[i,k] - Cy[j,l])^2 * P[i,j] * P[k,l] by decomposing it into three terms:
    	GW(Cx, Cy, P) = Cxx + Cyy - 2 * Cxy,
    with:
    	Cxx = sum_{i,k} Cx[i,k]^2 * a[i] * a[k]
    	Cyy = sum_{j,l} Cy[j,l]^2 * b[j] * b[l]
    	Cxy = sum_{k,l} (sum_{i,j} Cx[i,k] * Cy[j,l] * P[i,j])) * P[k,l]
   
    Args:
        Cx: Source cost matrix, shape (N, N)
        Cy: Target cost matrix, shape (M, M)
        P: Transport plan, shape (N, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        eps: Entropic regularization parameter
        include_divergence: If True, add KL divergence term
    
    Returns:
        torch.Tensor: Scalar GW loss value
    """
   
    Cxx = ((Cx ** 2) * a[:, None] * a[None, :]).sum()
    Cyy = ((Cy ** 2) * b[:, None] * b[None, :]).sum()

    PCx = torch.einsum('ij,ik->jk', P, Cx)
    CyPCx = torch.einsum('jl,jk->kl', Cy, PCx)
    PCx = None
    Cxy = (CyPCx * P).sum()
    CyPCx = None

    energy = Cxx + Cyy - 2 * Cxy
    if include_divergence: energy += eps * kl_divergence(a[:, None, None], b[None, :, None], P)

    return energy


def gw_loss_from_points(X, Y, cost_x, cost_y, P, a, b, eps=None, include_divergence=True):
    """
    Compute GW loss from point clouds by computing explicit cost matricess.
    
    Args:
        X: Source point cloud, shape (N, D_x)
        Y: Target point cloud, shape (M, D_y)
        cost_x: Cost function for source space
        cost_y: Cost function for target space
        P: Transport plan, shape (N, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        eps: Entropic regularization parameter
        include_divergence: If True, add KL divergence term
    
    Returns:
        torch.Tensor: Scalar GW loss value
    """
    #TODO changer cuda
    Dx = cost_x(X[:, None, :], X[None, :, :])
    Dy = cost_y(Y[:, None, :], Y[None, :, :])

    return gw_loss(Dx.to('cuda'), Dy.to('cuda'), P.to('cuda'), a.to('cuda'), b.to('cuda'), eps, include_divergence)


def gw_loss_from_lowrank(Cx1, Cx2, Cy1, Cy2, P, a, b, eps=None, include_divergence=True):
    """
    Compute GW loss using low-rank factorization of cost matrices.
    
    Uses factorizations Cx ≈ Cx1 @ Cx2.T and Cy ≈ Cy1 @ Cy2.T for efficient
    computation with large cost matrice, using the same decomposition as gw_loss.
    
    Args:
        Cx1: First factor of source cost, shape (N, D)
        Cx2: Second factor of source cost, shape (N, D)
        Cy1: First factor of target cost, shape (M, D')
        Cy2: Second factor of target cost, shape (M, D')
        P: Transport plan, shape (N, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        eps: Entropic regularization parameter
        include_divergence: If True, add KL divergence term
    
    Returns:
        torch.Tensor: Scalar approximate GW loss value
    """
    a_i, a_j = LazyTensor(a[:, None, None]), LazyTensor(a[None, :, None])
    b_i, b_j = LazyTensor(b[:, None, None]), LazyTensor(b[None, :, None])

    Cx1_i, Cx2_j =  LazyTensor(Cx1[:, None, :]), LazyTensor(Cx2[None, :, :])
    Cxx = (a_i * a_j * (Cx1_i * Cx2_j).sum(-1) ** 2).sum(dim=1).sum()

    Cy1_i, Cy2_j =  LazyTensor(Cy1[:, None, :]) , LazyTensor(Cy2[None, :, :])
    Cyy = (b_i * b_j * (Cy1_i * Cy2_j).sum(-1) ** 2).sum(dim=1).sum()

    Cx2_i = LazyTensor(Cx2[:, None, :])  # (N, 1, D)
    Cx2P = (P * Cx2_i).sum(dim=0)  # (M, D)
    G1 = torch.einsum("ju,jv->uv", Cx2P, Cy1)  # (D, D')

    Cx1_i = LazyTensor(Cx1[:, None, :])  # (N, 1, D)
    Cx1P = (P * Cx1_i).sum(dim=0)  # (M, D)
    G2 = torch.einsum("ju,jv->uv", Cx1P, Cy2)  # (D, D')

    Cxy = (G1 * G2).sum()

    cost = Cxx + Cyy - 2 * Cxy
    if include_divergence: cost += eps * kl_divergence(a_i, b_j, P)

    return cost

def gw_loss_from_kernel(Kx, Ky, P, a, b, eps=None, include_divergence=True):
    # TODO supprimer
    """
    Compute GW loss using kernel matrices instead of cost matrices.
    
    Args:
        Kx: Source kernel matrix, shape (N, N)
        Ky: Target kernel matrix, shape (M, M)
        P: Transport plan, shape (N, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        eps: Entropic regularization parameter
        include_divergence: If True, add KL divergence term
    
    Returns:
        torch.Tensor: Scalar GW loss value in kernel space
    """
    cste_x = ((cost_from_kernelmatrix(Kx) ** 2) * a[:, None] * a[None, :]).sum()
    cste_y = ((cost_from_kernelmatrix(Ky) ** 2) * b[:, None] * b[None, :]).sum()
    cste_xy = (Kx.diag() * a).sum() * (Ky.diag() * b).sum()

    KyP = torch.einsum("jl,kl->jk", Ky, P)
    C = - 8 * torch.einsum("ik,jk->ij", Kx, KyP)
    KyP = None
    C -= 4 * Kx.diag()[:, None] * Ky.diag()[None, :]

    energy = cste_x + cste_y - 4 * cste_xy + (P * C).sum()

    if include_divergence: energy += eps * kl_divergence(a[:, None, None], b[None, :, None], P)

    return energy


def gw_loss_euclidean(X, Y, P, a, b, eps=None, include_divergence=True):
    """
    Compute GW loss for squared Euclidean norms clouds using quadratic structure.
    
    Decomposes the squared norm GW cost as:
    	GW(Cx, Cy, P) = Cxx + Cyy - 4 * Cxy - 4 * Cq - 8 * Cr,
    with:
       	Cxx = sum_{i,k} (X[i] - X[k])^2 * a[i] * a[k]
    	Cyy = sum_{j,l} (Y[j] - Y[l])^2 * b[j] * b[l]
    	Cxy = sum_{i} ||X[i]||^2 * a[i] * sum_{j} ||Y[j]||^2 * b[j]
    	
    	Cq = sum_{ij} ||X[i]||^2 * ||Y[j]||^2 * P[i,j]    	
    	Cr = ||Z||^2 where Z is the cross-correlation matrix of P: Z = sum_{i,j} (X[i] @ Y[j].T) * P[i,j]
   
    Args:
        X: Source point cloud, shape (N, D_x)
        Y: Target point cloud, shape (M, D_y)
        P: Transport plan, shape (N, M)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        eps: Entropic regularization parameter
        include_divergence: If True, add KL divergence term
    
    Returns:
        torch.Tensor: Scalar GW loss for Euclidean quadratic cost
    """
    X = center(X, a)
    Y = center(Y, b)

    X_i, X_j = LazyTensor(X[:, None, :]), LazyTensor(X[None, :, :])
    Y_i, Y_j = LazyTensor(Y[:, None, :]), LazyTensor(Y[None, :, :])

    a_i, a_j = LazyTensor(a[:, None, None]), LazyTensor(a[None, :, None])
    b_i, b_j = LazyTensor(b[:, None, None]), LazyTensor(b[None, :, None])

    X_sqnorm = squared_norm(X)
    Y_sqnorm = squared_norm(Y)

    Cxx = (a_i * a_j * (((X_i - X_j) ** 2).sum(dim=-1) ** 2)).sum(dim=1).sum()
    Cyy = (b_i * b_j * (((Y_i - Y_j) ** 2).sum(dim=-1) ** 2)).sum(dim=1).sum()
    Cxy = (a[:, None] * X_sqnorm).sum() * (b[:, None] * Y_sqnorm).sum()

    X_sqnorm_i = LazyTensor(X_sqnorm[:, None, :])  # (N, 1, 1)
    Y_sqnorm_j = LazyTensor(Y_sqnorm[None, :, :])  # (1, M, 1)

    Cq = (X_sqnorm_i * Y_sqnorm_j * P).sum(dim=1).sum()

    XP = (P * X_i).sum(dim=0)  # (M, D)
    Z = torch.einsum("ju,jv->uv", XP, Y)
    Cr = (Z ** 2).sum()

    cost = Cxx + Cyy - 4 * Cxy - 4 * Cq - 8 * Cr
    if include_divergence: cost += eps * kl_divergence(a_i, b_j, P)

    return cost

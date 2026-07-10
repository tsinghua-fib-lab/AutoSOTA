"""
Mathematical Utility Functions

"""
import numpy as np
import torch
from pykeops.torch import LazyTensor


######################################
###   Statistics-based functions   ###
######################################


def mean(X, a=None):
    """
    Compute weighted or unweighted mean of points.
    
    Args:
        X: Points, shape (N, D)
        a: Optional weights, shape (N,). If None, uniform weights used
    
    Returns:
        torch.Tensor: Mean point, shape (D,)
    """
    return (X * a[:, None]).sum(dim=0) if a is not None else torch.mean(X, dim=0)


def center(X, a=None):
    """
    Center points by subtracting weighted mean.
    
    Args:
        X: Points, shape (N, D)
        a: Optional weights, shape (N,)
    
    Returns:
        torch.Tensor: Centered points, shape (N, D)
    """
    return X - mean(X, a=a)


def variance(X, a=None):
    """
    Compute weighted variance of point cloud.
    
    Args:
        X: Points, shape (N, D)
        a: Optional weights, shape (N,)
    
    Returns:
        torch.Tensor: Scalar variance value
    """
    X = center(X, a)
    return ((X ** 2).sum(-1) * a).sum() if a is not None else torch.mean((X ** 2).sum(-1))


def squared_norm(X, keepdim=True):
    """
    Compute squared Euclidean norm of vectors.
    
    Args:
        X: Vectors, shape (N, D)
        keepdim: If True, keep dimension as (N, 1). Otherwise (N,)
    
    Returns:
        torch.Tensor: Squared norms
    """
    return (X ** 2).sum(dim=-1, keepdim=keepdim)


def covariance(X, a=None):
    """
    Compute weighted covariance matrix of point cloud.
    
    Args:
        X: Points, shape (N, D)
        a: Optional weights, shape (N,)
    
    Returns:
        torch.Tensor: Covariance matrix, shape (D, D)
    """
    X = center(X, a)
    if a is None:
        a = torch.ones(X.shape[0]) / X.shape[0]
    return ((X[:, :, None] * X[:, None, :]) * a[:, None, None]).sum(0)


def coupling_covariance(X, Y, P, lazy=True):
    """
    Compute the covariance matrix X^T P Y of a coupling (transport plan) P.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        P: Transport plan, shape (N, M)
        lazy: If True, use LazyTensor implementation
    
    Returns:
        torch.Tensor: Coupling covariance, shape (D_x, D_y)
    """
    X_i = LazyTensor(X[:, None, :]) if lazy else X[:, None, :]  # (N, 1, D)

    XP = (P * X_i).sum(dim=0)  # (M, D)
    return torch.einsum("ju,jv->uv", XP, Y)


def moment(X, a=None, p=2):
    """
    Compute p-th moment of point cloud Mp = (||X - X.mean()||**p).mean()
    
    Args:
        X: Points, shape (N, D)
        a: Optional weights, shape (N,)
        p: Moment order
    
    Returns:
        torch.Tensor: Scalar moment value
    """
    X_centered = center(X, a=a)
    X_sqnorm = squared_norm(X_centered)

    return mean(X_sqnorm ** (p / 2), a=a)


###############################
###   Norms and divergences ###
###############################


def froebenius(A):
    """
    Compute Frobenius norm of a matrix.
    
    Args:
        A: Input matrix
    
    Returns:
        torch.Tensor: Frobenius norm ||A||_F = sqrt(sum(A_ij^2))
    """
    return torch.sqrt((A ** 2).sum(-1).sum(-1))


def kl_divergence(a, b, P, tol=1e-30):
    """
    Compute KL divergence between transport plan and trivial plan a @ b.T.
    
    Computes KL(P || a @ b.T) = sum P_ij log(P_ij / (a_i * b_j))
    
    Args:
        a: Source distribution, shape (N, 1, 1) or broadcastable
        b: Target distribution, shape (1, M, 1) or broadcastable
        P: Transport plan, shape (N, M) or (N, M, 1)
        tol: Small constant to avoid log(0)
    
    Returns:
        torch.Tensor: Scalar KL divergence value
    """
    return (P * (tol + P / (a * b).sum(dim=-1)).log()).sum(dim=1).sum()


################################
###   Geometric transforms   ###
################################


def linear(points, A, origin=None):
    """
    Apply linear transformation to points.

    Computes (points - origin) @ A + origin

    Args:
        points: Input points, shape (N, D)
        A: Transformation matrix, shape (D, D')
        origin: Origin of the linear plane. If None, use 0 as origin

    Returns:
        torch.Tensor: Transformed points, shape (N, D')
    """
    if origin is None:
        transformed_points = torch.einsum('iu,uv->iv', points, A)
    else:
        transformed_points = torch.einsum('iu,uv->iv', points - origin, A) + origin

    return transformed_points


def rotation(points, theta=0, origin=None):
    """
    Rotate 2D points by angle theta around origin.

    Args:
        points: Input points, shape (N, 2)
        theta: Rotation angle in radians
        origin: Center of rotation. If None, rotates around (0,0)

    Returns:
        torch.Tensor: Rotated points, shape (N, 2)
    """
    R = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    return linear(points, R, origin)


def noise(points, std=0.1):
    """
    Add Gaussian noise to points.

    Args:
        points: Input points, shape (N, D)
        std: Standard deviation of Gaussian noise

    Returns:
        torch.Tensor: Noisy points, shape (N, D)
    """
    return points + torch.normal(mean=torch.zeros(points.shape), std=std)


def normalize(points):
    """
    Normalize point cloud to unit scale centered at origin.

    Centers points and scales so maximum distance from origin is 1.

    Args:
        points: Input points, shape (N, D)

    Returns:
        torch.Tensor: Normalized points, shape (N, D)
    """
    points = points - points.mean(dim=0, keepdims=True)
    return points / torch.max(torch.linalg.norm(points, axis=-1))


def subsample(points, N):
    """
    Randomly subsample points without replacement.

    Args:
        points: Input points, shape (M, D)
        N: Number of points to sample

    Returns:
        torch.Tensor: Subsampled points, shape (N, D) if N <= M, else original
    """
    M = points.shape[0]
    if N > M:
        return points
    else:
        idx = torch.tensor(np.random.choice(M, N, replace=False))
        return points[idx]

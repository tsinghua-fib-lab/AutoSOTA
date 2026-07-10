"""
Quadratic Gromov-Wasserstein Solver

This module implements the quadratic Gromov-Wasserstein distance for point clouds with squared Euclidean cost,
exploiting the quadratic structure for efficient computation.
"""

import torch
from pykeops.torch import LazyTensor

from solvers.gromov_wasserstein.generic.embedding_based import EmbeddingBasedGW
from utils.implementation.gw_losses import gw_loss_euclidean
from utils.implementation.initializations import initialize_quadraticgw
from utils.math.costs import euclidean
from utils.math.functions import center, squared_norm, coupling_covariance


class QuadraticGW(EmbeddingBasedGW):
    """
    Gromov-Wasserstein solver for quadratic (squared Euclidean) costs.

    This solver is specifically designed for the case where the cost function is the squared Euclidean distance.
    It exploits the quadratic structure to represent the problem more efficiently using centered coordinates and
    covariance-like potential matrices.

    Attributes:
        X (torch.Tensor): Source point cloud, shape (N, D_x)
        Y (torch.Tensor): Target point cloud, shape (M, D_y)
        X_centered (torch.Tensor): Mean-centered source points, shape (N, D_x)
        Y_centered (torch.Tensor): Mean-centered target points, shape (M, D_y)
        X_sqnorm (torch.Tensor): Squared norms of centered source points, shape (N, 1)
        Y_sqnorm (torch.Tensor): Squared norms of centered target points, shape (M, 1)
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
    """

    def __init__(self, X, Y, a=None, b=None, **kwargs):
        """
        Initialize the quadratic Gromov-Wasserstein solver.

        Args:
            X: Source point cloud, shape (N, D_x)
            Y: Target point cloud, shape (M, D_y)
            a: Source distribution. If None, uses uniform distribution
            b: Target distribution. If None, uses uniform distribution
            **kwargs: Additional arguments passed to EmbeddingBasedGW
        """
        a = torch.ones((X.shape[0],), dtype=torch.float32) / X.shape[0] if a is None else a
        b = torch.ones((Y.shape[0],), dtype=torch.float32) / Y.shape[0] if b is None else b

        # Center the point clouds according to the distributions
        self.X_centered = center(X, a)
        self.Y_centered = center(Y, b)

        # Precompute squared norms for efficient distance computation
        self.X_sqnorm = squared_norm(self.X_centered, keepdim=True)  # (N,1)
        self.Y_sqnorm = squared_norm(self.Y_centered, keepdim=True)  # (M,1)

        super().__init__(X=X, Y=Y, costs=lambda u, v: euclidean(u, v, p=2), a=a, b=b, lazy=True, **kwargs)

    def initialize_potential(self):
        """
        Initialize the potential matrix.

        For quadratic GW, the potential is a (D_x, D_y) matrix representing a linear covariance matrix between the
        two spaces.

        Returns:
            torch.Tensor: Initial potential matrix, shape (D_x, D_y)
        """
        return initialize_quadraticgw(self.X, self.Y, self.a, self.b, self.initialization_mode)

    def update_potential(self):
        """
        Update the potential matrix based on current transport plan.

        Computes the coupling covariance matrix:
        Z = X_centered^T P Y_centered

        Returns:
            torch.Tensor: Updated potential matrix, shape (D_x, D_y)
        """
        return coupling_covariance(X=self.X_centered, Y=self.Y_centered, P=self.transport_plan(lazy=True))

    def cost_matrix(self, lazy=True):
        """
        Compute the cost matrix for Sinkhorn using the quadratic structure.

        The cost is computed as:
        C_ij = -16 * <X_i, Z Y_j> - 4 * ||X_i||^2 * ||Y_j||^2

        This exploits the quadratic form to avoid computing full distance matrices.

        Args:
            lazy: If True, return C_ij as a LazyTensor

        Returns:
            torch.Tensor or LazyTensor: Cost matrix, shape (N, M)
        """
        XZ = torch.einsum("ik,kl->il", self.X_centered, self.Z)  # (N, D')

        XZ_i = LazyTensor(XZ[:, None, :]) if lazy else XZ[:, None, :]  # (N, 1, D')
        Y_j = LazyTensor(self.Y_centered[None, :, :]) if lazy else self.Y_centered[None, :, :]  # (1, M, D')

        X_sqnorm_i = LazyTensor(self.X_sqnorm[:, None, :]) if lazy else self.X_sqnorm[:, None, :]  # (N, 1, 1)
        Y_sqnorm_j = LazyTensor(self.Y_sqnorm[None, :, :]) if lazy else self.Y_sqnorm[None, :, :]  # (1, M, 1)

        C_ij = - 16 * (XZ_i * Y_j).sum(dim=2) - 4 * (X_sqnorm_i * Y_sqnorm_j).sum(dim=2)  # (N, M)

        return C_ij

    def loss(self, include_divergence=True, **kwargs):
        """
        Compute the quadratic Gromov-Wasserstein loss.

        Args:
            include_divergence: If True, include entropic regularization term
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar loss value
        """
        P = self.transport_plan(lazy=True)
        return gw_loss_euclidean(self.X, self.Y, P, self.a, self.b, self.eps, include_divergence)

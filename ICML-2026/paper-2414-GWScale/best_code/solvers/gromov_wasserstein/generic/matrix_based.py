"""
Matrix-Based Gromov-Wasserstein Solver

This module implements Gromov-Wasserstein distance computation from precomputed base cost matrices.
"""

import torch

from solvers.gromov_wasserstein.generic.generic_sinkhorngw import SinkhornBasedGW
from utils.implementation.gw_losses import gw_loss


class MatrixBasedGW(SinkhornBasedGW):
    """
    Gromov-Wasserstein solver that works with precomputed base cost matrices.

    This solver takes precomputed pairwise cost matrices for both spaces and computes the Gromov-Wasserstein distance
    between them. This is efficient when distance matrices are already available or when working with structured/graph
    data, but is memory-intensive.

    Attributes:
        Cx (torch.Tensor): Cost matrix for source space, shape (N, N)
        Cy (torch.Tensor): Cost matrix for target space, shape (M, M)
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
    """

    def __init__(self, Cx, Cy, a=None, b=None, **kwargs):
        """
        Initialize the matrix-based Gromov-Wasserstein solver.

        Args:
            Cx: Cost/distance matrix for source space, shape (N, N)
            Cy: Cost/distance matrix for target space, shape (M, M)
            a: Source distribution. If None, uses uniform distribution
            b: Target distribution. If None, uses uniform distribution
            **kwargs: Additional arguments passed to SinkhornBasedGW
        """
        self.Cx = Cx
        self.Cy = Cy

        a = torch.ones((Cx.shape[0],), dtype=Cx.dtype, device=Cx.device) / Cx.shape[0] if a is None else a
        b = torch.ones((Cy.shape[0],), dtype=Cy.dtype, device=Cy.device) / Cy.shape[0] if b is None else b

        super().__init__(a=a, b=b, **kwargs)

    def to(self, device):
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
        """
        super().to(device)
        self.Cx = self.Cx.to(device)
        self.Cy = self.Cy.to(device)

    def cost_matrix(self, lazy=None):
        """
        Return the current potential matrix as the cost matrix.

        For matrix-based GW, the potential Z is directly used as the cost matrix for the Sinkhorn problem.

        Args:
            lazy: Unused (kept for interface compatibility)

        Returns:
            torch.Tensor: Cost matrix Z, shape (N, M)
        """
        return self.Z

    def loss(self, include_divergence=True, **kwargs):
        """
        Compute the Gromov-Wasserstein loss from cost matrices.

        Args:
            include_divergence: If True, include entropic regularization term
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar Gromov-Wasserstein loss value
        """
        P = self.sinkhorn_solver.transport_plan()

        return gw_loss(self.Cx, self.Cy, P, self.a, self.b, self.sinkhorn_solver.eps, include_divergence)

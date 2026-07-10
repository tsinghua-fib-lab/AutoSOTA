"""
Embedding-Based Gromov-Wasserstein Solver

This module implements Gromov-Wasserstein cost computation directly from point cloud embeddings (coordinates) rather
than precomputed cost matrices.
"""

import torch

from solvers.gromov_wasserstein.generic.generic_sinkhorngw import SinkhornBasedGW
from utils.implementation.gw_losses import gw_loss_from_points


class EmbeddingBasedGW(SinkhornBasedGW):
    """
    Gromov-Wasserstein solver that works directly with point embeddings.

    This solver computes the Gromov-Wasserstein distance between two point clouds (X and Y) by computing distances
    on-the-fly using specified cost functions, rather than requiring precomputed matrices. This is memory-efficient
    for large point clouds.

    Cost functions should broadcast correctly with embedding tensors, e.g. if X is a tensor/LazyTensor of shape
    (N,1,D_x) and X' a tensor/LazyTensor of shape  (1,P,D_x), then cost_x(X, X') should be a tensor/LazyTensor of shape
    (N, P).

    Attributes:
        X (torch.Tensor): Source point cloud, shape (N, D_x)
        Y (torch.Tensor): Target point cloud, shape (M, D_y)
        cost_x (callable): Cost function for computing base costs in X space
        cost_y (callable): Cost function for computing base costs in Y space
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
    """

    def __init__(self, X, Y, costs, a=None, b=None, **kwargs):
        """
        Initialize the embedding-based Gromov-Wasserstein solver.

        Args:
            X: Source point cloud, shape (N, D_x)
            Y: Target point cloud, shape (M, D_y)
            costs: Cost function(s) for computing distances. Can be:
                   - Single function: used for both X and Y
                   - Tuple (cost_x, cost_y): separate functions for each space
            a: Source distribution. If None, uses uniform distribution
            b: Target distribution. If None, uses uniform distribution
            **kwargs: Additional arguments passed to SinkhornBasedGW
        """
        self.X, self.Y = X, Y

        a = torch.ones((X.shape[0],), dtype=torch.float32) / X.shape[0] if a is None else a
        b = torch.ones((Y.shape[0],), dtype=torch.float32) / Y.shape[0] if b is None else b

        self.cost_x, self.cost_y = costs if isinstance(costs, tuple) else costs, costs

        super().__init__(a=a, b=b, **kwargs)

    def loss(self, include_divergence=True, **kwargs):
        """
        Compute the Gromov-Wasserstein loss.

        Args:
            include_divergence: If True, include entropic regularization term
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar Gromov-Wasserstein loss value
        """
        return gw_loss_from_points(X=self.X, Y=self.Y, cost_x=self.cost_x, cost_y=self.cost_y,
                                   P=self.transport_plan(lazy=False, device='cuda'), a=self.a, b=self.b, eps=self.eps,
                                   include_divergence=include_divergence)

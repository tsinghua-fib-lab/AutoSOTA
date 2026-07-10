"""
Low-Rank Gromov-Wasserstein Solver

Implements a memory-efficient Gromov-Wasserstein solver using low-rank approximations of the cost matrices.
This is the KeOps implementation of Algorithm 2 in:

Scetbon, M., Peyré, G., & Cuturi, M. (2022, June).
Linear-time gromov wasserstein distances using low rank couplings and costs.
In International Conference on Machine Learning (pp. 19347-19365). PMLR.
"""

import torch
from pykeops.torch import LazyTensor

from solvers.gromov_wasserstein.generic.embedding_based import EmbeddingBasedGW
from utils.implementation.gw_losses import gw_loss_from_lowrank
from utils.implementation.initializations import initialize_lowrankgw
from utils.math.dimension_reduction import low_rank_approximation


class LowRankGW(EmbeddingBasedGW):
    """
    Gromov-Wasserstein solver using low-rank approximations of cost matrices.

    This solver factorizes the base cost matrices into low-rank representations C_x = C_x1 @ C_x2.T and
    C_y = C_y1 @ C_y2.T, significantly reducing memory requirements and computational cost for large point clouds.

    The low-rank representations are computed by applying truncated PCA on input matrices.

    Attributes:
        X_init (torch.Tensor): Original source point cloud, shape (N, D_x)
        Y_init (torch.Tensor): Original target point cloud, shape (M, D_y)
        cost_x (callable): Cost function for source space
        cost_y (callable): Cost function for target space
        Cx1 (torch.Tensor): First factor of source cost matrix, shape (N, approx_dim_x)
        Cx2 (torch.Tensor): Second factor of source cost matrix, shape (N, approx_dim_x)
        Cy1 (torch.Tensor): First factor of target cost matrix, shape (M, approx_dim_y)
        Cy2 (torch.Tensor): Second factor of target cost matrix, shape (M, approx_dim_y)
        approx_dim_x (int): Approximation dimension for source space
        approx_dim_y (int): Approximation dimension for target space
    """

    def __init__(self, X_init, Y_init, costs, approx_dims=10, **kwargs):
        """
        Initialize the low-rank Gromov-Wasserstein solver.

        Args:
            X_init: Source point cloud, shape (N, D_x)
            Y_init: Target point cloud, shape (M, D_y)
            costs: Cost function(s) for computing base costs. Can be:
                   - Single function: used for both X and Y
                   - Tuple (cost_x, cost_y): separate functions for each space
            approx_dims: Low-rank approximation dimension(s). Can be:
                        - Single int: same dimension for both spaces
                        - Tuple (dim_x, dim_y): different dimensions
            **kwargs: Additional arguments passed to EmbeddingBasedGW
        """
        self.X_init = X_init
        self.Y_init = Y_init

        self.approx_dim_x, self.approx_dim_y = approx_dims if isinstance(approx_dims,
                                                                         tuple) else approx_dims, approx_dims

        cost_x, cost_y = costs if isinstance(costs, tuple) else costs, costs
        self.Cx1, self.Cx2 = low_rank_approximation(X_init, cost_x, self.approx_dim_x)
        self.Cy1, self.Cy2 = low_rank_approximation(Y_init, cost_y, self.approx_dim_y)

        super().__init__(X=X_init, Y=Y_init, costs=costs, lazy=True, **kwargs)

    def parameters(self):
        params = super().parameters()
        params['approx_dim_x'] = self.approx_dim_x
        params['approx_dim_y'] = self.approx_dim_y

        return params

    def approx_base_costs(self, i=None, which='x'):
        """
        Compute the full or partial approximated base cost matrices.

        Args:
            i: If provided, compute only row i. If None, compute full matrix
            which: 'x' for source space or 'y' for target space

        Returns:
            torch.Tensor: Approximated cost matrix or row
        """
        if i is None:
            return self.Cx1 @ self.Cx2.T if which == 'x' else self.Cy1 @ self.Cy2.T
        else:
            return self.Cx1[i] @ self.Cx2.T if which == 'x' else self.Cy1[i] @ self.Cy2.T

    def initialize_potential(self):
        """
        Initialize the potential matrix.

        For low-rank GW, the potential is an (approx_dim_x, approx_dim_y) matrix factorizing the low-rank cost
        approximation.

        Returns:
            torch.Tensor: Initial potential matrix, shape (D_x, D_y)
        """
        return initialize_lowrankgw(self.Cx2, self.Cy1, self.a, self.b, initialization_mode=None)

    def update_potential(self):
        """
        Update the potential matrix based on current transport plan.

        The potential corresponds to the low-dimensional matrix:
        Z = 4 * (Cx2^T P Cy1)

        Returns:
            torch.Tensor: Updated potential matrix, shape (approx_dim_x, approx_dim_y)

        """
        Dx2_i = LazyTensor(self.Cx2[:, None, :])  # (N, 1, D)
        P_ij = self.sinkhorn_solver.transport_plan()  # (N, M)
        DxP = (P_ij * Dx2_i).sum(dim=0)  # (M, D)

        Z = 4 * torch.einsum("ju,jv->uv", DxP, self.Cy1)  # (D, D')

        return Z

    def cost_matrix(self, lazy=True):
        """
        Compute the cost matrix for Sinkhorn using low-rank factors.

        Computes: C_ij = -(Cx1 @ Z @ Cy2^T)_ij efficiently using the factored form.

        Args:
            lazy: If True, return C_ij as a LazyTensor

        Returns:
            torch.Tensor or LazyTensor: Cost matrix, shape (N, M)
        """
        Dx1Z = torch.einsum("ik,kl->il", self.Cx1, self.Z)  # (N, D')

        Dx1Z_i = LazyTensor(Dx1Z[:, None, :]) if lazy else Dx1Z[:, None, :]  # (N, 1, D')
        Dy2_j = LazyTensor(self.Cy2[None, :, :]) if lazy else self.Cy2[None, :, :]  # (1, M, D')

        C_ij = (Dx1Z_i * Dy2_j).sum(dim=2)  # (N, M)

        return - C_ij

    def loss(self, include_divergence=True, approx=False, **kwargs):
        """
        Compute the Gromov-Wasserstein loss.

        Args:
            include_divergence: If True, include entropic regularization term
            approx: If True, use low-rank approximation for fast computation.
                    If False, compute exact loss using full cost matrices
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar loss value
        """
        if approx:
            P = self.transport_plan(lazy=True)
            return gw_loss_from_lowrank(self.Cx1, self.Cx2, self.Cy1, self.Cy2, P=P, a=self.a, b=self.b, eps=self.eps,
                                        include_divergence=include_divergence)
        else:
            return super().loss(include_divergence=include_divergence, **kwargs)

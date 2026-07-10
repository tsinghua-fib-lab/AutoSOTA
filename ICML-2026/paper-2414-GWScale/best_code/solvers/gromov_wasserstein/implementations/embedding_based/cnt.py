"""
Conditionally Negative Type Gromov-Wasserstein Solver

Implements a GW solver for costs conditionally of negative types, by embedding them into Euclidean spaces
using Kernel PCA.
"""

import torch

from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
from utils.implementation.gw_losses import gw_loss_from_points
from utils.implementation.kernels import reduce_kernel
from utils.math.costs import euclidean


class CntGW(QuadraticGW):
    """
    Gromov-Wasserstein solver for costs conditionally of negative type (CNT).

    This solver first embeds the input spaces into lower-dimensional Euclidean spaces using kernel methods that preserve
    negative type distances. It then solves the quadratic GW problem on these embeddings using QuadraticGW

    Attributes:
       X_init (torch.Tensor): Original source data
       Y_init (torch.Tensor): Original target data
       X (torch.Tensor): Embedded source points, shape (N, approx_dim_x)
       Y (torch.Tensor): Embedded target points, shape (M, approx_dim_y)
       cost_x (callable): Cost function for source space
       cost_y (callable): Cost function for target space
       approx_dim_x (int): Embedding dimension for source
       approx_dim_y (int): Embedding dimension for target
    """

    def __init__(self, X_init, Y_init, costs, a=None, b=None, approx_dims=10, **kwargs):
        """
        Initialize the negative type Gromov-Wasserstein solver.

        Args:
            X_init: Source point cloud, shape (N, D_x)
            Y_init: Target point cloud, shape (M, D_y)
            costs: Cost function(s) for computing base costs. Can be:
                   - Single function: used for both X and Y
                   - Tuple (cost_x, cost_y): separate functions for each space
            approx_dims: Embedding dimension(s). Can be:
                        - Single int: same dimension for both spaces
                        - Tuple (dim_x, dim_y): different dimensions
            a: Source distribution. If None, uses uniform
            b: Target distribution. If None, uses uniform
            approx_dims: Embedding dimension(s). Can be int or tuple
            **kwargs: Additional arguments passed to QuadraticGW
        """
        self.X_init = X_init
        self.Y_init = Y_init

        a = a if a is not None else torch.ones((X_init.shape[0],), dtype=torch.float32) / X_init.shape[0]
        b = b if b is not None else torch.ones((Y_init.shape[0],), dtype=torch.float32) / Y_init.shape[0]

        cost_x, cost_y = costs if isinstance(costs, tuple) else costs, costs
        approx_dim_x, approx_dim_y = approx_dims if isinstance(approx_dims, tuple) else approx_dims, approx_dims

        self.X = reduce_kernel(X_init, a, cost_x, approx_dim_x)
        self.Y = reduce_kernel(Y_init, b, cost_y, approx_dim_y)

        super().__init__(X=self.X, Y=self.Y, a=a, b=b, **kwargs)

        self.cost_x, self.cost_y = cost_x, cost_y
        self.approx_dim_x, self.approx_dim_y = approx_dim_x, approx_dim_y

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
        U = self.X if which == 'x' else self.Y
        if i is None:
            return euclidean(U[:, None], U[None, :], p=2)
        else:
            return euclidean(U[i], U[None, :], p=2).squeeze()

    def loss(self, include_divergence=True, approx=False, **kwargs):
        """
        Compute the Gromov-Wasserstein loss.

        Args:
            include_divergence: If True, include entropic regularization
            approx: If True, use embedded space. If False, use original space
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar loss value
        """
        if approx:
            return super().loss(include_divergence)
        else:
            return gw_loss_from_points(X=self.X_init, Y=self.Y_init, cost_x=self.cost_x, cost_y=self.cost_y,
                                       P=self.transport_plan(lazy=False, device='cuda'), a=self.a, b=self.b,
                                       eps=self.eps,
                                       include_divergence=include_divergence)

"""
Multiscale Conditionally Negative Type Gromov-Wasserstein Solver

This module implements the multiscale version of CNT Gromov-Wasserstein.

"""
from solvers.gromov_wasserstein.implementations.embedding_based.cnt import CntGW
from solvers.gromov_wasserstein.multiscale.multiscale_quadratic import MultiscaleQuadraticGW


class MultiscaleCntGW(CntGW, MultiscaleQuadraticGW):
    """
    Multiscale solver using conditional negative type (CNT) embeddings.

    This solver combines CNT embedding with Multiscale optimization. It inherists from both CntGromovSolver
    and MultiscaleQuadraticGW.

    Attributes:
        X_init (torch.Tensor): Original source data before embedding
        Y_init (torch.Tensor): Original target data before embedding
        X (torch.Tensor): Embedded source points
        Y (torch.Tensor): Embedded target points
        cost_x (callable): Cost function for source space
        cost_y (callable): Cost function for target space
        approx_dim_x (int): Embedding dimension for source
        approx_dim_y (int): Embedding dimension for target
        ratio (float): Coarsening ratio for multiscale approach
    """

    def __init__(self, X_init, Y_init, costs, approx_dims=10, **kwargs):
        """
        Initialize multiscale CNT Gromov-Wasserstein solver.

        The initialization process:
        1. Computes CNT embeddings of both spaces into Euclidean space
        2. Creates coarsened versions of embedded point clouds
        3. Sets up both coarse and fine solvers

        Args:
            X_init: Original source data (any metric space)
            Y_init: Original target data (any metric space)
            costs: Cost/distance function(s) - single or tuple (cost_x, cost_y)
            approx_dims: Embedding dimension(s) - int or tuple (dim_x, dim_y)
            **kwargs: Additional arguments including:
                - ratio: Coarsening ratio (default from MultiscaleQuadraticGW)
                - a, b: Distributions
                - eps: Regularization
                - Other QuadraticGW parameters

        """
        super().__init__(X_init=X_init, Y_init=Y_init, costs=costs, approx_dims=approx_dims, **kwargs)

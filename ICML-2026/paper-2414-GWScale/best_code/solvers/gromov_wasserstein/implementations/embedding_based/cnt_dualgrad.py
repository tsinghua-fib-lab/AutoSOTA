"""
Conditional Negative Type GW with Dual Gradient Descent

Combines CNT embeddings with the dual gradient of:

Rioux, G., Goldfeld, Z., & Kato, K. (2024).
Entropic gromov-wasserstein distances: Stability and algorithms.
Journal of Machine Learning Research, 25(363), 1-52.
"""

from solvers.gromov_wasserstein.implementations.embedding_based.cnt import CntGW
from solvers.gromov_wasserstein.implementations.embedding_based.quadratic_dualgrad import QuadraticDualGradGW


class CntDualGradGW(CntGW, QuadraticDualGradGW):
    """
    CNT embedding solver with dual gradient acceleration.

    It inherists from both CntGromovSolver and QuadraticDualGradGromovSolver.

    Attributes:
        X_init (torch.Tensor): Original source data (any metric space)
        Y_init (torch.Tensor): Original target data (any metric space)
        X (torch.Tensor): CNT-embedded source points, shape (N, approx_dim_x)
        Y (torch.Tensor): CNT-embedded target points, shape (M, approx_dim_y)
        L (float): Lipschitz constant for gradient step size
        M (float): Feasibility bound for projection
        k (int): Iteration counter for momentum scheduling
        W (torch.Tensor): Momentum state for acceleration
    """

    def __init__(self, X_init, Y_init, costs, approx_dims=10, **kwargs):
        """
        Initialize CNT solver with dual gradient acceleration.

        Args:
            X_init: Original source data in any metric space
            Y_init: Original target data in any metric space
            costs: Cost/distance function(s) - single or tuple (cost_x, cost_y)
            approx_dims: Embedding dimension(s) for CNT - int or tuple
            **kwargs: Additional arguments including:
            - L: Manual Lipschitz constant (optional)
            - a, b: Distributions
            - eps: Regularization parameter
            - Other QuadraticGW/GW parameters

        """
        super().__init__(X_init=X_init, Y_init=Y_init, costs=costs, approx_dims=approx_dims, **kwargs)

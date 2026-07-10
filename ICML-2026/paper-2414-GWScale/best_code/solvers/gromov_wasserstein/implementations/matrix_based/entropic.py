"""
Entropic Gromov-Wasserstein Solver

Implements the standard entropic Gromov-Wasserstein distance computation using full precomputed cost
matrices. It implements the algorithm of:


Peyré, G., Cuturi, M., & Solomon, J. (2016, June).
Gromov-wasserstein averaging of kernel and distance matrices.
In International conference on machine learning (pp. 2664-2672). PMLR.
"""

import torch

from solvers.gromov_wasserstein.generic.matrix_based import MatrixBasedGW
from utils.implementation.initializations import initialize_entropicgw


class EntropicGW(MatrixBasedGW):
    """
    Standard entropic Gromov-Wasserstein solver.

    This is the basic matrix-based implementation of the Gromov-Wasserstein distance with entropic regularization.
    It works with full precomputed cost matrices and  does not exploit any special structure for computational
    efficiency.

    The solver alternates between:
        1. Solving an entropy-regularized optimal transport problem via Sinkhorn
        2. Updating the cost matrix based on the Gromov-Wasserstein gradient

    Attributes:
        Cx (torch.Tensor): Pairwise cost matrix for source space, shape (N, N)
        Cy (torch.Tensor): Pairwise cost matrix for target space, shape (M, M)
        a (torch.Tensor): Source distribution (probability weights), shape (N,)
        b (torch.Tensor): Target distribution (probability weights), shape (M,)
        Z (torch.Tensor): Current cost matrix for optimal transport, shape (N, M)
    """

    def __init__(self, Cx, Cy, **kwargs):
        """
        Initialize the entropic Gromov-Wasserstein solver.

        Args:
            **kwargs: Keyword arguments passed to MatrixBasedGW. Expected arguments include:
                - Cx (torch.Tensor): Source space cost matrix, shape (N, N)
                - Cy (torch.Tensor): Target space cost matrix, shape (M, M)
                - a (torch.Tensor, optional): Source distribution. Defaults to uniform
                - b (torch.Tensor, optional): Target distribution. Defaults to uniform
                - eps (float, optional): Entropic regularization parameter. Default 0.1
                - numItermax (int, optional): Maximum number of iterations. Default 1000
                - stopThr (float, optional): Convergence threshold
                - Additional parameters inherited from SinkhornBasedGW
        """
        super().__init__(Cx=Cx, Cy=Cy, lazy=False, **kwargs)

    def initialize_potential(self):
        """
        Initialize the cost matrix (potential) for the Gromov-Wasserstein problem.

        The initialization computes an initial guess for the cost matrix Z that will be used in the first Sinkhorn
        iteration.

        Returns:
            torch.Tensor: Initial cost matrix Z, shape (N, M)
        """
        return initialize_entropicgw(self.Cx, self.Cy, self.a, self.b, self.initialization_mode)

    def update_potential(self):
        """
        Update the cost matrix based on the current transport plan.

        This method implements the Gromov-Wasserstein gradient update step. Given the
        current optimal transport plan P, it computes the new cost matrix as:

            Z_new[i,j] = -4 * sum_{k,l} P[k,l] * Cx[i,k] * Cy[j,l]

        Returns:
            torch.Tensor: Updated cost matrix Z, shape (N, M)
        """
        P = self.sinkhorn_solver.transport_plan()  # (N, M)
        return - 4 * torch.einsum('ij,iu,jv->uv', P, self.Cx, self.Cy)

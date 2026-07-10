"""
Proximal Gromov-Wasserstein Solver

This module implements the Gromov-Wasserstein distance with an additional proximal regularization term.
This is an adaptation of the algorithm of:

Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019, May).
Gromov-wasserstein learning for graph matching and node embedding.
In International conference on machine learning (pp. 6932-6941). PMLR.
"""

import torch

from solvers.gromov_wasserstein.generic.matrix_based import MatrixBasedGW
from utils.implementation.gw_losses import gw_loss
from utils.implementation.initializations import initialize_entropicgw


class ProximalGW(MatrixBasedGW):
    """
    Proximal Gromov-Wasserstein solver.

    It iteratively minimizes the following loss:
        GW(Cx, Cy, P_{t+1}) + eps_entropic * KL(P_{t+1} | P_t)

    To do so, it reformulates each step as a classical EntropicGW step with Sinkhorn cost equal to:
        Cprox = C + eps * log(P)

    Attributes:
        Cx (torch.Tensor): Pairwise cost matrix for source space, shape (N, N)
        Cy (torch.Tensor): Pairwise cost matrix for target space, shape (M, M)
        a (torch.Tensor): Source distribution (probability weights), shape (N,)
        b (torch.Tensor): Target distribution (probability weights), shape (M,)
        eps (float): Entropic regularization parameter
    """

    def __init__(self, Cx, Cy, a=None, b=None, eps=0.1, **kwargs):
        """
        Initialize the proximal Gromov-Wasserstein solver.

        Args:
            Cx: Cost matrix for source space, shape (N, N)
            Cy: Cost matrix for target space, shape (M, M)
            a: Source distribution. If None, uses uniform distribution
            b: Target distribution. If None, uses uniform distribution
            eps: Entropic regularization parameter
            **kwargs: Additional arguments passed to MatrixBasedGW including:
                - numItermax (int): Maximum number of iterations
                - stopThr (float): Convergence threshold

        """
        self.eps = eps
        super().__init__(Cx=Cx, Cy=Cy, a=a, b=b, eps=eps, **kwargs)

    def initialize_potential(self):
        """
        Initialize the potential with proximal cost matrix.

        The initial potential includes both the standard GW cost initialization and a proximal term based on the
        product of marginals:
            Z_init = Z_gw - eps * log(a b.T)
        where a b.T is the outer product of the marginal distributions, representing the independent coupling.

        Returns:
            torch.Tensor: Initial potential matrix Z, shape (N, M)

        """
        cost_matrix = initialize_entropicgw(self.Cx, self.Cy, self.a, self.b, self.initialization_mode)
        logP = (self.a.log()[:, None] + self.b.log()[None, :])
        return cost_matrix - self.eps * logP

    def update_potential(self):
        """
        Update the potential with the proximal regularization term.

        The update combines the standard Gromov-Wasserstein gradient with a
        proximal term based on the current transport plan:
            Z_new = -4 * sum_{k,l} P[k,l] * Cx[i,k] * Cy[j,l] - eps * log(P)

        Returns:
            torch.Tensor: Updated cost matrix Z, shape (N, M)

        Note:
            The absolute value in P.abs() is used for numerical safety, though in practice P should always be positive.
        """
        P = self.sinkhorn_solver.transport_plan()  # (N, M)
        return - 4 * torch.einsum('ij,iu,jv->uv', P, self.Cx, self.Cy) - self.eps * P.abs().log()

    def loss(self, **kwargs):
        """
        Compute the Gromov-Wasserstein loss.

        Since the proximal solver converges to minima of the unregularied GW problem, the loss only computes the GW loss
        (without adding the KL divergence regularization term present in other solvers).

        Args:
            **kwargs: Additional arguments (unused, kept for interface compatibility)


        Returns:
            torch.Tensor: Scalar GW loss value

        """
        P = self.sinkhorn_solver.transport_plan()
        return gw_loss(self.Cx, self.Cy, P, self.a, self.b, self.sinkhorn_solver.eps, include_divergence=False)

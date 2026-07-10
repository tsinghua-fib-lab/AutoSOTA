"""
Proximal Kernel Gromov-Wasserstein Solver

A kernel-based variant of ProximalGW.
"""

import torch

from solvers.gromov_wasserstein.generic.matrix_based import MatrixBasedGW
from utils.implementation.gw_losses import gw_loss
from utils.implementation.initializations import initialize_kernelgw
from utils.implementation.kernels import kernel_from_costmatrix


class ProximalKernelGW(MatrixBasedGW):
    """
    Gromov-Wasserstein solver combining kernel formulation with proximal regularization.

    Attributes:
        Cx (torch.Tensor): Original cost matrix for source space, shape (N, N)
        Cy (torch.Tensor): Original cost matrix for target space, shape (M, M)
        Kx (torch.Tensor): Kernel matrix for source space, shape (N, N)
        Ky (torch.Tensor): Kernel matrix for target space, shape (M, M)
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
        eps (float): Entropic regularization parameter
    """

    def __init__(self, Cx, Cy, a=None, b=None, eps=0.1, **kwargs):
        """
        Initialize the proximal kernel Gromov-Wasserstein solver.

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
        a = torch.ones((Cx.shape[0],), dtype=Cx.dtype, device=Cx.device) / Cx.shape[0] if a is None else a
        b = torch.ones((Cy.shape[0],), dtype=Cy.dtype, device=Cy.device) / Cy.shape[0] if b is None else b

        self.Kx = kernel_from_costmatrix(Cx, a, center=True)
        self.Ky = kernel_from_costmatrix(Cy, b, center=True)

        self.eps = eps

        super().__init__(lazy=False, Cx=Cx, Cy=Cy, a=a, b=b, eps=eps, **kwargs)

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
        kernel_matrix = initialize_kernelgw(self.Kx, self.Ky, self.a, self.b, self.initialization_mode)
        logP = (self.a.log()[:, None] + self.b.log()[None, :])
        return kernel_matrix - self.eps * logP

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
        KyP = torch.einsum("jl,kl->jk", self.Ky, self.sinkhorn_solver.transport_plan())

        Z = - 16 * torch.einsum("ik,jk->ij", self.Kx, KyP)
        Z -= 4 * self.Kx.diag()[:, None] * self.Ky.diag()[None, :]

        return Z - self.eps * P.abs().log()

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

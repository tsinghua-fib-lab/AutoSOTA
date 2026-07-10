"""
Kernel-Based Gromov-Wasserstein Solver

Implements a Gromov-Wasserstein solver for costs Conditionally of Negative Type (CNT) using kernel matrices instead of
cost matrices.
"""

import torch

from solvers.gromov_wasserstein.generic.matrix_based import MatrixBasedGW
from utils.implementation.gw_losses import gw_loss_from_kernel, gw_loss
from utils.implementation.initializations import initialize_kernelgw
from utils.implementation.kernels import kernel_from_costmatrix


class KernelGW(MatrixBasedGW):
    """
    Gromov-Wasserstein solver using kernel matrix formulation adapted to costs Conditionally of Negative Type (CNT).

    Instead of working directly with cost/distance matrices Cx and Cy, this solver converts them into kernel
    matrices Kx and Ky.

    The solver maintains both kernel matrices (for computation) and cost matrices (for exact loss evaluation), with
    device management to optimize memory usage.

    Attributes:
        Cx (torch.Tensor): Original cost matrix for source space, shape (N, N)
        Cy (torch.Tensor): Original cost matrix for target space, shape (M, M)
        Kx (torch.Tensor): Kernel matrix for source space, shape (N, N)
        Ky (torch.Tensor): Kernel matrix for target space, shape (M, M)
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
    """

    def __init__(self, Cx, Cy, a=None, b=None, **kwargs):
        """
        Initialize the kernel-based Gromov-Wasserstein solver.

        Args:
            Cx: Cost/distance matrix for source space, shape (N, N)
            Cy: Cost/distance matrix for target space, shape (M, M)
            a: Source distribution. If None, uses uniform distribution
            b: Target distribution. If None, uses uniform distribution
            **kwargs: Additional arguments passed to MatrixBasedGW, including:
                - eps (float): Entropic regularization parameter
                - numItermax (int): Maximum number of outer iterations
                - Other SinkhornBasedGW parameters

        Note:
            The cost matrices are automatically converted to centered kernel matrices during initialization. The
             original cost matrices are retained for computing exact loss values when needed.
        """
        a = torch.ones((Cx.shape[0],), dtype=Cx.dtype, device=Cx.device) / Cx.shape[0] if a is None else a
        b = torch.ones((Cy.shape[0],), dtype=Cy.dtype, device=Cy.device) / Cy.shape[0] if b is None else b

        self.Kx = kernel_from_costmatrix(Cx, a, center=True)
        self.Ky = kernel_from_costmatrix(Cy, b, center=True)

        super().__init__(lazy=False, Cx=Cx, Cy=Cy, a=a, b=b, **kwargs)

    def to(self, device, kernel_only=True):
        """
        Move tensors to the specified device with flexible memory management.

        The kernel_only parameter allows keeping cost matrices on CPU while moving kernel matrices to GPU, which can be
        beneficial to avoid GPU memory overflow when Cx and Cy are large matrices.

        Args:
            device: Target device (e.g., 'cuda:0', 'cpu', torch.device('cuda'))
            kernel_only: If True, only move kernel matrices to the target device while keeping cost matrices on their
                            original device (typically CPU).
                         If False, move all tensors including cost matrices.
                         Default is True to save GPU memory.

        """
        self.sinkhorn_solver.to(device)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.Z = self.Z.to(device)
        self.Z_new = self.Z_new.to(device)
        self.Kx = self.Kx.to(device)
        self.Ky = self.Ky.to(device)

        if not kernel_only:
            self.Cx = self.Cx.to(device)
            self.Cy = self.Cy.to(device)

    def initialize_potential(self):
        """
        Initialize the potential (cost) matrix using kernel matrices.

        Returns:
            torch.Tensor: Initial potential matrix Z, shape (N, M)

        """
        return initialize_kernelgw(self.Kx, self.Ky, self.a, self.b, self.initialization_mode)

    def update_potential(self):
        """
        Update the cost matrix using the kernel formulation.

        This method computes the Gromov-Wasserstein gradient in kernel space. Given the current transport plan P, the update is:
            Z[i,j] = -16 * sum_{kl} Kx[i,k] * Ky[j,l] * P[k,l] - 4 * Kx[i,i] * Ky[j,j]

        Returns:
            torch.Tensor: Updated cost matrix Z, shape (N, M)

        """
        KyP = torch.einsum("jl,kl->jk", self.Ky, self.sinkhorn_solver.transport_plan())

        Z = - 16 * torch.einsum("ik,jk->ij", self.Kx, KyP)
        Z -= 4 * self.Kx.diag()[:, None] * self.Ky.diag()[None, :]

        return Z

    def loss(self, include_divergence=True, approx=False, **kwargs):
        """
        Compute the Gromov-Wasserstein loss.

        This method can compute the loss in two ways:
        1. Approximate: Use kernel-based loss computation (inaccurate when P does not satisfy marginal constraints).
        2. Exact: Use original cost matrices for true GW loss (may require to switch Cx and Cy from device).

        Args:
            include_divergence: If True, include entropic regularization term
            approx: If True, compute loss using kernel approximation.
                    If False, compute exact loss using original cost matrices.
                    Default is False for accuracy.
            **kwargs: Additional arguments (unused, kept for interface compatibility)

        Returns:
            torch.Tensor: Scalar Gromov-Wasserstein loss value

        Note:
            When approx=False, the method temporarily moves cost matrices to the same device as the transport plan for
            computation, then moves them back. This device juggling is done to balance GPU memory usage.
        """
        P = self.sinkhorn_solver.transport_plan()

        if approx:
            energy = gw_loss_from_kernel(self.Kx, self.Ky, P, self.a, self.b, self.sinkhorn_solver.eps,
                                         include_divergence)
        else:
            old_device = self.Cx.device

            self.Cx, self.Cy = self.Cx.to(device=P.device), self.Cy.to(device=P.device)
            self.Kx, self.Ky = self.Kx.to(device=old_device), self.Ky.to(device=old_device)

            energy = gw_loss(self.Cx, self.Cy, P, self.a, self.b, self.sinkhorn_solver.eps, include_divergence)

            self.Cx, self.Cy = self.Cx.to(device=old_device), self.Cy.to(device=old_device)
            self.Kx, self.Ky = self.Kx.to(device=P.device), self.Ky.to(device=P.device)

        return energy

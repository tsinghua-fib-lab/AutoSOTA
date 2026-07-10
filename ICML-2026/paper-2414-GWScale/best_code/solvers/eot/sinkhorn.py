import logging
import math

import torch
from pykeops.torch import LazyTensor

from utils.math.functions import kl_divergence


class SinkhornSolver:
    """
    Solver for entropic optimal transport using the Sinkhorn algorithm.

    The Sinkhorn algorithm solves the regularized optimal transport problem:
        min_{P} <C, P> + eps * KL(P | a b.T)

    where C is the cost matrix, P is the transport plan, and eps is the regularization parameter.

    Attributes:
        a (torch.Tensor): Source distribution (marginal constraints), shape (N,)
        b (torch.Tensor): Target distribution (marginal constraints), shape (M,)
        C (torch.Tensor): Cost matrix, shape (N, M)
        f (torch.Tensor): Dual potential for source, shape (N,)
        g (torch.Tensor): Dual potential for target, shape (M,)
        lazy (bool): Whether to use lazy tensor operations (set to True if C is a LazyTensor)
        eps (float): Entropic regularization parameter
        numItermax (int): Maximum number of Sinkhorn iterations
        stopThr (float or None): Stopping threshold for convergence
        last_numIters (int): Number of iterations performed in last solve
        symmetrize (bool): Whether to symmetrize the updates
        annealing (str or None): Annealing scheduling for entropic regularization
        verbose (bool): Whether to print iteration information
    """

    def __init__(self,
                 a: torch.Tensor,
                 b: torch.Tensor,
                 C: torch.Tensor | None = None,
                 f: torch.Tensor | None = None,
                 g: torch.Tensor | None = None,
                 lazy: bool = False,
                 eps: float = 0.1,
                 numItermax: int = 1000,
                 stopThr: float | None = None,
                 symmetrize: bool = True,
                 annealing: str = None,
                 verbose: bool = False,
                 ):
        """
        Initialize the Sinkhorn solver.

        Args:
            a: Source distribution (must sum to 1), shape (N,)
            b: Target distribution (must sum to 1), shape (M,)
            C: Cost matrix, shape (N, M). Can be None if set later
            f: Initial dual potential for source. If None, initialized to zeros
            g: Initial dual potential for target. If None, initialized to zeros
            lazy: If True, computations are done with LazyTensors
            eps: Entropic regularization parameter
            numItermax: Maximum number of Sinkhorn iterations
            stopThr: Convergence threshold based on marginal errors. If None, runs all iterations
            symmetrize: If True, use symmetric Sinkhorn updates
            annealing: If 'linear' or 'root', decrease the regularization at each iteration to get sharper transport
                       plans. If None, regularization is kept constant. Annealing is incompatible with stopThr
            verbose: If True, log iteration information
        """
        self.a = a
        self.b = b

        self.C = C
        self.f = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device) if f is None else f
        self.g = torch.zeros(b.shape[0], dtype=b.dtype, device=b.device) if g is None else g

        self.eps = eps
        self.numItermax = numItermax
        self.stopThr = stopThr

        self.last_numIters = 0

        self.symmetrize = symmetrize
        self.annealing = annealing

        self.lazy = lazy

        self.verbose = verbose

    def parameters(self):
        """
        Get solver parameters as a dictionary.

        Returns:
            dict: Dictionary containing solver configuration parameters
        """
        return {'eps': self.eps,
                'numItermax': self.numItermax,
                'stopThr': self.stopThr,
                'symmetrize': self.symmetrize,
                'annealing': self.annealing,
                'lazy': self.lazy}

    def to(self, device):
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
        """
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.C = self.C.to(device) if self.C is not None else None
        self.f = self.f.to(device)
        self.g = self.g.to(device)

    def clear(self):
        """Reset dual potentials to zero."""
        self.f = torch.zeros(self.a.shape[0], dtype=self.a.dtype, device=self.a.device)
        self.g = torch.zeros(self.b.shape[0], dtype=self.a.dtype, device=self.b.device)

    def wrap_cost(self, C, lazy=None, device=None):
        """
        Wrap cost matrix for computation, handling lazy tensor operations.

        Args:
            C: Matrix to wrap, shape (N,M)
            lazy: If True, wrap in LazyTensor. If None, uses self.lazy
            device: Target device for non-lazy operations

        Returns:
            tensor: Properly shaped and wrapped cost matrix
        """
        if lazy is None:
            lazy = self.lazy

        if lazy:
            return C
        else:
            return C.view((*C.shape, 1)).to(device)

    def wrap_tensor(self, u, dim=0, lazy=None, device=None):
        """
        Wrap tensor for computation, handling lazy tensor operations.

        Args:
            u: Tensor to wrap, shape (N,) or (M,)
            dim: Which dimension the tensor corresponds to
            lazy: If True, wrap in LazyTensor. If None, uses self.lazy
            device: Target device for non-lazy operations

        Returns:
            tensor: Properly shaped and wrapped tensor
        """
        if lazy is None:
            lazy = self.lazy

        u_wrapped = u[:, None, None] if dim == 0 else u[None, :, None]

        if lazy:
            return LazyTensor(u_wrapped)
        else:
            return u_wrapped.to(device)

    def annealing_schedule(self, it):
        """
        Determine the annealing schedule at current iteration.

        If 'root', use a squared root decrease of eps; if 'linear', use a linear decrease. The 10% last iterations are
        done at regularization self.eps.

        Args:
            it: Current Sinkhorn iteration

        Returns:
            float: Regularization parameter to use in current iteration
        """
        if self.annealing == 'root' and it < int(0.9 * self.numItermax) - 1:
            return self.eps * math.sqrt(int(0.9 * self.numItermax)) / math.sqrt(it + 1)
        elif self.annealing == 'linear' and it < int(0.9 * self.numItermax) - 1:
            return self.eps * int(0.9 * self.numItermax) / (it + 1)
        else:
            return self.eps

    def marginal_errors(self, psi=None, phi=None, eps=None):
        """
        Compute marginal constraint errors.

        The marginal errors measure how well the current transport plan satisfies the marginal constraints P1 = a and P^T1 = b.

        Args:
            psi: Precomputed log-sum-exp over target dimension. If None, computed internally
            phi: Precomputed log-sum-exp over source dimension. If None, computed internally
            eps: Entropic regularization parameter. If None, use self.eps

        Returns:
            tuple: (err_a, err_b) where err_a is the error for source marginals and err_b is the error for target marginals
        """
        if eps is None:
            eps = self.eps

        if psi is None or phi is None:
            f_i, g_j, C_ij = self.wrap_tensor(self.f, dim=0), self.wrap_tensor(self.g, dim=1), self.wrap_cost(self.C)
            alogs_i, blogs_j = self.wrap_tensor(self.a.log(), dim=0), self.wrap_tensor(self.b.log(), dim=1)

            phi = - eps * (((f_i - C_ij) / eps) + alogs_i).logsumexp(dim=0).squeeze()
            psi = - eps * (((g_j - C_ij) / eps) + blogs_j).logsumexp(dim=1).squeeze()

        margin_a = ((self.f - psi) / eps).exp()
        margin_b = ((self.g - phi) / eps).exp()

        err_a = (self.a * ((self.a * (1. - margin_a)).abs())).sum()
        err_b = (self.b * ((self.b * (1. - margin_b)).abs())).sum()

        return err_a, err_b

    def solve(self, logger=None):
        """
        Solve the optimal transport problem using the Sinkhorn algorithm.
        
        Performs iterative scaling updates to the dual potentials f and g until convergence or maximum iterations
        is reached. Updates are done in-place.

        Args:
            logger: Logger instance for the solver. If None, created internally
        """
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)

        eps = self.annealing_schedule(it=0)
        self.last_numIters = self.numItermax

        with torch.no_grad():
            f_i, g_j, C_ij = self.wrap_tensor(self.f, dim=0), self.wrap_tensor(self.g, dim=1), self.wrap_cost(self.C)
            alogs_i, blogs_j = self.wrap_tensor(self.a.log(), dim=0), self.wrap_tensor(self.b.log(), dim=1)

            phi = - eps * (((f_i - C_ij) / eps) + alogs_i).logsumexp(dim=0).squeeze() if self.symmetrize else None
            psi = - eps * (((g_j - C_ij) / eps) + blogs_j).logsumexp(dim=1).squeeze()

            for it in range(self.numItermax):
                eps = self.annealing_schedule(it)

                if self.symmetrize:
                    self.f[:] = (self.f + psi) / 2
                    self.g[:] = (self.g + phi) / 2

                    phi = - eps * (((f_i - C_ij) / eps) + alogs_i).logsumexp(dim=0).squeeze()
                    psi = - eps * (((g_j - C_ij) / eps) + blogs_j).logsumexp(dim=1).squeeze()
                else:
                    self.f[:] = psi
                    phi = - eps * (((f_i - C_ij) / eps) + alogs_i).logsumexp(dim=0).squeeze()

                    self.g[:] = phi
                    psi = - eps * (((g_j - C_ij) / eps) + blogs_j).logsumexp(dim=1).squeeze()

                err_a, err_b = self.marginal_errors(psi=psi, phi=phi, eps=eps)

                if self.verbose:
                    msg = f"\t Sinkhorn iter: {it} - residuals = {err_a:.4G}, {err_b:.4G}"
                    logger.info(msg)

                if self.stopThr is not None and err_a < self.stopThr and err_b < self.stopThr:
                    self.last_numIters = it
                    break

    def transport_plan(self, C=None, lazy=None, device=None):
        """
        Compute the optimal transport plan from the current dual potentials.

        The transport plan P is computed as:
        P_ij = exp((f_i + g_j - C_ij) / eps)

        Args:
            C: Cost matrix to use. If None, uses self.C
            lazy: Whether to use lazy tensor operations. If None, uses self.lazy
            device: Device for computation (non-lazy mode)

        Returns:
            torch.Tensor or LazyTensor: Transport plan, shape (N, M)
        """
        if C is None:
            C = self.C

        f_i = self.wrap_tensor(self.f, dim=0, lazy=lazy, device=device)
        g_j = self.wrap_tensor(self.g, dim=1, lazy=lazy, device=device)
        C_ij = self.wrap_cost(C, lazy=lazy, device=device)
        a_i = self.wrap_tensor(self.a, dim=0, lazy=lazy, device=device)
        b_j = self.wrap_tensor(self.b, dim=1, lazy=lazy, device=device)

        return (a_i * b_j * ((f_i + g_j - C_ij) / self.eps).exp()).sum(dim=-1)  # (N, M)

    def loss(self, include_divergence=True):
        """
        Compute the optimal transport loss (primal objective).

        Computes: <C, P> + eps * KL(P | a b.T) if include_divergence=True
                  <C, P> otherwise

        Args:
            include_divergence: If True, include the KL divergence term

        Returns:
            torch.Tensor: Scalar loss value
        """
        P = self.transport_plan(lazy=True)
        loss = (P * self.C).sum()
        if include_divergence: loss += self.eps * kl_divergence(self.a[:, None, None], self.b[None, :, None], P)

    def dual_loss(self, C=None, lazy=None, device=None):
        """
        Compute the dual objective of the optimal transport problem. This loss can be differentiated to obtain EOT
        gradients w.r.t. the cost matrix C.

        The dual objective is: <a, f> + <b, g>
        where f and g are recomputed from the marginal constraints.

        Args:
            C: Cost matrix to use. If None, uses self.C
            lazy: Whether to use lazy tensor operations. If None, uses self.lazy
            device: Device for computation (non-lazy mode)

        Returns:
            torch.Tensor: Scalar dual loss value
        """
        if C is None:
            C = self.C

        f_i = self.wrap_tensor(self.f, dim=0, lazy=lazy, device=device)
        g_j = self.wrap_tensor(self.g, dim=1, lazy=lazy, device=device)
        C_ij = self.wrap_cost(C, lazy=lazy, device=device)
        alogs_i = self.wrap_tensor(self.a.log(), dim=0, lazy=lazy, device=device)
        blogs_j = self.wrap_tensor(self.b.log(), dim=1, lazy=lazy, device=device)

        f = - self.eps * (((g_j - C_ij) / self.eps) + blogs_j).logsumexp(dim=1).squeeze()
        g = - self.eps * (((f_i - C_ij) / self.eps) + alogs_i).logsumexp(dim=0).squeeze()

        return (self.a * f).sum() + (self.b * g).sum()

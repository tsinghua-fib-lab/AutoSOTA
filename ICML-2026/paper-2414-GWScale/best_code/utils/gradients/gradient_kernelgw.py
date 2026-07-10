"""
Differentiable Kernel Gromov-Wasserstein

Gradient computation for kernel-based GW formulation.
"""

import torch

from solvers.gromov_wasserstein.implementations.matrix_based.kernel import KernelGW
from utils.gradients.helpers import get_gradient_from_diffloss
from utils.implementation.eot import matching


def _kernel_from_cost(X1, X2, a1, a2, cost):
    """
    Convert cost function to centered kernel matrix.
    
    Args:
        X1: First point set, shape (N, D)
        X2: Second point set, shape (M, D)
        a1: Distribution over X1, shape (N,)
        a2: Distribution over X2, shape (M,)
        cost: Cost function
    
    Returns:
        torch.Tensor: Centered kernel matrix, shape (N, M)
    """
    C = cost(X1[:, None, :], X2[None, :, :])
    K = (C[:, 0].view(-1, 1).detach() + C[0, :].view(1, -1).detach() - C) / 2

    K_sum = (a1[:, None] * K).sum(dim=0)
    K_sum2 = (a2 * K_sum).sum()

    return K - K_sum[:, None] - K_sum[None, :] + K_sum2


def _kernelgw_costmatrix(X, Y, a, b, cost_x, cost_y, P):
    """
    Construct Sinkhorn cost matrix in kernel formulation.
    
    Given transport plan P, computes the corresponding Sinkhorn cost matrix using kernel representations of the spaces.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        cost_x: Cost function for source space
        cost_y: Cost function for target space
        P: Current transport plan, shape (N, M)
    
    Returns:
        torch.Tensor: Cost matrix, shape (N, M)
    """
    # Detach the dimensions corresponding to Gamma = \int X Y^T P
    Ky = _kernel_from_cost(X1=Y, X2=Y.detach(), a1=b, a2=b, cost=cost_y)
    KyP = torch.einsum("jl,kl->jk", Ky, P)
    Ky = None

    Kx = _kernel_from_cost(X1=X, X2=X.detach(), a1=a, a2=a, cost=cost_x)
    C = - 16 * torch.einsum("ik,jk->ij", Kx, KyP)
    Kx, KyP = None, None

    # Compute the diagonal of Kx, Ky and differentiate w.r.t. both dimensions
    Kx_diag = _kernel_from_cost(X1=X, X2=X, a1=a, a2=a, cost=cost_x).diag()
    Ky_diag = _kernel_from_cost(X1=Y, X2=Y, a1=b, a2=b, cost=cost_y).diag()

    C -= 4 * Kx_diag[:, None] * Ky_diag[None, :]
    return C


def _kernelgw_constant(X, Y, a, b, cost_x, cost_y):
    """
    Compute the constant part of quadratic GW loss (independent of transport plan).
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        a: Source distribution, shape (N,)
        b: Target distribution, shape (M,)
        cost_x: Cost function for source space
        cost_y: Cost function for target space
    
    Returns:
        torch.Tensor: Scalar constant term
    """
    Cxx = (a[None, :] * a[:, None] * (cost_x(X[:, None, :], X[None, :, :]) ** 2)).sum()
    Cyy = (b[None, :] * b[:, None] * (cost_y(Y[:, None, :], Y[None, :, :]) ** 2)).sum()

    Cx = (a * _kernel_from_cost(X1=X, X2=X, a1=a, a2=a, cost=cost_x).diag()).sum()
    Cy = (b * _kernel_from_cost(X1=Y, X2=Y, a1=b, a2=b, cost=cost_y).diag()).sum()

    return Cxx + Cyy - 4 * Cx * Cy


def diffloss_kernelgw(X, Y, costs, a=None, b=None, f_init=None, g_init=None, Z_init=None, solver_kwargs=None,
                      return_solution=False, cuda=True):
    """
    Compute differentiable kernel GW loss with automatic differentiation support.
    
    Returns a dual loss that is differentiable w.r.t. X and Y, so that auto-differentiation will return the EGW gradient
    of kernel Gromov-Wasserstein loss w.r.t. the input positions.
    
    It proceeds as follows:
    	1. Solver the kernel GW problem in a non-differentiable forward pass (within no_grad)
    	2. Activate auto-differentiation
    	3. Compute the GW constant term (independent of the transport plan)
    	4. Compute the Sinkhorn cost C_ij associated to the optimal GW solution
    	5. Compute the dual loss of Sinkhorn for the matrix C_ij
    	6. Compute the final loss equal to the sum of the GW constant (step 3) and the Sinkhorn dual loss (step 5)
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        costs: Cost function(s), single function or tuple (cost_x, cost_y)
        a: Source distribution. If None, defaults to uniform
        b: Target distribution. If None, defaults to uniform
        f_init: Initial source dual variable for warm start
        g_init: Initial target dual variable for warm start
        Z_init: Initial potential matrix for warm start
        solver_kwargs: Additional keyword arguments for KernelGW solver
        return_solution: If True, return solution components along with loss
        cuda: If True, use GPU memory for solving GW
    
    Returns:
        torch.Tensor: Differentiable scalar loss value
        tuple (optional): (f, g, Z, matching) if return_solution=True where f, g are dual variables, Z is potential, matching is hard assignment
    """
    cost_x, cost_y = costs if isinstance(costs, tuple) else costs, costs

    with torch.no_grad():
        Cx = cost_x(X[:, None, :], X[None, :, :])
        Cy = cost_y(Y[:, None, :], Y[None, :, :])
        if solver_kwargs is None:
            solver = KernelGW(Cx=Cx, Cy=Cy, a=a, b=b, Z=Z_init)
        else:
            solver = KernelGW(Cx=Cx, Cy=Cy, a=a, b=b, Z=Z_init, **solver_kwargs)

        if f_init is not None:
            solver.sinkhorn_solver.f = f_init
        if g_init is not None:
            solver.sinkhorn_solver.g = g_init

        if cuda:
            solver.to('cuda')

        solver.solve()
        if cuda:
            solver.to('cpu')

        a, b, eps = solver.a, solver.b, solver.eps
        P = solver.transport_plan()

    Cste = _kernelgw_constant(X=X, Y=Y, a=a, b=b, cost_x=cost_x, cost_y=cost_y)
    C_ij = _kernelgw_costmatrix(X=X, Y=Y, a=a, b=b, P=P, cost_x=cost_x, cost_y=cost_y)

    loss = Cste + solver.sinkhorn_solver.dual_loss(C=C_ij)

    if return_solution:
        return loss, solver.sinkhorn_solver.f, solver.sinkhorn_solver.g, solver.Z, matching(P)
    else:
        return loss


def gradient_kernelgw(X, Y, costs, which='both', a=None, b=None, f_init=None, g_init=None, Z_init=None,
                      solver_kwargs=None, return_solution=False, cuda=True):
    """
    Compute gradients of kernel GW distance with respect to input point clouds.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        costs: Cost function(s), single function or tuple (cost_x, cost_y)
        which: Controls which gradients to compute ('x', 'y', or 'both')
        a: Source distribution
        b: Target distribution
        f_init: Initial source dual for warm start
        g_init: Initial target dual for warm start
        Z_init: Initial potential for warm start
        solver_kwargs: Additional keyword arguments for KernelGW solver
        return_solution: If True, return solution components in addition to gradients
        cuda: If True, use GPU memory for solving GW

    Returns:
        torch.Tensor or tuple: Gradients w.r.t X, Y, or both
        tuple (optional): (gradients, (f, g, Z, matching)) if return_solution=True
    
    """
    if which in ['x', 'both']:
        X = X.clone().requires_grad_(True)
    if which in ['y', 'both']:
        Y = Y.clone().requires_grad_(True)

    res = diffloss_kernelgw(X, Y, costs, a, b, f_init, g_init, Z_init, solver_kwargs, return_solution, cuda)
    loss = res[0] if return_solution else res

    grads = get_gradient_from_diffloss(X, Y, loss, which=which)

    return (grads, res[1:]) if return_solution else grads

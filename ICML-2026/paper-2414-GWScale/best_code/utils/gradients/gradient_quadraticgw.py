"""
Differentiable Quadratic Gromov-Wasserstein

Implements gradient computation for squared Euclidean Gromov-Wasserstein distance.
"""

import torch
from pykeops.torch import LazyTensor

from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
from utils.gradients.helpers import get_gradient_from_diffloss
from utils.implementation.eot import matching
from utils.math.functions import center, squared_norm


def _quadraticgw_constant(X, Y, a, b):
    """
    Compute the constant part of quadratic GW loss (independent of transport plan).
        
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        a: Source distribution weights, shape (N,)
        b: Target distribution weights, shape (M,)
    
    Returns:
        torch.Tensor: Scalar constant term
    """
    X_centered, Y_centered = center(X, a), center(Y, b)
    X_sqnorm, Y_sqnorm = squared_norm(X), squared_norm(Y)

    X_i, X_j = LazyTensor(X_centered[:, None, :]), LazyTensor(X_centered[None, :, :])
    Y_i, Y_j = LazyTensor(Y_centered[:, None, :]), LazyTensor(Y_centered[None, :, :])

    a_i, a_j = LazyTensor(a[:, None, None]), LazyTensor(a[None, :, None])
    b_i, b_j = LazyTensor(b[:, None, None]), LazyTensor(b[None, :, None])

    Cxx = (a_i * a_j * (((X_i - X_j) ** 2).sum(dim=-1) ** 2)).sum(dim=1).sum()
    Cxy = (a[:, None] * X_sqnorm).sum() * (b[:, None] * Y_sqnorm).sum()

    Cyy = (b_i * b_j * (((Y_i - Y_j) ** 2).sum(dim=-1) ** 2)).sum(dim=1).sum()
    return Cxx + Cyy - 4 * Cxy


def _quadraticgw_costmatrix(a, X, b, Y, Z):
    """
    Build the cost matrix for the Sinkhorn subproblem given potential Z.
        
    Args:
        a: Source distribution, shape (N,)
        X: Source points, shape (N, D_x)
        b: Target distribution, shape (M,)
        Y: Target points, shape (M, D_y)
        Z: Potential matrix from GW optimization, shape (D_x, D_y)
    
    Returns:
        LazyTensor: Cost matrix C, shape (N, M)
    """
    X_centered, Y_centered = center(X, a), center(Y, b)
    X_sqnorm, Y_sqnorm = squared_norm(X_centered), squared_norm(Y_centered)

    XZ = torch.einsum("ik,kl->il", X_centered, Z)
    XZ_i = LazyTensor(XZ[:, None, :])
    Y_j = LazyTensor(Y_centered[None, :, :])

    X_sqnorm_i, Y_sqnorm_j = LazyTensor(X_sqnorm[:, None, :]), LazyTensor(Y_sqnorm[None, :, :])  # (1, M, 1)

    return - 16 * (XZ_i * Y_j).sum(dim=2) - 4 * (X_sqnorm_i * Y_sqnorm_j).sum(dim=2)  # (N, M)


def diffloss_quadraticgw(X, Y, a=None, b=None, f_init=None, g_init=None, Z_init=None, solver_kwargs=None,
                         return_solution=False):
    """
    Compute differentiable quadratic GW loss with automatic differentiation support.
    
    Returns a dual loss that is differentiable w.r.t. X and Y, so that auto-differentiation will return the EGW gradient of squared euclidean Gromov-Wasserstein loss w.r.t. the input positions.
    
    It proceeds as follow:
    	1. Solver the quadratic GW problem in a non-differentiable forward pass (within no_grad)
    	2. Activate auto-differentiation
    	3. Compute the GW constant term (independent of the transport plan)
    	4. Compute the Sinkhorn cost C_ij associated to the optimal GW solution
    	5. Compute the dual loss of Sinkhorn for the matrix C_ij
    	6. Compute the final loss equal to the sum of the GW constant (step 3) and the Sinkhorn dual loss (step 5)
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        a: Source distribution. If None, defaults to uniform
        b: Target distribution. If None, defaults to uniform
        f_init: Initial source dual variable for warm start
        g_init: Initial target dual variable for warm start
        Z_init: Initial potential matrix for warm start
        solver_kwargs: Additional keyword arguments for QuadraticGW solver
        return_solution: If True, return solution components along with loss
    
    Returns:
        torch.Tensor: Differentiable scalar loss value
        tuple (optional): (f, g, Z, matching) if return_solution=True where f, g are dual variables, Z is potential, matching is hard assignment
    """
    with (torch.no_grad()):
        if solver_kwargs is None:
            solver = QuadraticGW(X=X, Y=Y, a=a, b=b, Z=Z_init)
        else:
            solver = QuadraticGW(X=X, Y=Y, a=a, b=b, Z=Z_init, **solver_kwargs)

        if f_init is not None:
            solver.sinkhorn_solver.f = f_init
        if g_init is not None:
            solver.sinkhorn_solver.g = g_init
        solver.solve()

        a, b, eps = solver.a, solver.b, solver.eps
        Z = solver.Z

    Cste = _quadraticgw_constant(X=X, Y=Y, a=a, b=b)
    C_ij = _quadraticgw_costmatrix(X=X, Y=Y, a=a, b=b, Z=Z)

    loss = Cste + solver.sinkhorn_solver.dual_loss(C=C_ij)

    if return_solution:
        return loss, solver.sinkhorn_solver.f, solver.sinkhorn_solver.g, solver.Z, matching(solver.transport_plan())
    else:
        return loss


def gradient_quadraticgw(X, Y, which='both', a=None, b=None, f_init=None, g_init=None, Z_init=None, solver_kwargs=None,
                         return_solution=False):
    """
    Compute gradients of quadratic GW distance with respect to input point clouds.
    
    Args:
        X: Source points, shape (N, D_x)
        Y: Target points, shape (M, D_y)
        which: Controls which gradients to compute ('x', 'y', or 'both')
        a: Source distribution
        b: Target distribution
        f_init: Initial source dual for warm start
        g_init: Initial target dual for warm start
        Z_init: Initial potential for warm start
        solver_kwargs: Additional keyword arguments for QuadraticGW solver
        return_solution: If True, return solution components in addition to gradients
    
    Returns:
        torch.Tensor or tuple: Gradients w.r.t X, Y, or both
        tuple (optional): (gradients, (f, g, Z, matching)) if return_solution=True
    
    """
    if which in ['x', 'both']:
        X = X.clone().requires_grad_(True)
    if which in ['y', 'both']:
        Y = Y.clone().requires_grad_(True)

    res = diffloss_quadraticgw(X, Y, a, b, f_init, g_init, Z_init, solver_kwargs, return_solution)
    loss = res[0] if return_solution else res

    grads = get_gradient_from_diffloss(X, Y, loss, which=which)

    return (grads, res[1:]) if return_solution else grads

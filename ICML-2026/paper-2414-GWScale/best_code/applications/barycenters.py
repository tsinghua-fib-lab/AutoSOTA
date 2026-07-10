"""
Gromov-Wasserstein Barycenters

Compute barycenters in the Gromov-Wasserstein space using gradient-based optimization.
"""

import time
import torch

from utils.gradients.gradient_cntgw import gradient_cntgw
from utils.gradients.helpers import gradient_step
from utils.viz.data import plot_pointcloud


def egw_barycenter(X, Y, costs, weight=0.5, a=None, b=None, Bar_init=None, approx_dims=10, iters=100, lbda=1,
                   momentum=None, normalize_gradient=False, clip=None, solver_kwargs=None):
    """
    Compute Entropic Gromov-Wasserstein barycenter between two point clouds.

    The objective is:
        min_Bar  (1-w) * GW(Bar,X) + w * GW(Bar,Y) - GW(Bar,Bar)

    Args:
        X: First input point cloud, shape (N, D)
        Y: Second input point cloud, shape (M, D)
        costs: Cost function(s) for computing distances, single or tuple (cost_x, cost_y)
        weight: Interpolation weight in [0, 1]. 0=closer to X, 1=closer to Y, 0.5=midpoint
        a: Distribution over X vertices. If None, uses uniform
        b: Distribution over Y vertices. If None, uses uniform
        Bar_init: Initial barycenter, shape (K, D).
                  If None, initializes as copy of X
        approx_dims: Dimensionality for kernel PCA reduction (int or tuple)
        iters: Number of gradient descent iterations
        lbda: Learning rate (step size) for gradient descent
        momentum: Momentum coefficient in [0, 1). If None, uses vanilla gradient descent
        normalize_gradient: If True, normalize gradient by mean norm for stability
        clip: Maximum allowed gradient norm per point. If None, no clipping applied.
              Gradients with norm > clip are scaled down to clip
        solver_kwargs: Additional arguments passed to GW solver

    Returns:
        torch.Tensor: Barycenter point cloud, shape (K, D)

    """
    start_time = time.time()

    Bar = Bar_init.clone() if Bar_init is not None else X.clone()
    fx, gx, fy, gy, f_auto, g_auto, matchx, matchy = None, None, None, None, None, None, None, None
    match_auto = torch.arange(Bar.shape[0], dtype=torch.int64, device=Bar.device)

    old_Bar = Bar.clone()
    for it in range(iters):
        grad_x, (fx, gx, _, matchx) = gradient_cntgw(Bar, X, costs, b=a, which='x', approx_dims=approx_dims,
                                                     f_init=fx, g_init=gx, match_init=matchx,
                                                     solver_kwargs=solver_kwargs, return_solution=True)

        grad_y, (fy, gy, _, matchy) = gradient_cntgw(Bar, Y, costs, b=b, which='x', approx_dims=approx_dims,
                                                     f_init=fy, g_init=gy, match_init=matchy,
                                                     solver_kwargs=solver_kwargs, return_solution=True)

        grad_auto, (f_auto, g_auto, _, _) = gradient_cntgw(Bar, Bar, costs, which='x', approx_dims=approx_dims,
                                                           f_init=f_auto, g_init=g_auto, match_init=match_auto,
                                                           solver_kwargs=solver_kwargs, return_solution=True)

        grad = (1 - weight) * grad_x + weight * grad_y - grad_auto

        Bar, old_Bar = gradient_step(Bar, grad, old_X=old_Bar, lbda=lbda, momentum=momentum,
                                     normalize_gradient=normalize_gradient, clip=clip)

        if ((it + 1) % 10) == 0:
            print(f"Step {it + 1} / {iters} ({time.time() - start_time:.2g} s)")

    return Bar

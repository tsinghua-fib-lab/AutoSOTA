"""
Gromov-Wasserstein Gradient Flow

Implements gradient flow for continuously deforming one point cloud into another along the Gromov-Wasserstein gradient.
"""

import time
import torch

from utils.gradients.gradient_cntgw import gradient_cntgw
from utils.gradients.helpers import gradient_step
from utils.viz.data import plot_pointcloud


def gradient_flow(X, Y, costs, a=None, b=None, match_init=None, approx_dims=10, iters=100, lbda=1,
                  tmax=None, momentum=None, normalize_gradient=False, clip=None, record_step=1, solver_kwargs=None,
                  plot=False):
    """
    Compute Gromov-Wasserstein gradient flow from source X toward target Y.

    A gradient flow deforms the source point cloud X along the GW gradient to progressively match the target Y's
    geometry.

    The flow follows:
        dX/dt = -∇_X [GW(X,Y) - GW(X,X)]

    Args:
        X: Source point cloud to deform, shape (N, D)
        Y: Target point cloud (fixed), shape (M, D)
        costs: Cost function(s) for distance computation, single or tuple
        weight: Unused parameter (kept for interface compatibility)
        a: Distribution over X vertices. If None, uses uniform
        b: Distribution over Y vertices. If None, uses uniform
        match_init: Initial matching for warm start. If None, computed fresh
        approx_dims: Dimensionality for kernel PCA reduction (int or tuple)
        iters: Number of gradient flow steps. If None, computed as tmax/lbda
        lbda: Step size (learning rate) for gradient descent
        tmax: Total flow time. If provided, iters = tmax/lbda
        momentum: Momentum coefficient for acceleration. If None, no momentum.
                 Typical values: 0.5-0.9
        normalize_gradient: If True, normalize gradient by mean norm for stability
        clip: Maximum allowed gradient norm per point. If None, no clipping applied.
              Gradients with norm > clip are scaled down to clip
        record_step: Save point cloud every record_step iterations. 1=every step
        solver_kwargs: Additional arguments for GW solver (e.g., eps, numItermax)
        plot: If True, plot point cloud at each recorded step (for debugging)

    Returns:
        list[torch.Tensor]: Trajectory of point clouds [X(0), X(t₁), ..., X(T)] where each element has shape (N, D)
    """
    start_time = time.time()

    if iters is None:
        iters = int(tmax / lbda)

    X_t = X.clone()

    X_t_list = [X_t]

    f, g, f_auto, g_auto = None, None, None, None
    match = match_init
    match_auto = torch.arange(X.shape[0], dtype=torch.int64, device=X.device)

    X_t_old = X_t.clone()
    for it in range(iters):

        grad_y, (f, g, _, match) = gradient_cntgw(X_t, Y, costs, a=a, b=b, which='x', approx_dims=approx_dims,
                                                  f_init=f, g_init=g, match_init=match,
                                                  solver_kwargs=solver_kwargs, return_solution=True)

        grad_auto, (f_auto, g_auto, _, _) = gradient_cntgw(X_t, X_t, costs, which='x', approx_dims=approx_dims,
                                                           f_init=f_auto, g_init=g_auto, match_init=match_auto,
                                                           solver_kwargs=solver_kwargs, return_solution=True)

        grad = grad_y - grad_auto

        X_t, X_t_old = gradient_step(X_t, grad, old_X=X_t_old, lbda=lbda, momentum=momentum,
                                     normalize_gradient=normalize_gradient, clip=clip)

        if (it % record_step) == record_step - 1:
            X_t_list.append(X_t)
            if plot:
                plot_pointcloud(X_t)

        if ((it + 1) % 10) == 0:
            print(f"Step {it + 1} / {iters} ({time.time() - start_time:.2g} s)")

    return X_t_list

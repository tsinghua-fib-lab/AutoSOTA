"""
Gradient Computation Helpers

Utility functions for extracting gradients from differentiable losses.
"""

import torch


def get_gradient_from_diffloss(X, Y, loss, which='both'):
    """
    Extract gradients of a scalar loss with respect to point clouds.
    
    Computes gradients via automatic differentiation for one or both inputs.
    
    Args:
        X: Source point cloud, shape (N, D_x)
        Y: Target point cloud, shape (M, D_y)
        loss: Scalar differentiable loss (torch.Tensor with grad_fn)
        which: Specifies which gradients to compute:
               - 'x': Return gradient w.r.t X only
               - 'y': Return gradient w.r.t X only
               - 'both': Return both gradients
    
    Returns:
        torch.Tensor: Gradient of loss w.r.t X if which='x', shape (N, D_x)
        torch.Tensor: Gradient of loss w.r.t Y if which='y', shape (M, D_y)
        tuple: with both gradient if which='both'
    """
    gradX = torch.autograd.grad(loss, X)[0] if which in ['x', 'both'] else None
    gradY = torch.autograd.grad(loss, Y)[0] if which in ['y', 'both'] else None

    if which == 'x':
        output = gradX
    elif which == 'y':
        output = gradY
    else:
        output = (gradX, gradY)

    return output


def gradient_step(X, grad, old_X=None, lbda=1, momentum=None, normalize_gradient=False, clip=None):
    """
    Perform a single gradient descent step with optional momentum and gradient processing.

    This is a utility function for updating point cloud coordinates during gradient-based optimization (gradient flows,
    barycenter computation, etc.). It supports various stabilization techniques including normalization, clipping, and momentum.

    Args:
       X: Current point cloud positions, shape (N, D)
       grad: Gradient tensor, shape (N, D)
       old_X: Previous point cloud positions for momentum, shape (N, D).
              Required if momentum is not None
       lbda: Sstep size for gradient descent
       momentum: Momentum coefficient in [0, 1). If None, uses vanilla gradient descent
       normalize_gradient: If True, normalize gradient by mean norm for stability
       clip: Maximum allowed gradient norm per point. If None, no clipping applied.
             Gradients with norm > clip are scaled down to clip

    Returns:
       tuple: (X_new, cur_X) where:
           - X_new: Updated positions, shape (N, D)
           - cur_X: Copy of X before update (for next momentum step), or None if no momentum

    """
    cur_X = None

    grad = grad.clone()

    if normalize_gradient:
        grad = grad / grad.norm(dim=1).mean()

    if clip is not None:
        fltr = grad_norms > clip
        if fltr.sum() > 0:
            grad[fltr] *= (clip / grad[fltr].norm(dim=1))

    if momentum is None:
        X = X - lbda * grad
    else:
        cur_X = X.clone()
        X = X - lbda * grad + momentum * (X - old_X)

    return X, cur_X

# Implement computations of Wasserstein distances

# Libraries
import math
import torch
import ot as pot
from functools import partial
from ..utils.se3_utils import calculate_rmsd_matrix


def compute_wasserstein_distance(x0, x1, weights1=None, method='exact', reg=0.05, power=2, flattn_dims=True):
    """Compute the (regularized) Wasserstein distance

    If method == 'exact' the regularization is ignored.

    Args:
        * x0 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        * x1 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        * weights1 (torch.Tensor (batch_size, dim)): Weights of the second distribution (default is None)
        * method (str): OT solver (either 'exact' or 'sinkhorn') (default is 'exact')
        * reg (float): Regularization factor (default is 0.05)
        * power (int): Power for the distance (default is 2)

    Returns:
        * w2_ref (float): regularized Wasserstein distance
    """
    # Get the OT solver
    assert x0.ndim == x1.ndim
    if method == 'exact' or method is None:
        ot_fn = pot.emd2
    elif method == 'sinkhorn':
        ot_fn = partial(pot.sinkhorn2, reg=reg, method='sinkhorn_log')
    else:
        raise ValueError("Unknown method: " + method)
    # Get uniform weights
    a = torch.ones((x0.shape[0],), device=x0.device) / x0.shape[0]
    if weights1 is None:
        b = torch.ones((x1.shape[0],), device=x1.device) / x1.shape[0]
    else:
        b = weights1.clone().to(x1.device)
    # Reshape the data
    if x0.dim() > 2 and flattn_dims:
        x0 = x0.reshape((x0.shape[0], -1))
    if x1.dim() > 2 and flattn_dims:
        x1 = x1.reshape((x1.shape[0], -1))
    # Compute the pairwise distances
    if x0.ndim == 2:
        M = torch.cdist(x0.detach(), x1.detach(), p=power)
    elif x0.ndim == 3:
        M = calculate_rmsd_matrix(x0, x1).detach()
    # Compute the OT plan
    ret = ot_fn(a, b, M)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def interp(x, xp, fp):
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)
    return m[indicies] * x + b[indicies]


def wasserstein_distance_1d(u_values, v_values, u_weights=None, v_weights=None):
    """
    Compute the 1D Wasserstein distance between two 1D distributions using PyTorch.

    Parameters:
        - u_values: 1D tensor of samples from distribution u
        - v_values: 1D tensor of samples from distribution v
        - u_weights: optional 1D tensor of weights for u (same length as u_values)
        - v_weights: optional 1D tensor of weights for v (same length as v_values)

    Returns:
        - torch scalar: Wasserstein distance
    """
    u_values = u_values.flatten()
    v_values = v_values.flatten()

    if u_weights is None:
        u_weights = torch.ones_like(u_values) / len(u_values)
    else:
        u_weights = u_weights / u_weights.sum()

    if v_weights is None:
        v_weights = torch.ones_like(v_values) / len(v_values)
    else:
        v_weights = v_weights / v_weights.sum()

    # Sort the distributions
    u_sorted, u_indices = torch.sort(u_values)
    v_sorted, v_indices = torch.sort(v_values)

    u_weights_sorted = u_weights[u_indices]
    v_weights_sorted = v_weights[v_indices]

    # Compute cumulative sums (CDFs)
    u_cdf = torch.cumsum(u_weights_sorted, dim=0)
    v_cdf = torch.cumsum(v_weights_sorted, dim=0)

    # Interpolate CDFs over sorted union of values
    all_values = torch.cat([u_sorted, v_sorted])
    all_values, _ = torch.sort(all_values)

    u_interp = interp(all_values, u_sorted, u_cdf)
    v_interp = interp(all_values, v_sorted, v_cdf)

    dx = torch.diff(all_values, prepend=all_values[0:1])
    distance = torch.sum(torch.abs(u_interp - v_interp) * dx)

    return distance

import torch
import torch.nn as nn
from math import sqrt
import math
import random

from wildcat.math_utils import lambert_w_circ_exp
from wildcat.rp_nystrom import rp_nystrom

def compress_kv(
        keys,
        values,
        r,
        scale = None,
        q_scale = None
):
    """ Reduces key-value pairs of input sequence to a weighted coreset of size r.

    Only accepts tensor shapes of length three.
    Fold extra dimensions into batch dimension. 

    Args:
        keys (torch.Tensor): Key tensor of shape
            (batch_size, seq_len, qk_dim).
        values (torch.Tensor): Value tensor of shape
            (batch_size, seq_len, v_dim).
        r (int): Target coreset size.
        scale (float, optional): Scale factor for dot-product attention.
            Defaults to 1 / sqrt(qk_dim) if None.
        q_scale (float, optional): Upper bound on query norms. 
            Defaults to the maximum key norm if None.

    Returns:
        cmpd_keys (torch.Tensor): Selected coreset keys of shape
            (batch_size, r, qk_dim).
        cmpd_values (torch.Tensor): Nyström-weighted aggregation of values,
            shape (batch_size, r, v_dim).
        w (torch.Tensor): Per-coreset-key sum of Nyström weights, shape
            (batch_size, r).
    """

    E = keys.shape[-1]
    N = keys.shape[-2]

    if N <= r:
        w = torch.ones(*keys.shape[:2], device = keys.device, dtype = keys.dtype)
        return keys, values, w

    scale = scale or 1 / sqrt(E)

    sqd_knorm = keys.square().sum(dim=-1)

    k_scale = sqd_knorm.sqrt().amax(dim = -1, keepdim=True)
    if q_scale is None:
        q_scale = k_scale
    else:
        B = q_scale.shape[0]
        C = k_scale.shape[0]//B
        q_scale = q_scale.reshape(B, 1).expand(B, C).reshape(B*C, 1)

    # Shape (B, 1)
    tau = find_kernel_temperature(
        scale = scale,
        q_scale=q_scale,
        k_scale=k_scale,
        N = N,
        phi = None
    )

    # Rescale keys before compression
    key_multiplier = sqrt(scale) / tau
    keys = keys * key_multiplier.unsqueeze(-1)
    sqd_knorm = sqd_knorm * (key_multiplier**2)

    # Compression of keys and values
    # Outputs kernel_inv and kernel_core computed from Gaussian kernel
    coreset, kernel_core_inv, kernel_rows = rp_nystrom(
        keys=keys,
        sqd_knorm=sqd_knorm,
        r=r
    )

    # Select compressed keys:
    # Shape (B, r, E)
    cmpd_keys = keys.gather(-2, coreset.unsqueeze(-1).expand(*coreset.shape, E))
    cmpd_sqd_knorms = sqd_knorm.gather(-1, coreset)

    # Undo rescaling of keys
    # Undoing of rescaling only has to be applied to core_keys, not the norms, since the weights are computed for rescaled keys.
    cmpd_keys /= key_multiplier.unsqueeze(-1)

    # Compute Nystrom weights for Gaussian kernel
    W = torch.einsum("...rs, ...sl -> ...rl", kernel_core_inv, kernel_rows)

    # Adjust to weights for the exponential kernel
    scaling = -cmpd_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2)
    W = W * torch.exp(scaling / 2.)

    # Compute compressed values and softmax normalisation weight
    cmpd_values = torch.einsum("...rn, ...nd -> ...rd", W, values)
    w = W.sum(dim=-1)

    return cmpd_keys, cmpd_values, w
    

def compress(
        module: torch.nn.Module | None,
        hidden_states: torch.Tensor | None,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        r: int | float,
        kwargs: dict
):

    scale = module.scaling

    return compress_kv(keys, values, r, scale)



# Two times the constant rho_0 = sqrt(1+exp(2W_0(2/e^2)+2))
# up to machine precision
TWO_RHO_0 = 6.383202050647408
def find_kernel_temperature(
        scale,
        q_scale,
        k_scale,
        N: int,
        phi: float | None = None,
):
    """Finds the relative scale between keys and queries that optimises the trade-off
    between low-rank approximability of the attention kernel incurred error factors.

    Args:   q_scale (torch.Tensor): shape (batch_dims, 1): Maximum query norm: max_i ||q_i||_2
            k_scale (torch.Tensor): shape (batch_dims, 1): Maximum key norm: max_i ||k_i||_2
            N (int): number of key-value pairs
            phi (float): adjustable hyperparameter, default 1.0

    Returns: tau (torch.Tensor): shape (batch_dims, 1)
    """

    if phi is not None:
        N = N*phi**2

    b = math.log(N)/(scale*q_scale*k_scale) + 2.
    upper = b/(2*lambert_w_circ_exp((b/TWO_RHO_0).log()))
    tau = torch.sqrt(k_scale/q_scale * upper)

    return tau
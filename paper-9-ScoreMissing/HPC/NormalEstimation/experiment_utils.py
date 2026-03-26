import torch
from torch.nn import functional as F


def make_planar_funcs(norm_vec: torch.Tensor, intercept: float):
    """Creates planar g functions

    Args:
        norm_vec (torch.Tensor): Vector perp to hyperplane
        intercept (float): intercept.

    Returns:
        (planar_g, planar_g_mask, planar_boundary)
    """
    def planar_g(x: torch.Tensor):
        return torch.clamp(x @ norm_vec + intercept, 0.0, 1.0)

    def planar_boundary(x: torch.Tensor):
        return planar_g(x) > 0

    def planar_g_mask(x: torch.tensor, mask: torch.tensor):
        "Returns g is every coordinate is 1 otherwise returns 0"
        all_obs = torch.min(mask, dim=-1)[0]
        return all_obs * planar_g(x) + (1 - all_obs) * torch.ones_like(x[..., 0])

    return (planar_g, planar_g_mask, planar_boundary)


def my_g_weights(x, max_val=1):
    return F.hardtanh(x, min_val=0, max_val=max_val)/max_val

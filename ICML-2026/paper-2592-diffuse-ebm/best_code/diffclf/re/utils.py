# Utils for replica exchange

# Libraries
import torch
from ..sde.diffusion import VE, VP

def make_re_pairings(num_noise_levels, device=None):
    """Make the pairings for replica exchange

    Args:
        * num_noise_levels (int): Number of noise levels
        * device (torch.device): Torch device to use (default is None)

    Returns:
        * pairings (list of torch.Tensor): List of two index arrays provided the noise levels
            to jump to in the even and in the odd case
    """

    arr = torch.arange(num_noise_levels, device=device)
    # Even pass
    mask_a = (arr % 2 == 0) & (arr + 1 < num_noise_levels)
    a = torch.stack([arr[mask_a], arr[mask_a] + 1], dim=-1)
    # Odd pass
    mask_b = (arr % 2 == 1) & (arr + 1 < num_noise_levels)
    b = torch.stack([arr[mask_b], arr[mask_b] + 1], dim=-1)
    # Return everything
    return [a, b]
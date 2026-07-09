# Denoising Score Matching (DSM) loss

# Libraries
import torch
from ..utils.se3_utils import remove_mean

def compute_loss(ts, data, net, sde, antithetic=True, is_particles=False):
    """Compute the Denoising Score Matching (DSM) loss

    This corresponds to the orignal loss from ArXiv:2011.13456 with sigma^2_t weighting.

    NOTE: Here the squared norm is divided by the dimension.

    Args:
            * ts (torch.Tensor of shape (batch_size, *data_shape_ones)): Evaluation times
            * data (torch.Tensor of shape (batch_size, *data_shape)): Evaluation points
            * net (function): Network modeling the score
            * sde (LinearSDE): SDE
            * antithetic (bool): Whether to apply the antithetic trick (default is True)
            * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
            * loss (torch.Tensor of shape (batch_size,)): Value of the loss
    """

    # Get the shapes
    data_shape = data.shape[1:]
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Noise the data
    loc, var = sde.noise_sample_params(ts, data)
    zs = torch.randn_like(data)
    if is_particles:
        zs = remove_mean(zs)
    ys = loc + torch.sqrt(var) * zs
    # Compute the MSE
    loss = torch.mean(torch.square(torch.sqrt(var) * net(ts, ys) + zs), dim=sum_indexes)
    # Apply the antithetic trick
    if antithetic:
        ys_antithetic = loc - torch.sqrt(var) * zs
        loss += torch.mean(torch.square(torch.sqrt(var) * net(ts, ys_antithetic) - zs), dim=sum_indexes)
        loss /= 2.0
    return loss

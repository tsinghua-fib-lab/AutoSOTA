# EDM-preconditioned denoising score matching loss

# Libraries
import torch
from ..utils.se3_utils import remove_mean

def compute_loss(ts, data, net, sde, data_var_scalar, antithetic=True, is_particles=False):
    """Compute the Denoising Score Matching (DSM) from Karras et al. 2022 (EDM)

    WARNING: This learns the denoiser directly

    NOTE: Here the squared norm is divided by the dimension.

    Args:
            * ts (torch.Tensor of shape (batch_size, *data_shape_ones)): Evaluation times
            * data (torch.Tensor of shape (batch_size, *data_shape)): Evaluation points
            * net (function): Network modeling the denoiser
            * sde (LinearSDE): SDE
            * data_var_scalar (float): Scalar variance of the data
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
    sigma_sq_ = sde.sigma_sq(ts)
    zs = torch.randn_like(data)
    if is_particles:
        zs = remove_mean(zs)
    ys = loc + torch.sqrt(var) * zs
    # Compute the weights
    weights = ((sigma_sq_ + data_var_scalar) / (sigma_sq_ * data_var_scalar)).flatten()
    # Compute the MSE
    loss = torch.mean(torch.square(net(ts, ys) - data), dim=sum_indexes)
    # Apply the antithetic trick
    if antithetic:
        ys_antithetic = loc - torch.sqrt(var) * zs
        loss += torch.mean(torch.square(net(ts, ys_antithetic) - data), dim=sum_indexes)
        loss /= 2.0
    return weights * loss

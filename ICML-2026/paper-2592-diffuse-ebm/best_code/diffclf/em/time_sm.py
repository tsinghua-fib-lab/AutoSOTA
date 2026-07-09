# Denoising Time-score matching loss

# Libraries
import torch
import math
from ..utils.se3_utils import remove_mean

def compute_loss_time_dsm(ebm, x0, time_sampler, sde, is_particles=False):
    """Compute the time score matching loss

    See (Guth et al. 2025) [2506.05310] and [Yu et al. 2025] [2502.02300]

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    ts = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Noise the data
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    zt = torch.randn_like(x0)
    if is_particles:
        zt = remove_mean(zt)
    xt = alpha_t * x0 + torch.sqrt(gamma_sq_t) * zt
    pred_log_prob_dot = ebm.log_prob_dot(ts, xt)
    cond_log_prob_dot = sde.s_dot_over_gamma(ts).flatten() * (x0 * zt).sum(dim=sum_indexes)
    cond_log_prob_dot += sde.gamma_dot_over_gamma(ts).flatten() * (torch.square(zt).sum(dim=sum_indexes) - dim)
    # Compute the MSE
    time_sm_loss = torch.square(cond_log_prob_dot - pred_log_prob_dot.flatten())
    time_sm_weights = 1. / torch.square(sde.gamma_dot_over_gamma(ts)).flatten()
    return time_sm_loss * time_sm_weights


def compute_loss_both(ebm, x0, time_sampler, sde, data_var_scalar, is_particles=False):
    """Compute the both the denoiser matching loss and the time score matching losses

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * edm_loss (torch.Tensor of shape (batch_size,)): Loss
        * tsm_loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Build the times
    ts = time_sampler.sample((x0.shape[0],)).view((-1, *data_shape_ones))
    # Noise the data
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    sigma_sq_t = sde.sigma_sq(ts)
    zt = torch.randn_like(x0)
    if is_particles:
        zt = remove_mean(zt)
    xt = alpha_t * x0 + torch.sqrt(gamma_sq_t) * zt
    _, denoiser, pred_log_prob_dot = ebm.log_prob_and_grad_and_dot(ts, xt, return_denoiser=True)
    if is_particles:
        denoiser = remove_mean(denoiser)
    cond_log_prob_dot = -(sde.s_dot(ts) / sde.gamma(ts)).flatten() * (x0 * zt).sum(dim=sum_indexes)
    cond_log_prob_dot += sde.gamma_dot(ts).flatten() / sde.gamma(ts).flatten() * (torch.square(zt).sum(dim=sum_indexes) - dim)
    # Compute the MSE
    time_sm_loss = torch.square(cond_log_prob_dot - pred_log_prob_dot.flatten()) / dim
    edm_weights = ((sigma_sq_t + data_var_scalar) / (sigma_sq_t * data_var_scalar)).flatten()
    edm_loss = torch.sum(torch.square(denoiser - x0), dim=sum_indexes) / dim
    # time_sm_weights = 1. / torch.square(sde.gamma_dot_over_gamma(ts)).flatten()
    time_sm_weights = gamma_sq_t.flatten()
    return edm_loss * edm_weights, time_sm_loss * time_sm_weights

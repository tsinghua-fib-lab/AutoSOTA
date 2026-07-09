# Radon-Nikodym Estimator (RNE) energy regularizer

# Libraries
import torch
import math
from ..utils.se3_utils import remove_mean

def compute_loss_edm(ebm, x0, time_sampler, sde, data_var_scalar, is_particles=False):
    """Compute the denoiser matching loss (see Karras et al. 2022)

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
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
    # Compute the loss
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    gamma_t = torch.sqrt(gamma_sq_t)
    sigma_sq_t = sde.sigma_sq(ts)
    zt = torch.randn_like(x0)
    if is_particles:
         zt = remove_mean(zt)
    xt = alpha_t * x0 + gamma_t * zt
    # Compute the loss
    weights = ((sigma_sq_t + data_var_scalar) / (sigma_sq_t * data_var_scalar)).flatten()
    x0_hat = ebm.denoiser(ts, xt)
    if is_particles:
        x0_hat = remove_mean(x0_hat)
    loss = torch.sum(torch.square(x0_hat - x0), dim=sum_indexes) / dim
    return weights * loss


def log_prob_gaussian(y, mean, variance, sum_indexes=-1, is_normalized=False):
    """Compute log N(y; mean, variance) for a (possibly broadcasted) diagonal Gaussian.

    Notes:
    - `sum_indexes` should specify the event dimensions to sum over (typically all non-batch dims).
    - `variance` is allowed to be broadcastable to `y` (e.g. shape (B,1,1,...) for isotropic noise).
    """
    assert mean.ndim == variance.ndim and mean.shape[0] == variance.shape[0], (
        "Mean and variance must have the same number of dimensions and the first dimension must be the batch size."
    )
    var = variance + 1e-6
    log_prob = -0.5 * torch.sum(torch.square(y - mean) / var, dim=sum_indexes)
    if is_normalized:
        # Number of scalar event components being reduced.
        if isinstance(sum_indexes, int):
            event_size = y.shape[sum_indexes]
        else:
            event_size = 1
            for d in sum_indexes:
                event_size *= y.shape[d]
        log_prob -= 0.5 * float(event_size) * math.log(2.0 * math.pi)
        # Broadcast variance to y for the log-det term so isotropic variances contribute `event_size * log(var)`.
        log_prob -= 0.5 * torch.sum(torch.log(var).expand_as(y), dim=sum_indexes)
    return log_prob

def compute_loss_rne(ebm, x0, f, time_sampler, sde, step_size=1e-3, is_particles=False, type="consecutive"):
    """Compute the RNE loss

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * step_size (float): Step size for the RNE loss
        * is_particles (bool): Whether it is a particle system (default is False)
        * type (str): Type of time sampling (default is "consecutive")

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    dim = math.prod(data_shape)
    assert type == "consecutive", "Only consecutive time sampling is supported for RNE."
    s, s_ind = time_sampler.sample((batch_size, 1), return_idx=True, exclude_last_level=True)
    t_ind = (s_ind + 1).clamp(max=sde.n_levels-1)
    t = time_sampler.times[t_ind]
    st = torch.cat([s, t], dim=1)
    st_ind = torch.cat([s_ind, t_ind], dim=1)

    # Ensure s < t (and keep discrete indices aligned if present)
    swap = st[:, 0] > st[:, 1]
    s0, t0 = st[:, 0], st[:, 1]
    st = torch.stack([torch.where(swap, t0, s0), torch.where(swap, s0, t0)], dim=1)
    if st_ind is not None:
        s_ind0, t_ind0 = st_ind[:, 0], st_ind[:, 1]
        st_ind = torch.stack(
            [torch.where(swap, t_ind0, s_ind0), torch.where(swap, s_ind0, t_ind0)], dim=1
        )
    st = st.view((batch_size, 2, *data_shape_ones))
    s, t = st[:,0], st[:,1]
    dt = t - s
    if callable(f):
        f_s, f_t = f(s.flatten()), f(t.flatten())
    else:
        if st_ind is None:
            raise ValueError('Can\'t use continuous time with discrete f.')
        s_ind, t_ind = st_ind[:,0], st_ind[:,1]
        f_t, f_s = f[t_ind], f[s_ind]
    alpha_s, gamma_sq_s = sde.transition_params_from_data(s)
    xs = alpha_s * x0 + torch.sqrt(gamma_sq_s) * torch.randn_like(x0)
    noise_xs_to_xt = torch.randn_like(xs)
    if is_particles:
        xs = remove_mean(xs)
        noise_xs_to_xt = remove_mean(noise_xs_to_xt)
    alpha_st, gamma_sq_st = sde.transition_params(s, t)
    noising_mean = alpha_st * xs
    xt = noising_mean + torch.sqrt(gamma_sq_st) * noise_xs_to_xt
    log_prob_xs_to_xt = log_prob_gaussian(
        xt, noising_mean, gamma_sq_st, sum_indexes=sum_indexes, is_normalized=True
    )
    log_prob_xs_s = ebm.log_prob(s, xs)
    log_prob_xt_t, score_xt_t = ebm.log_prob_and_grad(t, xt)

    gt_sq = sde.g(t) ** 2
    denoising_mean = xt - dt * (sde.f(t) * xt - gt_sq * score_xt_t.clone().detach())
    denoising_var = gt_sq * dt
    log_prob_xt_to_xs = log_prob_gaussian(
        xs, denoising_mean, denoising_var, sum_indexes=sum_indexes, is_normalized=True
    ).detach()
    rne_loss = (log_prob_xt_t - f_t + log_prob_xs_to_xt - log_prob_xs_s + f_s - log_prob_xt_to_xs)
    return rne_loss


def compute_loss_both(ebm, x0, f, time_sampler, sde, data_var_scalar, is_particles=False, type="consecutive"):
    """Compute the RNE and EDM losses

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
        * is_particles (bool): Whether it is a particle system (default is False)
        * type (str): Type of time sampling (default is "consecutive")

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    dim = math.prod(data_shape)
    assert type == "consecutive", "Only consecutive time sampling is supported for RNE."
    s, s_ind = time_sampler.sample((batch_size, 1), return_idx=True, exclude_last_level=True)
    t_ind = (s_ind + 1).clamp(max=time_sampler.n_levels-1)
    t = time_sampler.times[t_ind]
    st = torch.cat([s, t], dim=1)
    st_ind = torch.cat([s_ind, t_ind], dim=1)

    # Ensure s < t (and keep discrete indices aligned if present)
    swap = st[:, 0] > st[:, 1]
    s0, t0 = st[:, 0], st[:, 1]
    st = torch.stack([torch.where(swap, t0, s0), torch.where(swap, s0, t0)], dim=1)
    if st_ind is not None:
        s_ind0, t_ind0 = st_ind[:, 0], st_ind[:, 1]
        st_ind = torch.stack(
            [torch.where(swap, t_ind0, s_ind0), torch.where(swap, s_ind0, t_ind0)], dim=1
        )
    st = st.view((batch_size, 2, *data_shape_ones))
    s, t = st[:,0], st[:,1]
    dt = t - s
    if callable(f):
        f_s, f_t = f(s.flatten()), f(t.flatten())
    else:
        if st_ind is None:
            raise ValueError('Can\'t use continuous time with discrete f.')
        s_ind, t_ind = st_ind[:,0], st_ind[:,1]
        f_t, f_s = f[t_ind], f[s_ind]
    alpha_s, gamma_sq_s = sde.transition_params_from_data(s)
    xs = alpha_s * x0 + torch.sqrt(gamma_sq_s) * torch.randn_like(x0)
    noise_xs_to_xt = torch.randn_like(xs)
    if is_particles:
        xs = remove_mean(xs)
        noise_xs_to_xt = remove_mean(noise_xs_to_xt)
    alpha_st, gamma_sq_st = sde.transition_params(s, t)
    noising_mean = alpha_st * xs
    xt = noising_mean + torch.sqrt(gamma_sq_st) * noise_xs_to_xt

    log_prob_xs_to_xt = log_prob_gaussian(
        xt, noising_mean, gamma_sq_st, sum_indexes=sum_indexes, is_normalized=True
    )
    log_prob_xs_s = ebm.log_prob(s, xs)

    alpha_t, gamma_sq_t = sde.transition_params_from_data(t)

    log_prob_xt_t, x0_hat = ebm.log_prob_and_grad(t, xt, return_denoiser=True)
    score_xt_t = (alpha_t * x0_hat - xt) / gamma_sq_t

    gt_sq = sde.g(t) ** 2
    denoising_mean = xt - dt * (sde.f(t) * xt - gt_sq * score_xt_t.clone().detach())
    denoising_var = gt_sq * dt
    log_prob_xt_to_xs = log_prob_gaussian(
        xs, denoising_mean, denoising_var, sum_indexes=sum_indexes, is_normalized=True
    ).detach()

    rne_loss = (log_prob_xt_t - f_t + log_prob_xs_to_xt - log_prob_xs_s + f_s - log_prob_xt_to_xs) ** 2

    sigma_sq_t = sde.sigma_sq(t)
    weights = ((sigma_sq_t + data_var_scalar) / (sigma_sq_t * data_var_scalar)).flatten()
    if is_particles:
        x0_hat = remove_mean(x0_hat)
    loss = torch.sum(torch.square(x0_hat - x0), dim=sum_indexes) / dim
    return weights * loss, rne_loss
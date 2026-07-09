# Bi-level energy classification loss (Bregman-divergence variant)

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

def compute_loss_bilevel_bregman(
    ebm, 
    x0, 
    f, 
    time_sampler, 
    sde,
    bregman_divergence_type="jensen_shannon", 
    is_particles=False, 
    type="uniform"
):
    """Compute the bi-level classification loss with random noise levels

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * phi_fn (callable): Bregman divergence function
        * phi_prime_fn (callable): Bregman divergence function derivative
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Sample the consecutive levels
    if type == "uniform":
        st, st_ind = time_sampler.sample((batch_size, 2), return_idx=True)
    elif type == "consecutive":
        if time_sampler.use_continuous_time:
            raise NotImplementedError('consecutive times not implemented with continuous time.')
        s, s_ind = time_sampler.sample((batch_size, 1), return_idx=True, exclude_last_level=True)
        t_ind = (s_ind + 1).clamp(max=sde.n_levels-1)
        t = time_sampler.times[t_ind]
        st = torch.cat([s, t], dim=1)
        st_ind = torch.cat([s_ind, t_ind], dim=1)
    else:
        raise NotImplementedError(f"Type {type} not implemented!")
    st = st.view((batch_size, 2, *data_shape_ones))
    s, t = st[:,0], st[:,1]
    if callable(f):
        f_s, f_t = f(s.flatten()), f(t.flatten())
    else:
        if st_ind is None:
            raise ValueError('Can\'t use continuous time with discrete f.')
        s_ind, t_ind = st_ind[:,0], st_ind[:,1]
        f_t, f_s = f[t_ind], f[s_ind]
    alpha_s, gamma_sq_s = sde.transition_params_from_data(s)
    xs = alpha_s * x0 + torch.sqrt(gamma_sq_s) * torch.randn_like(x0)
    if is_particles:
        xs = remove_mean(xs)
    alpha_t, gamma_sq_t = sde.transition_params_from_data(t)
    xt = alpha_t * x0 + torch.sqrt(gamma_sq_t) * torch.randn_like(x0)
    if is_particles:
        xt = remove_mean(xt)
    log_prob_xs_s = ebm.log_prob(s, xs)
    log_prob_xt_t = ebm.log_prob(t, xt)
    log_prob_xs_t = ebm.log_prob(t, xs)
    log_prob_xt_s = ebm.log_prob(s, xt)
    log_g_xs = log_prob_xs_t - f_t - neg_en_ii[:, 0] + f_s
    log_g_xt = neg_en_ii[:, 1] - f_t - log_prob_xt_s + f_s
    if bregman_divergence_type == "jensen_shannon":
        bregman_loss = JensenShannonBregmanLoss(log_g_xs, log_g_xt)
    elif bregman_divergence_type == "forward_kl":
        bregman_loss = ForwardKullbackLeiblerBregmanLoss(log_g_xs, log_g_xt)
    else:
        raise NotImplementedError(f"Bregman divergence type {bregman_divergence_type} not implemented!")
    return bregman_loss


def compute_loss_both(ebm, x0, f, time_sampler, sde, data_var_scalar, bregman_divergence_type="jensen_shannon", is_particles=False, type="uniform"):
    """Compute the bi-level classification loss with random noise levels together
       with the denoiser matching loss

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
        * phi_fn (callable): Bregman divergence function
        * phi_prime_fn (callable): Bregman divergence function derivative
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * dm_loss (torch.Tensor of shape (batch_size,)): Denoiser matching loss
        * clf_loss (torch.Tensor of shape (batch_size,)): Classification loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    data_shape_minus_ones = (-1,) * len(data_shape)
    dim = math.prod(data_shape)
    sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
    # Sample the consecutive levels
    if type == "uniform":
        st, st_ind = time_sampler.sample((batch_size, 2), return_idx=True)
    elif type == "consecutive":
        if time_sampler.use_continuous_time:
            raise NotImplementedError('consecutive times not implemented with continuous time.')
        s, s_ind = time_sampler.sample((batch_size, 1), return_idx=True, exclude_last_level=True)
        t_ind = (s_ind + 1).clamp(max=len(time_sampler.times)-1)
        t = time_sampler.times[t_ind]
        st = torch.cat([s, t], dim=1)
        st_ind = torch.cat([s_ind, t_ind], dim=1)
    else:
        raise NotImplementedError(f"Type {type} not implemented!")
    st = st.view((batch_size, 2, *data_shape_ones))
    s, t = st[:,0], st[:,1]
    if callable(f):
        f_s, f_t = f(s.flatten()), f(t.flatten())
    else:
        if st_ind is None:
            raise ValueError('Can\'t use continuous time with discrete f.')
        s_ind, t_ind = st_ind[:,0], st_ind[:,1]
        f_t, f_s = f[t_ind], f[s_ind]
    # Noise all those samples
    alpha, gamma_sq = sde.transition_params_from_data(st)
    x0_expanded = x0.unsqueeze(1).expand((-1, 2, *data_shape_minus_ones))
    xst = alpha * x0_expanded + torch.sqrt(gamma_sq) * torch.randn_like(x0_expanded)
    if is_particles:
        xst = remove_mean(xst)
    neg_en_ii, denoiser_ii = ebm.log_prob_and_grad(
        st.view((-1, *data_shape_ones)), xst.view((-1, *data_shape)),
        return_denoiser=True)
    neg_en_ii = neg_en_ii.view((-1, 2))
    denoiser_ii = denoiser_ii.view((x0.shape[0], 2, *data_shape))
    if is_particles:
        denoiser_ii = remove_mean(denoiser_ii)
    log_prob_xs_t = ebm.log_prob(t, xst[:, 0, :])
    log_prob_xt_s = ebm.log_prob(s, xst[:, 1, :])
    log_g_xs = log_prob_xs_t - f_t - neg_en_ii[:, 0] + f_s
    log_g_xt = neg_en_ii[:, 1] - f_t - log_prob_xt_s + f_s
    if bregman_divergence_type == "jensen_shannon":
        bregman_loss = JensenShannonBregmanLoss(log_g_xs, log_g_xt)
    elif bregman_divergence_type == "forward_kl":
        bregman_loss = ForwardKullbackLeiblerBregmanLoss(log_g_xs, log_g_xt)
    else:
        raise NotImplementedError(f"Bregman divergence type {bregman_divergence_type} not implemented!")
    # clf_loss = 0.5 * torch.nn.functional.softplus(log_prob_xs_t - f_t - neg_en_ii[:, 0] + f_s)
    # clf_loss += 0.5 * torch.nn.functional.softplus(log_prob_xt_s - f_s - neg_en_ii[:, 1] + f_t)
    sigma_sq_ts = sde.sigma_sq(st)
    weights = ((sigma_sq_ts + data_var_scalar) / (sigma_sq_ts * data_var_scalar)).flatten()
    dm_loss = weights * torch.sum(torch.square(denoiser_ii - x0_expanded), dim=sum_indexes).flatten() / dim
    return dm_loss, bregman_loss

def JensenShannonBregmanLoss(log_g_xs, log_g_xt):
    return torch.nn.functional.softplus(log_g_xs) + torch.nn.functional.softplus(-log_g_xt)

def ForwardKullbackLeiblerBregmanLoss(log_g_xs, log_g_xt):
    log_g_xs_clamped = torch.clamp(log_g_xs, max=20.0)
    return torch.exp(log_g_xs_clamped) - log_g_xt

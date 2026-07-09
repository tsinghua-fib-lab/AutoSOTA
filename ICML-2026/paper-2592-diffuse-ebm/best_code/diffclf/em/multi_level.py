# Multi-level energy classification loss

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

def compute_loss_multi_level(ebm, x0, f, times, sde, is_particles=False):
    """Compute the true multi-level classification loss

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * times (torch.Tensor of shape (n_levels,)): Time discretization
        * sde (sde): SDE
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Simulate the noising prcess
    n_levels = times.shape[0]
    xt = torch.empty((n_levels, *x0.shape), device=x0.device)
    xt[0] = x0
    for k in range(1, n_levels):
        alpha_s_t, gamma_sq_s_t = sde.transition_params(times[k-1], times[k])
        xt[k] = alpha_s_t * xt[k-1] + torch.sqrt(gamma_sq_s_t) * torch.randn_like(x0)
    if is_particles:
        xt = remove_mean(xt)
    # Expand times
    t_ones = torch.ones((batch_size, *data_shape_ones), device=x0.device)
    # Compute the energies
    # NOTE: The loop is intentionally kept for memory efficiency
    neg_en = torch.empty((n_levels, n_levels, batch_size), device=x0.device)
    with torch.no_grad():
        for i in range(n_levels):
            for j in range(n_levels):
                neg_en[i, j] = ebm.log_prob(t_ones * times[i], xt[j]) - f[i]
    log_sum_exp = torch.logsumexp(neg_en, dim=0)
    # Compute the loss
    arr = torch.arange(n_levels, device=x0.device)
    return -torch.mean(neg_en[arr, arr] - log_sum_exp, dim=0)

def compute_loss_random_multi_level(ebm, x0, f, k, i_idx, j_idx, time_sampler, sde, is_particles=False):
    """Compute the multi-level classification loss with random noise levels

    This is the multi-level loss with randomly picked noise levels.

    Note that, given k, the tensors i_idx and j_idx can be built as

        i_idx, j_idx = torch.meshgrid(
            torch.arange(k, device=device),
            torch.arange(k, device=device),
            indexing='ij'
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * k (int): Number of random noise levels
        * i_idx, j_idx (torch.Tensor of shape (k * k)): Utility tensors
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    data_shape_minus_ones = (-1,) * len(data_shape)
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Noise all those samples
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    xt = alpha_t * x0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    xt += torch.sqrt(gamma_sq_t) * torch.randn_like(xt)
    if is_particles:
        xt = remove_mean(xt)
    # Compute the energies
    ts_ij = ts[:, i_idx]
    xt_ij = xt[:, j_idx, :]
    if callable(f):
        f_i = f(ts_ij.flatten())
    else:
        f_i = f[idx[:, i_idx]]
    neg_en = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en = neg_en.view((-1, k, k)) - f_i.view((-1, k, k))
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    arr = torch.arange(k, device=x0.device)
    return -(neg_en[:, arr, arr] - neg_en_lse).mean(dim=-1)


def compute_loss_both(ebm, x0, f, k, i_idx, j_idx, diag_mask, time_sampler, sde, data_var_scalar, is_particles=False):
    """Compute the multi-level classification loss with random noise levels together
       with the denoiser matching loss

    Note that, given k, the tensors i_idx and j_idx can be built as

        i_idx, j_idx = torch.meshgrid(
            torch.arange(k, device=device),
            torch.arange(k, device=device),
            indexing='ij'
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)
        diag_mask = torch.eye(k, dtype=torch.bool, device=device).flatten()

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * k (int): Number of random noise levels
        * i_idx, j_idx, diag_mask (torch.Tensor of shape (k*k)): Utility tensors
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
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
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Noise all those samples
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    x0_expanded = x0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    xt = alpha_t * x0_expanded + torch.sqrt(gamma_sq_t) * torch.randn_like(x0_expanded)
    if is_particles:
        xt = remove_mean(xt)
    # Compute the energy and the denoiser on the diagonal
    neg_en_ii, denoiser_ii = ebm.log_prob_and_grad(
        ts.view((-1, *data_shape_ones)), xt.view((-1, *data_shape)), return_denoiser=True)
    neg_en_ii = neg_en_ii.view((batch_size, k))
    denoiser_ii = denoiser_ii.view((batch_size, k, *data_shape))
    if is_particles:
        denoiser_ii = remove_mean(denoiser_ii)
    # Compute the denoiser mathching loss
    sigma_sq_t = sde.sigma_sq(ts).view((*ts.shape, *data_shape_ones))
    weights = ((sigma_sq_t + data_var_scalar) / (sigma_sq_t * data_var_scalar)).flatten()
    dm_loss = weights * torch.sum(torch.square(denoiser_ii - x0_expanded), dim=sum_indexes).flatten() / dim
    # Remove the f from neg_en_ii
    if callable(f):
        neg_en_ii -= f(ts).view(neg_en_ii.shape)
    else:
        neg_en_ii -= f[idx]
    # Compute the remaining energies
    ts_ij = ts[:, i_idx[~diag_mask]]
    xt_ij = xt[:, j_idx[~diag_mask], :]
    if callable(f):
        f_i = f(ts_ij.flatten())
    else:
        f_i = f[idx[:, i_idx[~diag_mask]]]
    neg_en_no_diag = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en_no_diag = neg_en_no_diag.view((batch_size, k-1, k)) - f_i.view((batch_size, k-1, k))
    # Reconstruct the full matrix
    diag_mask = diag_mask.view((k, k))
    neg_en = torch.zeros((batch_size, k, k), device=x0.device)
    neg_en[:, diag_mask] = neg_en_ii
    neg_en[:, ~diag_mask] = neg_en_no_diag.view((batch_size, -1))
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    return dm_loss, -(neg_en_ii - neg_en_lse).mean(dim=-1)


def compute_loss_random_multi_level_with_target(ebm, x0, f, f0, target_log_prob,
                                                k, i_idx, j_idx, time_sampler, sde,
                                                is_particles=False):
    """Compute the multi-level classification loss with random noise levels and force the target at 0

    This is the multi-level loss with randomly picked noise levels plus the fully denoised level.

    Note that, given k, the tensors i_idx and j_idx can be built as

        i_idx, j_idx = torch.meshgrid(
            torch.arange(k, device=device),
            torch.arange(k, device=device),
            indexing='ij'
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * f0 (torch.Tensor of shape (1,)): Current estimate of the target's normalizing constant
        * target_log_prob (function): Log-likelihood of the target distribution
        * k (int): Number of random noise levels
        * i_idx, j_idx (torch.Tensor of shape (k * k)): Utility tensors
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * is_particles (bool): Whether it is a particle system (default is False)

    Returns:
        * loss (torch.Tensor of shape (batch_size,)): Loss
    """
    # Get the shapes
    batch_size = x0.shape[0]
    data_shape = x0.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    data_shape_minus_ones = (-1,) * len(data_shape)
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True, exclude_first_level=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Noise all those samples
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    x0_expanded = x0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    xt = alpha_t * x0_expanded + torch.sqrt(gamma_sq_t) * torch.randn_like(x0_expanded)
    if is_particles:
        xt = remove_mean(xt)
    # Compute the energies
    ts_ij = ts[:, i_idx]
    xt_ij = xt[:, j_idx, :]
    if callable(f):
        f_i = f(ts_ij.flatten())
    else:
        f_i = f[idx[:, i_idx]]
    neg_en = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en = neg_en.view((-1, k, k)) - f_i.view((-1, k, k))
    # Compute the target at every noising points
    xt_with_x0 = torch.cat((x0.unsqueeze(1), xt), dim=1)
    target_log_prob_xt = target_log_prob(xt_with_x0.view((-1, *data_shape))).view((batch_size, k+1)) - f0
    # Compute every EBM at x0
    ebm_at_x0 = ebm.log_prob(
        ts.view((-1, *data_shape_ones)), x0_expanded.reshape((-1, *data_shape))
    ).view((batch_size, k))
    # Concatenate the new energies
    neg_en = torch.cat((
        target_log_prob_xt.unsqueeze(1), torch.cat(
            (ebm_at_x0.unsqueeze(-1), neg_en),
            dim=-1)
    ), dim=1)
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    arr = torch.arange(k+1, device=x0.device)
    return -(neg_en[:, arr, arr] - neg_en_lse).mean(dim=-1)


def compute_both_with_target(ebm, x0, f, f0, target_log_prob, k, i_idx, j_idx,
                             diag_mask, diag_mask_padded, diag_mask_inv_padded,
                             time_sampler, sde, data_var_scalar, is_particles=False):
    """Compute the multi-level classification loss with random noise levels as well as the denoiser matching loss
       and force the target at 0

    This is the multi-level loss with randomly picked noise levels plus the fully denoised level.

    Note that, given k, the tensors i_idx and j_idx can be built as

        i_idx, j_idx = torch.meshgrid(
            torch.arange(k, device=device),
            torch.arange(k, device=device),
            indexing='ij'
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)
        diag_mask = torch.eye(k, dtype=torch.bool, device=device)
        diag_mask_padded = torch.nn.functional.pad(diag_mask, (1,0,1,0))
        diag_mask_inv_padded = torch.nn.functional.pad(~diag_mask, (1,0,1,0))
        diag_mask = diag_mask.flatten()

    Args:
        * ebm (EBM): Energy based model
        * x0 (torch.Tensor of shape (batch_size, *data_shape)): Data samples
        * f (torch.Tensor of shape (n_levels,) or callable): Current normalizing constants estimates
        * f0 (torch.Tensor of shape (1,)): Current estimate of the target's normalizing constant
        * target_log_prob (function): Log-likelihood of the target distribution
        * k (int): Number of random noise levels
        * i_idx, j_idx, diag_mask (torch.Tensor of shape (k * k)): Utility tensors
        * time_sampler (TimeSampler): TimeSampler
        * sde (sde): SDE
        * data_var_scalar (float): Scalar variance of the data
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
    # Sample the times
    ts, idx = time_sampler.sample((batch_size, k), return_idx=True, unique=True, exclude_first_level=True)
    ts = ts.view((batch_size, k, *data_shape_ones))
    # Noise all those samples
    alpha_t, gamma_sq_t = sde.transition_params_from_data(ts)
    x0_expanded = x0.unsqueeze(1).expand((-1, k, *data_shape_minus_ones))
    xt = alpha_t * x0_expanded + torch.sqrt(gamma_sq_t) * torch.randn_like(x0_expanded)
    if is_particles:
        xt = remove_mean(xt)
    # Compute the energy and the denoiser on the diagonal
    neg_en_ii, denoiser_ii = ebm.log_prob_and_grad(
        ts.view((-1, *data_shape_ones)), xt.view((-1, *data_shape)),
        return_denoiser=True
    )
    neg_en_ii = neg_en_ii.view((batch_size, k))
    denoiser_ii = denoiser_ii.view((batch_size, k, *data_shape))
    if is_particles:
        denoiser_ii = remove_mean(denoiser_ii)
    # Compute the denoiser mathching loss
    sigma_sq_t = sde.sigma_sq(ts).view((*ts.shape, *data_shape_ones))
    weights = ((sigma_sq_t + data_var_scalar) / (sigma_sq_t * data_var_scalar)).flatten()
    dm_loss = weights * torch.sum(torch.square(denoiser_ii - x0_expanded), dim=sum_indexes).flatten() / dim
    # Remove the f from neg_en_ii
    if callable(f):
        neg_en_ii -= f(ts).view(neg_en_ii.shape)
    else:
        neg_en_ii -= f[idx]
    # Compute the remaining energies
    ts_ij = ts[:, i_idx[~diag_mask]]
    xt_ij = xt[:, j_idx[~diag_mask], :]
    if callable(f):
        f_i = f(ts_ij.flatten())
    else:
        f_i = f[idx[:, i_idx[~diag_mask]]]
    neg_en_no_diag = ebm.log_prob(ts_ij.view((-1, *data_shape_ones)), xt_ij.view((-1, *data_shape)))
    neg_en_no_diag = neg_en_no_diag.view((batch_size, k-1, k)) - f_i.view((batch_size, k-1, k))
    # Compute the target at every noising points
    xt_with_x0 = torch.cat((x0.unsqueeze(1), xt), dim=1)
    target_log_prob_xt = target_log_prob(xt_with_x0.view((-1, *data_shape))).view((batch_size, k+1)) - f0
    # Compute every EBM at x0
    ebm_at_x0 = ebm.log_prob(
        ts.view((-1, *data_shape_ones)), x0_expanded.reshape((-1, *data_shape))
    ).view((batch_size, k))
    # Reconstruct the full matrix
    neg_en = torch.zeros((batch_size, k+1, k+1), device=x0.device)
    neg_en[:, 0] = target_log_prob_xt
    neg_en[:, 1:, 0] = ebm_at_x0
    neg_en[:, diag_mask_padded] = neg_en_ii
    neg_en[:, diag_mask_inv_padded] = neg_en_no_diag.view((batch_size, -1))
    neg_en_ii = torch.cat((target_log_prob_xt[:, 0, None], neg_en_ii), dim=-1)
    # Compute the loss
    neg_en_lse = torch.logsumexp(neg_en, dim=1)
    return dm_loss, -(neg_en_ii - neg_en_lse).mean(dim=-1)

# %%
import numpy as np
import torch
from torch.optim import SGD
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from torch.autograd import grad
from typing import Union, Optional
from warnings import warn

from ..models.density_models import UDensity  # noqa: E402


class NaNModel(Exception):
    pass


class Prox_SGD(SGD):  # noqa: D101
    """SGD with proximal operator for L1 regularisation on each parameter group.
    """
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        l1_reg: float = 0,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            l1_reg=l1_reg,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGD, self).__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def step(self, closure=None):
        super().step(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = p.data.sign() * torch.maximum(
                    torch.abs(p.data) - group["lr"] * group["l1_reg"], torch.zeros_like(p.data)
                )


def get_grad_norm(model: nn.Module, norm_type: Union[float, str] = 2.0):
    """Gets norm of gradients of model

    Args:
        model (nn.Module): Torch model to get norm of gradients from
        norm_type (float or str, optional): Type of norm to use. Defaults to 2.
    """
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    return torch.linalg.norm(torch.cat([torch.linalg.norm(g, norm_type) for g in grads]), norm_type)


def convert_stored_vals(stored_vals):
    """Converts a state dict to a tensor

    Args:
        stored_vals (dict): A dictionary with list of losses and state_dicts to convert

    Returns:
        torch.Tensor: Tensor of the state dictionary
    """
    state_dicts = stored_vals["State_dicts"]
    parameters = {key: [state_dict[key] for state_dict in state_dicts]
                  for key in state_dicts[0]}
    return {"Losses": stored_vals["Losses"]} | parameters


def convert_state_dicts(state_dicts):
    """Converts a state dict to a tensor

    Args:
        stored_vals (dict): A dictionary with list of losses and state_dicts to convert

    Returns:
        torch.Tensor: Tensor of the state dictionary
    """
    parameters = {key: [state_dict[key] for state_dict in state_dicts]
                  for key in state_dicts[0]}
    return parameters


def exact_score_loss(mean_0, mean_1, cov_0, cov_1, give_prec=False, normalise=True):
    """Calculates the exact score matching for proposal normal given true underlying normal distribution

    Args:
        mean_0 (torch.Tensor): True mean
        mean_1 (torch.Tensor): Proposed Mean
        cov_0 (torch.Tensor): True Covariance
        cov_1 (torch.Tensor): Proposed Covariance (or Precision if give_prec is True)
        give_prec (bool, optional): Whether to give precision instead of covariance. Defaults to False.

    Returns:
        torch.Tensor: Exact score matching loss
    """
    if give_prec:
        prec_1 = cov_1
    else:
        prec_1 = torch.linalg.inv(cov_1)
    trace_term = torch.sum(torch.diagonal(prec_1, dim1=-1, dim2=-2), dim=-1)
    mean_term = torch.sum(
        (prec_1 @ (mean_0 - mean_1).unsqueeze(-1)).squeeze(-1) ** 2, dim=-1
    )
    var_term = torch.sum(
        torch.diagonal(prec_1 @ cov_0 @ prec_1, dim1=-2, dim2=-1), dim=-1
    )
    # Without term constant w.r.t. target distribution
    exact_loss = -2 * trace_term + (mean_term + var_term)
    if normalise:
        exact_loss -= exact_score_loss(mean_0, mean_0, cov_0, cov_0, normalise=False)
    return exact_loss


def exact_score_loss_em(mean_0, mean_1, cov_0, cov_1, miss_inds, miss_weights, give_prec=False):
    """Calculates the exact score matching for proposal normal given true underlying normal distribution

    Args:
        mean_0 (torch.Tensor): True mean
        mean_1 (torch.Tensor): Proposed Mean
        cov_0 (torch.Tensor): True Covariance
        cov_1 (torch.Tensor): Proposed Covariance (or Precision if give_prec is True)
        miss_inds (list): A list of tensors containing the possible observed indices patterns
        miss_weights (list): A list of probabilities for each missing pattern in miss_inds.
        give_prec (bool, optional): Whether to give precision instead of covariance. Defaults to False.

    Returns:
        torch.Tensor: Exact score matching loss
    """
    if give_prec:
        prec_1 = cov_1
    else:
        prec_1 = torch.linalg.inv(cov_1)
    prec_0 = torch.linalg.inv(cov_0)
    h_0 = prec_0@mean_0
    h_1 = prec_1@mean_1

    loss = 0
    for ind, weight in zip(miss_inds, miss_weights):
        # Get conditional parameters
        prec_0_sub = torch.inverse(cov_0[ind][:, ind])
        new_h = h_1.detach().clone()
        new_h[ind] = h_0[ind]
        new_prec = prec_1.detach().clone()
        new_prec[ind, :][:, ind] = prec_0_sub

        new_cov = torch.inverse(new_prec)
        new_mean = new_cov @ new_h
        loss += weight*exact_score_loss(new_mean, mean_1, new_cov, cov_1,
                                        give_prec=give_prec, normalise=False)
    return loss


def exact_score_loss_marginal(mean_0, mean_1, cov_0, cov_1, miss_inds, miss_weights, give_prec=False,  normalise=True):
    """Calculates the exact score matching for proposal normal given true underlying normal distribution

    Args:
        mean_0 (torch.Tensor): True mean
        mean_1 (torch.Tensor): Proposed Mean
        cov_0 (torch.Tensor): True Covariance
        cov_1 (torch.Tensor): Proposed Covariance (or Precision if give_prec is True)
        miss_inds (list): A list of tensors containing the possible observed indices patterns
        miss_weights (list): A list of probabilities for each missing pattern in miss_inds.

    Returns:
        torch.Tensor: Exact score matching loss
    """
    if give_prec:
        cov_1 = torch.linalg.inv(cov_1)
    loss = 0
    for ind, weight in zip(miss_inds, miss_weights):
        # Get conditional parameters
        mean_0_sub = mean_0[ind]
        mean_1_sub = mean_1[ind]
        cov_0_sub = cov_0[ind, :][:, ind]
        cov_1_sub = cov_1[ind, :][:, ind]

        loss += weight*exact_score_loss(mean_0_sub, mean_1_sub, cov_0_sub, cov_1_sub,
                                        give_prec=False, normalise=normalise)

    return loss


def exact_KL(mean_0, mean_1, cov_0, cov_1, give_prec=False, normalise=True):
    """Calculates the KL divergence for proposal normal given true underlying normal distribution

    Args:
        mean_0 (torch.Tensor): True mean
        mean_1 (torch.Tensor): Proposed Mean
        cov_0 (torch.Tensor): True Covariance
        cov_1 (torch.Tensor): Proposed Covariance or Precision if give_prec is True
        give_prec (bool, optional): Whether to give precision instead of covariance. Defaults to False.

    Returns:
        torch.Tensor: Exact score matching loss
    """
    if give_prec:
        prec_1 = cov_1
    else:
        prec_1 = torch.linalg.inv(cov_1)
    log_det_term_1 = -0.5 * torch.logdet(2 * torch.pi * cov_0)
    log_det_term_2 = -0.5 * torch.logdet(prec_1 / (2 * torch.pi))
    mean_var_term_1 = -mean_0.shape[0] / 2
    var_term_2 = 0.5 * torch.sum(
        torch.diagonal(prec_1 @ cov_0, dim1=-2, dim2=-1), dim=-1
    )
    mean_term_2 = 0.5 * (
        (mean_0 - mean_1).unsqueeze(-2) @ prec_1 @ (mean_0 - mean_1).unsqueeze(-1)
    ).squeeze((-1, -2))
    exact_loss = (
        log_det_term_1 + mean_var_term_1 + log_det_term_2 + var_term_2 + mean_term_2
    )
    if normalise:
        exact_loss -= exact_KL(mean_0, mean_0, cov_0, cov_0, normalise=False)
    return exact_loss


def approx_fisher_div(x: torch.Tensor, est_udensity: UDensity, true_udensity: UDensity):
    """Approximates the fisher divergence between two densities using samples

    Args:
        x (torch.Tensor): Sample from the true density
        est_udensity (nn.Module): Estimated density
        true_udensity (nn.Module): True density

    Returns:
        torch.Tensor: Fisher divergence
    """
    if isinstance(est_udensity, UDensity) & isinstance(true_udensity, UDensity):
        est_score = est_udensity.score(x)
        true_score = true_udensity.score(x)
    else:
        x.requires_grad_(True)
        true_logdens = torch.log(true_udensity(x))
        est_logdens = torch.log(est_udensity(x))
        true_score = grad(torch.sum(true_logdens), x)[0]
        est_score = grad(torch.sum(est_logdens), x)[0]
        x.requires_grad_(False)
    return torch.mean(torch.sum((true_score - est_score)**2, dim=-1))


def extrapolate(x, y, log=False):
    if log:
        x = np.log10(x)
        y = np.log10(y)
    newx = 2*x-y
    if log:
        newx = 10**newx
    return newx


def reg_gridsearch(runner, density_range, abs_density_range, start_range, steps,
                   threshold, abs_threshold, iter_per_step=10, burn_in=200, max_recur=5, log=True, **kwargs):
    lower_ind = torch.tril_indices(row=runner.q_theta.Precision.shape[0],
                                   col=runner.q_theta.Precision.shape[1],
                                   offset=-1)
    # Re-order start_range so its larger followed by smaller
    if start_range[0] < start_range[1]:
        currentrange = (start_range[1], start_range[0])
    else:
        currentrange = start_range
    print(f"Starting range: {currentrange}")
    niters = torch.tensor([burn_in]+[iter_per_step]*(steps-1)).int()
    for i in range(max_recur):
        if log:
            l1_regs = torch.logspace(
                np.log10(currentrange[0]), np.log10(currentrange[1]), steps=steps)
        else:
            l1_reg = torch.linspace(*currentrange, steps=steps)
        stored_vals = []
        abs_stored_vals = []
        for j, l1_reg in enumerate(l1_regs):
            # Update optimiser
            runner.q_theta_opt.param_groups[1]["l1_reg"] = l1_reg.item()
            # Update training args
            runner.train(niters=niters[j], nepochs=niters[j], snapshot_freq=100)
            # Add stored vals from this run to all stored_vals
            lower_prec = runner.q_theta.Precision[lower_ind[0], lower_ind[1]]
            stored_vals.append(torch.mean((lower_prec > threshold).float()).item())
            abs_stored_vals.append(torch.mean((torch.abs(lower_prec) > abs_threshold).float()).item())
        stored_vals = torch.tensor(stored_vals)
        abs_stored_vals = torch.tensor(abs_stored_vals)
        # At the end of each loop, update the l1_regs
        if (stored_vals[0] > density_range[0]) or (abs_stored_vals[0] > abs_density_range[0]):
            l1_new_start = extrapolate(l1_regs[0], l1_regs[(steps//10)+1], log=log).item()
            print("Upper extrapolated")
        else:
            new_start = torch.min(torch.cat(
                (torch.nonzero(stored_vals > density_range[0]),
                 torch.nonzero(abs_stored_vals > abs_density_range[0]))
                )).item()-1
            l1_new_start = l1_regs[new_start].item()
        if (stored_vals[-1] < density_range[1]) or (abs_stored_vals[-1] < abs_density_range[1]):
            l1_new_end = extrapolate(l1_regs[-1], l1_regs[((9*steps)//10)-1], log=log).item()
            print("Lower extrapolated")
        else:
            new_end = torch.max(torch.cat((
                torch.nonzero(stored_vals < density_range[1]),
                torch.nonzero(abs_stored_vals < abs_density_range[1])
                ))).item()+1
            l1_new_end = l1_regs[new_end].item()
        # Update currentrange
        currentrange = (l1_new_start, l1_new_end)
        print(f'New range: {currentrange}')
    return currentrange


def pos_rate_reg(positive_rates, type="changes", angles=True):
    n = positive_rates.shape[-1]
    if angles:
        neg_angles = F.relu(-torch.arctan(torch.diff(positive_rates, dim=-1) * n))
    else:
        neg_angles = F.relu(-torch.diff(positive_rates, dim=-1))
    if type == "changes":
        pos_angle_ind = neg_angles == 0
        regularisation = torch.sum(
            neg_angles[..., 1:] * (pos_angle_ind[..., :-1]), dim=-1
        )
    elif type == "all":
        regularisation = torch.sum(neg_angles, dim=-1)
    return regularisation


def threshold_selector(
    scores, start_range=(1e-6, 1e-1), threshold=5, iters=5, steps=10, verbose=False, **kwargs
):
    # Check start range is in increasing order and if not make it so
    if start_range[0] > start_range[1]:
        start_range = (start_range[1], start_range[0])

    current_range = start_range
    for i in range(iters):
        if verbose:
            print(f"Current range: {current_range}")
        positive_rates = []
        # Set-up grid
        candidate_thresholds = torch.logspace(
            np.log10(current_range[0]), np.log10(current_range[1]), steps=steps
        )
        for candidate_threshold in candidate_thresholds:
            # Get positive rates for thresholds in shape (steps, nchange_points)
            positive_rates.append(
                torch.mean((scores > candidate_threshold).float(), dim=-1)
            )
            # Get regularisation levels of shape (steps)
        positive_rates = torch.stack(positive_rates, dim=0)
        reg_levels = pos_rate_reg(positive_rates, **kwargs)
        print(f"Reg levels at current range are: {reg_levels[0], reg_levels[-1]}")
        # print(reg_levels)
        # Find the last time reg_level that is above the threshold
        failure_locations = torch.nonzero(reg_levels > threshold).squeeze()
        if failure_locations.nelement() == 0:
            print("Breaking early")
            return candidate_thresholds[0].item()
        else:
            new_start = torch.max(failure_locations).item()

        if new_start == steps - 1:
            warn("Upper threshold still too low for stability. Returning upper limit")
            return candidate_thresholds[new_start]
        else:
            current_range = (
                candidate_thresholds[new_start].item(),
                candidate_thresholds[new_start + 1].item(),
            )
    return candidate_thresholds[new_start + 1].item()


# %%

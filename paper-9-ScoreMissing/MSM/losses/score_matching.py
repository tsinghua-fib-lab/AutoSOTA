import torch
from torch.autograd import grad
from typing import Union, Tuple, Callable
from ..models import density_models as models
from ..models import variational_models as var_models


def ml_loss(X: torch.Tensor, q_theta: models.UDensity):
    """Maximum likelihood loss

    Args:
        X (torch.Tensor): Data
        q_theta (models.UDensity): unnormalised density model

    Returns:
        torch.Tensor: Maximum likelihood loss
    """
    return -torch.mean(q_theta.log_prob(X))


def get_score_and_hess(X: torch.Tensor, q_theta: models.UDensity) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get score and diagonal of hessian from unnormalised density

    Args:
        X (torch.Tensor): Data
        q_theta (models.UDensity): Unnormalised density model

    Returns:
        (torch.Tensor, torch.Tensor): Score and Hessian values for data
    """
    X = X.detach().requires_grad_(True)
    d = X.shape[-1]
    q_theta_log_prob = q_theta.log_prob(X)

    score_vals = grad(torch.sum(q_theta_log_prob), X, create_graph=True)[0]
    # Make appropriate terms of score vals 0
    score_vals = score_vals
    hess_vals = []
    # Compute Hessian diagonal (curl of gradient)
    for i in range(d):
        temp_hess = grad(torch.sum(score_vals[..., i]), X, create_graph=True)[0][..., i]
        hess_vals.append(temp_hess.clone())
    hess_vals = torch.stack(hess_vals, dim=-1)
    X.requires_grad_(False)
    return (score_vals, hess_vals)


def get_trunc_vals(X: torch.Tensor, trunc_func: Callable[[torch.Tensor], torch.Tensor],
                   elementwise_trunc=False, mask=None):
    # Get Truncation Values
    X.requires_grad_(True)
    if mask is not None:
        trunc_vals = trunc_func(X, mask)
    else:
        trunc_vals = trunc_func(X)
    grad_trunc_vals = grad(torch.sum(trunc_vals), X, create_graph=True)[0]
    trunc_vals = trunc_vals.detach()
    if not elementwise_trunc:
        trunc_vals = trunc_vals.unsqueeze(-1)
    X.requires_grad_(False)
    return trunc_vals, grad_trunc_vals


def score_matching(X: torch.Tensor, q_theta: models.UDensity):
    """Standard score matching objective

    Args:
        X (torch.Tensor): Data
        q_theta (models.UDensity): unnormalised density model

    Returns:
        torch.Tensor: Score matching objective
    """
    score_vals, hess_vals = get_score_and_hess(X, q_theta)
    return torch.mean(torch.sum(score_vals**2+2*hess_vals, dim=-1))


def imputed_score_matching(
        X_obs: torch.Tensor, mask: torch,
        p_phi: var_models.Imputer,
        q_theta: models.UDensity, sample: Union[torch.Tensor, None] = None):
    """Impute data then run normal score matching

    Args:
        X_obs (torch.Tensor): The corrupted data
        mask (torch): The data mask
        p_phi (var_models.VariationalDensity): Variational Model for imputation (needs impute_sample method)
        q_theta (models.UDensity): Unnormalised score model
        sample (torch.Tensor, optional): Optional imputing sample. Defaults to None.
    """
    X_imputed = p_phi.impute_sample(X_obs, mask, sample=sample, ncopies=1).squeeze(0)
    return score_matching(X_imputed, q_theta)


def imputed_zerod_score_matching(
        X_obs: torch.Tensor, mask: torch,
        p_phi: var_models.VariationalDensity,
        q_theta: models.UDensity, sample: Union[torch.Tensor, None] = None):
    """Impute data then run normal score matching only on observed dimensions.

    Args:
        X_obs (torch.Tensor): The corrupted data
        mask (torch): The data mask
        p_phi (var_models.VariationalDensity): Variational Model for imputation (needs impute_sample method)
        q_theta (models.UDensity): Unnormalised score model
        sample (torch.Tensor, optional): Optional imputing sample. Defaults to None.
    """
    X_imputed = p_phi.impute_sample(X_obs, mask, sample=sample, ncopies=1).squeeze(0)
    score_vals, hess_vals = get_score_and_hess(X_imputed, q_theta)
    return torch.mean(torch.sum(mask*(score_vals**2+2*hess_vals), dim=-1))


def trunc_score_matching(X: torch.Tensor, q_theta: models.UDensity,
                         trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         elementwise_trunc=False, trim_val=None,
                         return_all=False, *args, **kwargs):
    """Standard truncated score matching objective

    Args:
        X (torch.Tensor): Data
        q_theta (models.UDensity): unnormalised density model
        trunc_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Truncation function
        elementwise_trunc (bool, optional): Whether truncation is elementwise. Defaults to False.


    Returns:
        torch.Tensor: Score matching objective
    """
    score_vals, hess_vals = get_score_and_hess(X, q_theta)
    # Get Trunc Vals
    trunc_vals, grad_trunc_vals = get_trunc_vals(X, trunc_func, elementwise_trunc)

    return torch.mean(torch.sum(trunc_vals*(score_vals**2+2*hess_vals)+2*grad_trunc_vals*score_vals, dim=-1))


def trunc_imputed_score_matching(
        X_obs: torch.Tensor, mask: torch,
        p_phi: var_models.VariationalDensity,
        q_theta: models.UDensity,
        trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        elementwise_trunc=False, trim_val=None, sample: Union[torch.Tensor, None] = None):
    """Impute data then run truncated score matching

    Args:
        X_obs (torch.Tensor): The corrupted data
        mask (torch): The data mask
        p_phi (var_models.VariationalDensity): Variational Model for imputation (needs impute_sample method)
        q_theta (models.UDensity): Unnormalised score model
                trunc_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Truncation function
        elementwise_trunc (bool, optional): Whether truncation is elementwise. Defaults to False.
        sample (torch.Tensor, optional): Optional imputing sample. Defaults to None.
    """
    X_imputed = p_phi.impute_sample(X_obs, mask, sample=sample, ncopies=1).squeeze(0)
    return trunc_score_matching(X_imputed, q_theta, trunc_func, elementwise_trunc, trim_val)


def trunc_imputed_zerod_score_matching(
        X_obs: torch.Tensor, mask: torch,
        p_phi: var_models.VariationalDensity,
        q_theta: models.UDensity,
        trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        elementwise_trunc=False, trim_val=None, sample: Union[torch.Tensor, None] = None):
    """Impute data then run normal truncated score matching only on observed dimensions.

    Args:
        X_obs (torch.Tensor): The corrupted data
        mask (torch): The data mask
        p_phi (var_models.VariationalDensity): Variational Model for imputation (needs impute_sample method)
        q_theta (models.UDensity): Unnormalised score model
        trunc_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Truncation function
        elementwise_trunc (bool, optional): Whether truncation is elementwise. Defaults to False.
        sample (torch.Tensor, optional): Optional imputing sample. Defaults to None.
    """
    X_imputed = p_phi.impute_sample(X_obs, mask, sample=sample, ncopies=1).squeeze(0)
    score_vals, hess_vals = get_score_and_hess(X_imputed, q_theta)
    trunc_vals, grad_trunc_vals = get_trunc_vals(X_imputed, trunc_func, elementwise_trunc, mask)
    return torch.mean(torch.sum(mask*(trunc_vals*(score_vals**2+2*hess_vals)+2*grad_trunc_vals*score_vals), dim=-1))

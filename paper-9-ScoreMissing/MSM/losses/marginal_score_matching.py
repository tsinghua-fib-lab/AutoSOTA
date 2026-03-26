import torch
from torch.autograd import grad
from typing import Union, Callable
from ..models import density_models as models
from ..models import variational_models as var_models


def mylogsumexp(tensor: torch.Tensor, dim: int, keepdim=False):
    # the logsumexp of pytorch is not stable!
    tensor_max, _ = tensor.max(dim=dim, keepdim=True)
    ret = (tensor - tensor_max).exp().sum(dim=dim, keepdim=True).log() + tensor_max
    if not keepdim:
        ret.squeeze_(dim=dim)
    return ret


def variational_fisher(X_obs: torch.Tensor, mask: torch.Tensor,
                       p_phi: var_models.VariationalDensity,
                       q_theta: models.UDensity):
    miss_points = (torch.min(mask, 1)[0] == 0.)
    X_obs_sub = X_obs[miss_points].clone().detach()
    mask_sub = mask[miss_points].clone().detach()
    if p_phi.has_sample_and_log_prob:
        X_imputed, log_p_phi = p_phi.impute_sample_and_log_prob(X_obs_sub, mask_sub, ncopies=1)
        log_p_phi = log_p_phi.squeeze(0)
        X_imputed = X_imputed.squeeze(0)
    else:
        X_imputed = p_phi.impute_sample(X_obs_sub, mask_sub, ncopies=1).squeeze(0)
        # Get log density
        log_p_phi = p_phi.log_prob(X_imputed, mask_sub)
    # DELETE THIS
    if not X_imputed.requires_grad:
        X_imputed.requires_grad_(True)
    log_q_theta = q_theta.log_prob(X_imputed)
    # Take difference
    diff = log_p_phi - log_q_theta
    grad_diff = grad(diff.sum(), X_imputed, create_graph=True)[0]*(1-mask_sub)
    loss = (grad_diff ** 2).sum(dim=-1).mean()
    return loss


def variational_kl(X_obs: torch.Tensor, mask: torch.Tensor,
                   p_phi: var_models.VariationalDensity,
                   q_theta: models.UDensity):
    miss_points = (torch.min(mask, 1)[0] == 0.)
    X_obs_sub = X_obs[miss_points].clone().detach()
    mask_sub = mask[miss_points].clone().detach()
    if p_phi.has_sample_and_log_prob:
        X_imputed, log_p_phi = p_phi.impute_sample_and_log_prob(X_obs_sub, mask_sub, ncopies=1)
        log_p_phi = log_p_phi.squeeze(0)
        X_imputed = X_imputed.squeeze(0)
    else:
        X_imputed = p_phi.impute_sample(X_obs_sub, mask_sub, ncopies=1).squeeze(0)
        # Get log densities
        log_p_phi = p_phi.log_prob(X_imputed, mask_sub)
    log_q_theta = q_theta.log_prob(X_imputed)
    # Take difference
    diff = log_p_phi - log_q_theta
    return diff.mean()


def impute_mcmc(X_obs: torch.Tensor, mask: torch.Tensor, sample_ind: torch.Tensor,
                p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                markov_chain: torch.Tensor, n_candidates: int = 10):
    "Imputes using IW and update the Markov Chain inplace"
    # Get original imputing samples
    X_candidate = p_phi.impute_sample(X_obs, mask, ncopies=n_candidates-1)
    # Prepend on previous proposal
    local_mc = markov_chain[sample_ind]
    X_candidate = torch.cat([local_mc.unsqueeze(0), X_candidate], dim=0)
    # Get log densities
    log_p_phi = p_phi.log_prob(X_candidate, mask)
    log_q_theta = q_theta.log_prob(X_candidate)
    # Get Normalised Importance weights
    iw = torch.exp(log_q_theta - log_p_phi)
    iw = iw / iw.sum(dim=0, keepdim=True)
    # if torch.any(torch.isnan(iw)):
    #     print("NAN in iw")
    # Replace NaNs with 1.
    iw = torch.where(torch.isnan(iw), torch.ones_like(iw), iw)
    # Get new proposal
    indices = torch.multinomial(iw.T, 1).squeeze(1)
    X_imputed = X_candidate[indices, torch.arange(X_candidate.shape[1])]
    # Update markov chain inplace with these values
    markov_chain[sample_ind] = X_imputed
    return X_imputed


def variational_forward_fisher(
        X_obs: torch.Tensor, mask: torch.Tensor, indices: torch.Tensor,
        p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
        markov_chain: torch.Tensor, n_candidates: int = 10):

    miss_points = (torch.min(mask, 1)[0] == 0.)
    X_obs_sub = X_obs[miss_points].clone().detach()
    mask_sub = mask[miss_points].clone().detach()
    indices_sub = indices[miss_points].clone().detach()

    # Impute and update markov chain inplace
    X_imputed = impute_mcmc(X_obs_sub, mask_sub, indices_sub, p_phi, q_theta, markov_chain, n_candidates)
    # Allow us to take derivates w.r.t. x
    # Note: Now samples are "From" q_theta we no longer need to backpropr through sample gen hence the detach()
    X_imputed = X_imputed.detach().requires_grad_(True)
    # Get log density
    log_p_phi = p_phi.log_prob(X_imputed, mask_sub)
    log_q_theta = q_theta.log_prob(X_imputed)
    # Take difference
    diff = log_p_phi - log_q_theta
    grad_diff = grad(diff.sum(), X_imputed, create_graph=True)[0]*(1-mask_sub)
    loss = (grad_diff ** 2).sum(dim=-1).mean()
    return loss


def variational_forward_kl(X_obs: torch.Tensor, mask: torch.Tensor, indices: torch.Tensor,
                           p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                           markov_chain: torch.Tensor, n_candidates: int = 10):
    miss_points = (torch.min(mask, 1)[0] == 0.)
    X_obs_sub = X_obs[miss_points].clone().detach()
    mask_sub = mask[miss_points].clone().detach()
    indices_sub = indices[miss_points].clone().detach()
    X_imputed = impute_mcmc(X_obs_sub, mask_sub, indices_sub, p_phi, q_theta, markov_chain, n_candidates)
    # Get log densities
    log_p_phi = p_phi.log_prob(X_imputed, mask_sub)
    log_q_theta = q_theta.log_prob(X_imputed)
    # Take difference
    diff = log_p_phi - log_q_theta
    return diff.mean()


def bilevel_scoreloss(X_obs: torch.Tensor, mask: torch.Tensor,
                      p_phi: var_models.VariationalDensity,
                      q_theta: models.UDensity):
    X_obs = X_obs.detach().clone().requires_grad_(True)
    d = X_obs.shape[1]
    X_imputed = p_phi.impute_sample(X_obs, mask)
    log_p_phi = p_phi.log_prob(X_imputed, mask)
    log_q_theta = q_theta.log_prob(X_imputed)
    lop_q_theta_marg = log_q_theta-log_p_phi
    score = grad(lop_q_theta_marg.sum(), X_imputed, create_graph=True)[0]*(1-mask)
    score_sq_term = torch.sum(score**2, dim=-1)
    trace_term = 0
    for j in range(d):
        trace_term += 2*grad(score[..., j].sum(), X_imputed, create_graph=True)[0][..., j]*(1-mask[..., j])
    loss = score_sq_term + trace_term
    return loss.mean()


def get_iw(X_full: torch.Tensor, mask: torch.Tensor,
           p_phi: var_models.VariationalDensity,
           q_theta: models.UDensity, self_normalised=True, return_all=False):
    numer_density = q_theta.forward(X_full)
    denom_density = p_phi.forward(X_full, mask)
    iw = numer_density / denom_density
    if self_normalised:
        iw = iw / torch.mean(iw, dim=0, keepdim=True)
    if not return_all:
        return iw
    else:
        return iw, {"denom_density": denom_density,
                    "numer_density": numer_density}


def iw_cov(X: torch.Tensor, Y: torch.Tensor,
           iw: Union[torch.Tensor, None] = None):
    min_val = torch.tensor([-1e30])
    max_val = torch.tensor([1e30])
    if iw is None:
        cov_term1 = torch.mean(X * Y, dim=0)
        means = [torch.mean(X, dim=0),
                 torch.mean(Y, dim=0)]
    else:
        X = torch.clamp(X, min_val, max_val)
        Y = torch.clamp(Y, min_val, max_val)
        cov_term1 = torch.mean(iw * X * Y, dim=0)
        means = [torch.mean(iw * X, dim=0),
                 torch.mean(iw * Y, dim=0)]

    cov_term2 = means[0] * means[1]
    return cov_term1 - cov_term2


def iw_mean(X: torch.Tensor, iw: Union[torch.Tensor, None] = None):
    min_val = torch.tensor([-1e30])
    max_val = torch.tensor([1e30])
    if iw is None:
        return torch.mean(X, dim=0)
    else:
        return torch.mean(iw * torch.clamp(X, min_val, max_val), dim=0)


def get_iw_density_and_derivs(X: torch.Tensor, mask: torch.Tensor,
                              p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                              do_iw=True, ncopies=10, sample=None,
                              return_all=False):
    d = X.shape[1]
    if not do_iw:
        X_imputed = p_phi.impute_sample(X, mask, sample=sample, ncopies=ncopies, return_all=return_all)
        if return_all:
            X_imputed, imputing_extras = X_imputed
        iw = None
        p_phi_log_prob = torch.zeros_like(X_imputed[..., 0])
    else:
        if p_phi.has_sample_and_log_prob and sample is None:
            X_imputed, p_phi_log_prob = p_phi.impute_sample_and_log_prob(X, mask, ncopies=ncopies)
        else:
            if sample is not None:
                X_imputed = (mask*X+(1-mask)*sample).detach()
            else:
                X_imputed = p_phi.impute_sample(X, mask, ncopies=ncopies)
            p_phi_log_prob = p_phi.log_prob(X_imputed, mask)

    # Calculated log density of q but first allow backprop through x
    X_imputed = X_imputed.detach().requires_grad_(True)
    q_theta_log_prob = q_theta.log_prob(X_imputed)
    if do_iw:
        iw = torch.exp(q_theta_log_prob - p_phi_log_prob)
        # Keep 0s as 0s but move any v small values up to 1e-30
        iw = torch.where((0 < iw) & (iw < 1e-42), torch.tensor([1e-30]), iw)

        # Normalise to have mean 1
        iw = (iw / iw.sum(dim=0, keepdim=True))*iw.shape[0]

        # # Replace NaNs with 1
        nan_iws = torch.isnan(iw)
        if torch.any(torch.isnan(iw)):
            iw[nan_iws] = 1.
        #     # Re-normalise
        #     iw = iw / iw.mean(dim=0, keepdim=True)
        # We do not want to backprop through the importance weights also we multiply by multi-dimdata so unsqueeze
        iw = iw.detach().unsqueeze(-1)

    score_vals = grad(torch.sum(q_theta_log_prob), X_imputed, create_graph=True)[0]
    # Make appropriate terms of score vals 0
    score_vals = score_vals * mask.unsqueeze(0)
    hess_vals = []
    # Compute Hessian diagonal (curl of gradient)
    for i in range(d):
        temp_hess = grad(torch.sum(score_vals[..., i]), X_imputed, create_graph=True)[0][..., i]
        temp_hess = mask[..., i].unsqueeze(0) * temp_hess
        hess_vals.append(temp_hess.clone())
    hess_vals = torch.stack(hess_vals, dim=-1)
    X_imputed.requires_grad_(False)
    # Now set up gradient to behave as expected
    q_theta_log_prob = q_theta_log_prob.unsqueeze(-1)
    if not return_all:
        return (iw, q_theta_log_prob, score_vals, hess_vals)
    else:
        return_dict = {
            "numer_density": q_theta_log_prob.exp(),
            "denom_density": p_phi_log_prob.exp(),
            "X_imputed": X_imputed
        }
        if not do_iw:
            return_dict["imputing_extras"] = imputing_extras
        return (iw, q_theta_log_prob, score_vals, hess_vals, return_dict)


def scoreloss_marginal(X: torch.Tensor, mask: torch.Tensor,
                       p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                       do_iw=True, ncopies=10, sample=None,
                       trim_val=None, return_all=False):

    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)

    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out
    score_vals_sq = score_vals**2

    term_1 = -2*iw_mean(score_vals.detach(), iw)*(
        iw_mean(score_vals, iw) + iw_cov(score_vals.detach(), q_theta_log_prob, iw)
    )
    term_2 = 2*(iw_mean(score_vals_sq, iw)
                + iw_cov(score_vals_sq.detach(), q_theta_log_prob, iw))
    term_3 = 2*(iw_mean(hess_vals, iw) + iw_cov(hess_vals.detach(), q_theta_log_prob, iw))

    losses = torch.sum(term_1 + term_2 + term_3, dim=-1)
    if trim_val is not None:
        trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
        loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
    else:
        loss_val = torch.mean(losses[torch.isfinite(losses)])
    if not return_all:
        return loss_val
    else:
        return (loss_val, return_vals | {
            "iw": iw,
            "score_vals": score_vals,
            "hess_vals": hess_vals,
            "g_vals": q_theta_log_prob,
            "term_1": term_1,
            "term_2": term_2,
            "term_3": term_3})


def scoreloss_em(X: torch.Tensor, mask: torch.Tensor,
                 p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                 do_iw=True, ncopies=10, sample=None, trim_val=None):

    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample)
    iw, q_theta_log_prob, score_vals, hess_vals = out
    score_vals_sq = score_vals**2
    all_losses = torch.sum(2*hess_vals+score_vals_sq, dim=-1)
    losses = iw_mean(all_losses, iw.squeeze(-1))
    if trim_val is not None:
        trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
        loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
    else:
        loss_val = torch.mean(losses[torch.isfinite(losses)])
    return loss_val


def true_scoreloss_marginal(X: torch.Tensor, mask: torch.Tensor,
                            p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                            do_iw=True, ncopies=10, sample=None,
                            trim_val=None, return_all=False):

    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)
    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out
    with torch.no_grad():
        score_vals_sq = score_vals**2
        term_1 = -iw_mean(score_vals, iw)**2
        term_2 = 2*iw_mean(score_vals_sq, iw)
        term_3 = 2*iw_mean(hess_vals, iw)
        losses = torch.sum(term_1 + term_2 + term_3, dim=-1)
    if trim_val is not None:
        trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
        loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
    else:
        loss_val = torch.mean(losses[torch.isfinite(losses)])
    if not return_all:
        return loss_val
    else:
        return (loss_val, {
            "numer_density": return_vals["numer_density"],
            "denom_density": return_vals["denom_density"],
            "X_imputed": return_vals["X_imputed"],
            "iw": iw,
            "score_vals": score_vals,
            "hess_vals": hess_vals,
            "g_vals": q_theta_log_prob,
            "term_1": term_1,
            "term_2": term_2,
            "term_3": term_3})


def get_trunc_vals(X: torch.Tensor, mask: torch.Tensor,
                   trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   elementwise_trunc=False):
    # Get Truncation Values
    X.requires_grad_(True)
    trunc_vals = trunc_func(X, mask)
    grad_trunc_vals = grad(torch.sum(trunc_vals), X, create_graph=True)[0]
    trunc_vals = trunc_vals.detach()
    if not elementwise_trunc:
        trunc_vals = trunc_vals.unsqueeze(-1)
    X.requires_grad_(False)
    return trunc_vals, grad_trunc_vals


def scoreloss_marginal_trunc(X: torch.Tensor, mask: torch.Tensor,
                             p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                             trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                             elementwise_trunc=False, do_iw=True, ncopies=10,
                             sample=None, trim_val=None, return_all=False, *args, **kwargs):

    # Get score and Hessian etc
    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)
    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out
    score_vals_sq = score_vals**2

    # Get Trunc Vals
    trunc_vals, grad_trunc_vals = get_trunc_vals(X, mask, trunc_func, elementwise_trunc)

    # Set up each term in loss
    marg_score = -iw_mean(score_vals.detach(), iw)
    nabla_theta_marg_score = 2*(iw_mean(score_vals, iw) + iw_cov(score_vals.detach(), q_theta_log_prob, iw))
    nabla_theta_score_sq = 2*(iw_mean(score_vals_sq, iw)
                              + iw_cov(score_vals_sq.detach(), q_theta_log_prob, iw))
    nabla_theta_hess_score = 2*(iw_mean(hess_vals, iw) + iw_cov(hess_vals.detach(), q_theta_log_prob, iw))
    # Get loss for each sample
    losses = torch.sum(
        trunc_vals*(marg_score*nabla_theta_marg_score + nabla_theta_score_sq + nabla_theta_hess_score)
        + grad_trunc_vals*nabla_theta_marg_score, dim=-1)

    if trim_val is not None:
        trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
        loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
    else:
        loss_val = torch.mean(losses[torch.isfinite(losses)])
    if not return_all:
        return loss_val
    else:
        return (loss_val, return_vals | {
            "iw": iw,
            "score_vals": score_vals,
            "hess_vals": hess_vals,
            "g_vals": q_theta_log_prob,
            "trunc_vals": trunc_vals,
            "grad_trunc_vals": grad_trunc_vals,
            "term_10": marg_score,
            "term_11": nabla_theta_marg_score,
            "term_2": nabla_theta_score_sq,
            "term_3": nabla_theta_hess_score})


def scoreloss_em_trunc(X: torch.Tensor, mask: torch.Tensor,
                       p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
                       trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                       elementwise_trunc=False, do_iw=True, ncopies=10,
                       sample=None, trim_val=None, return_all=False, *args, **kwargs):
    # Get score and Hessian etc
    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)
    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out
    score_vals_sq = score_vals**2

    # Get Trunc Vals
    trunc_vals, grad_trunc_vals = get_trunc_vals(X, mask, trunc_func, elementwise_trunc)
    all_losses = torch.sum(trunc_vals*score_vals_sq
                           + 2*(trunc_vals*hess_vals+grad_trunc_vals*score_vals), dim=-1)
    losses = iw_mean(all_losses, iw.squeeze(-1))
    if trim_val is not None:
        trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
        loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
    else:
        loss_val = torch.mean(losses[torch.isfinite(losses)])
    return loss_val


def true_scoreloss_marginal_trunc(
    X: torch.Tensor, mask: torch.Tensor,
    p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
    trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elementwise_trunc=False, do_iw=True,
    ncopies=10, sample=None,
    trim_val=None, return_all=False,
    *args, **kwargs
):

    # Get score and Hessian etc
    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)
    trunc_vals, grad_trunc_vals = get_trunc_vals(X, mask, trunc_func, elementwise_trunc)
    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out

    with torch.no_grad():
        score_vals_sq = score_vals**2
        # Get Trunc Vals

        # Set up each term in loss
        term_1 = trunc_vals*(-iw_mean(score_vals, iw)**2+2*iw_mean(score_vals_sq, iw)+2*iw_mean(hess_vals, iw))
        term_2 = 2*grad_trunc_vals*iw_mean(score_vals, iw)
        losses = torch.sum(term_1 + term_2, dim=-1)
        if trim_val is not None:
            trim_vec = (losses < trim_val & torch.isfinite(losses)).float()
            loss_val = torch.sum(losses * trim_vec) / torch.sum(trim_vec)
        else:
            loss_val = torch.mean(losses[torch.isfinite(losses)])
        if not return_all:
            return loss_val
        else:
            return (loss_val, return_vals | {
                "iw": iw,
                "score_vals": score_vals,
                "hess_vals": hess_vals,
                "g_vals": q_theta_log_prob,
                "trunc_vals": trunc_vals,
                "grad_trunc_vals": grad_trunc_vals,
                "term_1": term_1,
                "term_2": term_2})


def true_fisher_marginal(
    X: torch.Tensor, mask: torch.Tensor,
    p_phi: var_models.VariationalDensity, q_theta: models.UDensity,
    true_q_theta: models.UDensity,
    do_iw=True, ncopies=10, sample=None,
    trim_val=None, return_all=False,
    *args, **kwargs
):

    # Get score and Hessian etc
    out = get_iw_density_and_derivs(X, mask, p_phi, q_theta, do_iw, ncopies, sample, return_all)
    out_true = get_iw_density_and_derivs(X, mask, p_phi, true_q_theta, do_iw, ncopies, sample, return_all)
    if not return_all:
        iw, q_theta_log_prob, score_vals, hess_vals = out
        iw_true, _, score_vals_true, _ = out_true
    else:
        iw, q_theta_log_prob, score_vals, hess_vals, return_vals = out
        iw_true, _, score_vals_true, _, return_vals_true = out_true

    with torch.no_grad():
        marg_scores = -iw_mean(score_vals.detach(), iw)
        marg_scores_true = -iw_mean(score_vals_true.detach(), iw_true)
        return torch.sum(marg_scores**2-2*marg_scores*marg_scores_true, dim=-1).mean()

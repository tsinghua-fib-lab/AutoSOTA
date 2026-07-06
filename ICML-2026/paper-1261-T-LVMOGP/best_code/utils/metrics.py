import torch

def gaussian_nll(gt, pred_mean, pred_var):
    """
    Gaussian Negative Log Likelihood.
    gt: [..., n]
    pred_mean / pred_var: [..., n]
    """
    pred_var = torch.clamp(pred_var, min=1e-8)
    const = 0.5 * torch.log(2 * torch.tensor(torch.pi, device=gt.device))
    nll = 0.5 * torch.log(pred_var) + 0.5 * (gt - pred_mean) ** 2 / pred_var + const
    return nll  # [..., n]

def mc_gaussian_nll(gt, pred_mean, pred_var):
    """
    Monte Carlo Negative Log Likelihood.
    gt: [..., n]
    pred_mean / pred_var: [..., s, n], where s is the number of Monte Carlo samples
        These are conditional predictive mean and variances, i.e. q(y | x, h_s), h_s ~ q(h)
    """
    assert gt.size(-1) == pred_mean.size(-1) == pred_var.size(-1)
    pred_var = torch.clamp(pred_var, min=1e-8)
    S = pred_mean.size(-2)  # number of MC samples

    # conditional (Gaussian) log likelihood
    con_gau_ll = - gaussian_nll(gt.unsqueeze(-2), pred_mean, pred_var)  # [..., s, n]

    # log likelihood via log-sum-exp
    log_lik = torch.logsumexp(con_gau_ll, dim=-2) - torch.log(torch.tensor(S, dtype=pred_mean.dtype, device=pred_mean.device))  # [..., n]

    return - log_lik  # [..., n]





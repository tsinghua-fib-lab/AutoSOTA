import torch


def pinball_loss(quantile: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, mask):
    # quantile  [batch_size, 1]
    # y         [batch_size, 1]
    # alpha     [batch_size, 1]
    diff = y - quantile
    sign = (diff >= 0).to(torch.float32)
    ones = torch.ones_like(diff, dtype=torch.int, device=quantile.device)
    if mask is None:
        return alpha * sign * diff - torch.sub(ones, alpha) * torch.sub(ones, sign) * diff
    else:
        return (alpha * sign * diff - torch.sub(ones, alpha) * torch.sub(ones, sign) * diff) * (~mask).to(torch.float32)


def mse_loss(prediction, label, mask):
    if mask is None:
        return torch.nn.functional.mse_loss(prediction, label, reduce=False)
    else:
        return torch.nn.functional.mse_loss(prediction, label, reduce=False) * (~mask).to(torch.float32)


def width_loss(quantile_low: torch.Tensor, quantile_high: torch.Tensor, mask):
    if mask is None:
        return (quantile_high - quantile_low).abs().mean(dim=0)  # Make absolute because we DO NOT want negative intervals
    else:
        return (quantile_high - quantile_low).abs().mean(dim=0) * (~mask).to(torch.float32)


def coverage_loss(Y, q_low, q_high, alpha: float):
    coverage = torch.sum(torch.logical_and(torch.ge(Y, q_low), torch.le(Y, q_high)).to(torch.int32)) / Y.shape[0]
    misscoverage = coverage - (1 - alpha)
    if misscoverage < 0:
        return torch.abs(misscoverage * Y.shape[0] * 1.5)
    else:
        return torch.abs(misscoverage * Y.shape[0])


def chung_calib_loss(quantile: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor):
    alphas, indices = torch.unique(alpha, return_inverse=True)
    loss = torch.zeros((1,), device=quantile.device)
    for idx, alpha in enumerate(alphas):
        y_sel, quantile_sel = y[indices == idx], quantile[indices == idx]
        diff = y_sel - quantile_sel
        idx_under = (diff <= 0)
        empirical_quantile = torch.mean(idx_under.float()).item()
        if empirical_quantile < 1 - alpha:
            loss += torch.mean((y_sel - quantile_sel)[~idx_under])
        else:
            loss += torch.mean((quantile_sel - y_sel)[idx_under])
    return loss


def orthogonal_qr_loss(Y, q_low, q_high):
    covered = torch.logical_and(torch.ge(Y, q_low), torch.le(Y, q_high)).to(torch.float)
    width = (q_high - q_low).abs()
    corr_vec = torch.cat((covered.T, width.T), dim=0)
    return torch.corrcoef(corr_vec)[0, 1].abs() * Y.shape[0]
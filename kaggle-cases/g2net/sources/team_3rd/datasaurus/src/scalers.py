import torch


# https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    is_inf = torch.isinf(v)
    v[is_nan] = 0
    v[is_inf] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def nanstd(v, *args, inplace=False, unbiased=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    is_inf = torch.isinf(v)
    v[is_nan] = 0
    v[is_inf] = 0

    mean = nanmean(v, *args, inplace=False, **kwargs)
    numerator = ((v - mean) ** 2).sum(*args, **kwargs)
    N = (~is_nan).float().sum(*args, **kwargs)

    if unbiased:
        N -= 1

    return torch.sqrt(numerator / N)


def nanstd_mean(v, *args, inplace=False, unbiased=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    is_inf = torch.isinf(v)
    v[is_nan] = 0
    v[is_inf] = 0

    mean = v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    numerator = ((v - mean) ** 2).sum(*args, **kwargs)
    N = (~is_nan).float().sum(*args, **kwargs)

    if unbiased:
        N -= 1

    std = torch.sqrt(numerator / N)
    return std, mean


def imagenet_norm(x):
    means = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    stds = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    means = torch.broadcast_to(means, (x.shape[0], 3)).unsqueeze(-1).unsqueeze(-1)
    stds = torch.broadcast_to(stds, (x.shape[0], 3)).unsqueeze(-1).unsqueeze(-1)
    return x * stds + means


def standard_scaler(features, imagenet=False):
    std, mean = nanstd_mean(features, dim=[2, 3], keepdim=True)
    features = (features - mean) / std
    features = torch.nan_to_num(features, 0, 5, -5)

    if imagenet:
        return imagenet_norm(features)
    else:
        return features


def standard_scaler_1d(features):
    std, mean = torch.std_mean(features, dim=-1, keepdim=True)
    features = (features - mean) / std
    features = torch.nan_to_num(features, 0, 5, -5)
    return features


def minmax_scaler(features):
    spec_min, _ = torch.min(features, dim=2, keepdim=True)
    spec_min, _ = torch.min(spec_min, dim=3, keepdim=True)

    spec_max, _ = torch.max(features, dim=2, keepdim=True)
    spec_max, _ = torch.max(spec_max, dim=3, keepdim=True)

    return (features - spec_min) / (spec_max - spec_min)


def robust_scaler(x):
    q = torch.tensor([0.25, 0.50, 0.75], device=x.device)
    p25, p50, p75 = (
        torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), q, dim=2)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    return (x - p50) / (p75 - p25)

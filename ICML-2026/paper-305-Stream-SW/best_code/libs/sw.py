import torch
from torch.nn.functional import pad
import numpy as np
def SW(X, Y, a, b, L, p=2,thetas=None):
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    if(thetas is None):
        thetas = torch.randn(L, d, device=X.device)
        thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=1, keepdim=True))
    X_prod = torch.matmul(X, thetas.T)
    Y_prod = torch.matmul(Y, thetas.T)
    distance=torch.mean(one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p))**(1./p)
    return distance


def quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    idx = torch.searchsorted(cws, qs, right=False).T
    return torch.gather(xs, 0, torch.clamp(idx, 0, n - 1))

def one_dimensional_Wasserstein(u_values, v_values, u_weights=None, v_weights=None, p=2):
    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                               dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                               dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    u_sorter = torch.sort(u_values, 0)[1]
    u_values = torch.gather(u_values, 0, u_sorter)

    v_sorter = torch.sort(v_values, 0)[1]
    v_values = torch.gather(v_values, 0, v_sorter)

    u_weights = torch.gather(u_weights, 0, u_sorter)
    v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    pad_width = [(1, 0)] + (qs.ndim - 1) * [(0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    qs = pad(qs, how_pad)

    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0, keepdim=True)


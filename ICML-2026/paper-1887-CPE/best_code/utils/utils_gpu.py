from __future__ import annotations
import numpy as np
import torch
@torch.no_grad()
def mean_any(x, axis=None):
    if isinstance(x, torch.Tensor):
        if axis is None:
            return x.mean()
        if isinstance(axis, int):
            return x.mean(dim=axis)
        return x.mean(dim=tuple(axis))
    return np.mean(x, axis=axis)

@torch.no_grad()
def softmax(logits):
    """Stable softmax. Accepts numpy or torch, returns same type."""
    if isinstance(logits, np.ndarray):
        x = torch.from_numpy(logits).to(dtype=torch.float64)
        z = x - torch.max(x)
        e = torch.exp(z)
        out = e / torch.sum(e)
        return out.cpu().numpy()
    x = logits.to(dtype=torch.float64)
    z = x - torch.max(x)
    e = torch.exp(z)
    return e / torch.sum(e)

@torch.no_grad()
def entropy_categorical(p):
    """Categorical entropy. Accepts numpy or torch, returns float."""
    if isinstance(p, np.ndarray):
        x = torch.from_numpy(p).to(dtype=torch.float64)
    else:
        x = p.to(dtype=torch.float64)
    x = torch.clamp(x, 1e-12, 1.0)
    return float(-(x * torch.log(x)).sum().item())

@torch.no_grad()
def ess(weights):
    if isinstance(weights, np.ndarray):
        w = torch.from_numpy(weights).to(dtype=torch.float64)
    else:
        w = weights.to(dtype=torch.float64)
    w = w / w.sum()
    return float((1.0 / (w * w).sum()).item())

@torch.no_grad()
def normalize(weights):
    """Normalize nonnegative weights. If sum<=0, return uniform."""
    if isinstance(weights, np.ndarray):
        w = torch.from_numpy(weights).to(dtype=torch.float64)
        s = float(w.sum().item())
        if s <= 0:
            out = torch.full_like(w, 1.0 / w.numel(), dtype=torch.float64)
        else:
            out = w / s
        return out.cpu().numpy()
    w = weights.to(dtype=torch.float64)
    s = w.sum()
    if float(s.item()) <= 0:
        return torch.full_like(w, 1.0 / w.numel(), dtype=torch.float64)
    return w / s

@torch.no_grad()
def adjacency_from_weights(W, eps: float = 1e-12):
    """
    Convert weight matrix to adjacency.
    Supports NumPy arrays and Torch tensors.
    """
    if isinstance(W, np.ndarray):
        return (np.abs(W) > eps).astype(int)
    elif isinstance(W, torch.Tensor):
        return (W.abs() > eps).to(torch.int64)
    else:
        raise TypeError(f"Unsupported type: {type(W)}")

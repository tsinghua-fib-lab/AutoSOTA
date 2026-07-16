from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import Tensor, CovarianceDict, CovariancePair


def compute_linear_covariances(
    activation: Tensor,
    backprop: Tensor,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    A = activation
    G = backprop

    if A.dim() > 2:
        A = A.reshape(-1, A.shape[-1])
    if G.dim() > 2:
        G = G.reshape(-1, G.shape[-1])

    A_bias = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)
    cov_A = (A_bias.T @ A_bias) / A.size(0) + eps * torch.eye(
        A_bias.size(1), device=A.device, dtype=A.dtype
    )
    cov_G = (G.T @ G) / G.size(0) + eps * torch.eye(
        G.size(1), device=G.device, dtype=G.dtype
    )

    return cov_A, cov_G


def compute_conv2d_covariances(
    activation: Tensor,
    backprop: Tensor,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    X = activation
    X_unfold = F.unfold(X, kernel_size=kernel_size, padding=padding, stride=stride)
    X_unfold = X_unfold.transpose(1, 2).reshape(-1, X_unfold.size(1))
    X_unfold_bias = torch.cat([X_unfold, torch.ones_like(X_unfold[:, :1])], dim=1)

    cov_A = (X_unfold_bias.T @ X_unfold_bias) / X_unfold_bias.size(0) + eps * torch.eye(
        X_unfold_bias.size(1), device=X.device, dtype=X.dtype
    )

    GY = backprop
    GY = GY.permute(0, 2, 3, 1).reshape(-1, GY.size(1))
    cov_G = (GY.T @ GY) / GY.size(0) + eps * torch.eye(
        GY.size(1), device=GY.device, dtype=GY.dtype
    )

    return cov_A, cov_G


def compute_covariances(
    model: nn.Module,
    activations: CovarianceDict,
    backprops: CovarianceDict,
    eps: float = 1e-5,
) -> CovariancePair:
    cov_A: CovarianceDict = {}
    cov_G: CovarianceDict = {}

    target = model._module if hasattr(model, "_module") else model
    name_to_module = dict(target.named_modules())

    for name in backprops:
        if name not in name_to_module:
            continue
        if name not in activations:
            continue

        module = name_to_module[name]
        act = activations[name]
        grad = backprops[name]

        if isinstance(module, nn.Linear):
            cov_A[name], cov_G[name] = compute_linear_covariances(act, grad, eps)

        elif isinstance(module, nn.Conv2d):
            cov_A[name], cov_G[name] = compute_conv2d_covariances(
                act,
                grad,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                eps=eps,
            )

    return CovariancePair(A=cov_A, G=cov_G)


def compute_inverse_sqrt(
    covariances: CovariancePair,
    damping: float = 1e-3,
) -> Tuple[CovarianceDict, CovarianceDict]:
    inv_A: CovarianceDict = {}
    inv_G: CovarianceDict = {}

    for name in covariances.A:
        A = covariances.A[name]
        eva, evc = torch.linalg.eigh(
            A + damping * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        )
        inv_A[name] = evc @ torch.diag(eva.clamp(min=1e-6).rsqrt()) @ evc.T

        G = covariances.G[name]
        evg, evcg = torch.linalg.eigh(
            G + damping * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        )
        inv_G[name] = evcg @ torch.diag(evg.clamp(min=1e-6).rsqrt()) @ evcg.T

    return inv_A, inv_G


def accumulate_covariances(
    cov_list: List[CovariancePair],
) -> CovariancePair:
    if not cov_list:
        return CovariancePair(A={}, G={})

    avg_A: CovarianceDict = {}
    avg_G: CovarianceDict = {}
    n = len(cov_list)

    for name in cov_list[0].A:
        stacked = torch.stack([c.A[name] for c in cov_list])
        avg_A[name] = stacked.mean(dim=0)
    for name in cov_list[0].G:
        stacked = torch.stack([c.G[name] for c in cov_list])
        avg_G[name] = stacked.mean(dim=0)

    return CovariancePair(A=avg_A, G=avg_G)

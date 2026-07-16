from typing import List
import torch
import torch.nn as nn

from .types import Tensor


def clip_and_noise_gradients(
    model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
    batch_size: int,
    store_summed_grad: bool = False,
) -> None:
    params = _get_params_with_grad_sample(model)
    if not params:
        return

    device = params[0].device
    total_norm_sq = _compute_per_sample_norms_squared(params, batch_size, device)
    clip_factors = _compute_clip_factors(total_norm_sq, max_grad_norm)

    for p in params:
        grad_sample = _get_grad_sample(p)
        grad_sample = grad_sample.contiguous().view(batch_size, -1)

        clipped = grad_sample * clip_factors.unsqueeze(1)
        summed = clipped.sum(dim=0)

        if store_summed_grad:
            p.summed_grad = (summed / batch_size).view_as(p)

        noise = torch.randn_like(summed) * noise_multiplier * max_grad_norm

        p.grad = ((summed + noise) / batch_size).view_as(p)
        p.grad_sample = None


def _get_params_with_grad_sample(model: nn.Module) -> List[nn.Parameter]:
    return [
        p for p in model.parameters()
        if hasattr(p, "grad_sample") and p.grad_sample is not None
    ]


def _get_grad_sample(param: nn.Parameter) -> Tensor:
    grad_sample = param.grad_sample
    if isinstance(grad_sample, list):
        grad_sample = grad_sample[-1]
    return grad_sample


def _compute_per_sample_norms_squared(
    params: List[nn.Parameter],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    total_norm_sq = torch.zeros(batch_size, device=device)
    for p in params:
        grad_sample = _get_grad_sample(p)
        grad_sample = grad_sample.contiguous().view(batch_size, -1)
        total_norm_sq += grad_sample.norm(2, dim=1).pow(2)
    return total_norm_sq


def _compute_clip_factors(
    total_norm_sq: Tensor,
    max_grad_norm: float,
) -> Tensor:
    total_norm = total_norm_sq.sqrt()
    return (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

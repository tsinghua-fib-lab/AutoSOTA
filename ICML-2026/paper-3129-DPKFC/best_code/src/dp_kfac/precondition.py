import torch
import torch.nn as nn

from .types import Tensor, CovarianceDict


def precondition_per_sample_gradients(
    model: nn.Module,
    inv_A: CovarianceDict,
    inv_G: CovarianceDict,
) -> None:
    target = model._module if hasattr(model, "_module") else model

    for name, module in target.named_modules():
        if name not in inv_A or name not in inv_G:
            continue
        if not hasattr(module, "weight"):
            continue
        if not hasattr(module.weight, "grad_sample"):
            continue
        if module.weight.grad_sample is None:
            continue

        g_sample_w = module.weight.grad_sample
        if isinstance(g_sample_w, list):
            g_sample_w = g_sample_w[-1]

        if isinstance(module, nn.Conv2d):
            batch_size, out_ch = g_sample_w.shape[:2]
            g_sample_w = g_sample_w.view(batch_size, out_ch, -1)

        if module.bias is not None and hasattr(module.bias, "grad_sample"):
            g_sample_b = module.bias.grad_sample
            if isinstance(g_sample_b, list):
                g_sample_b = g_sample_b[-1]
            g_sample_aug = torch.cat([g_sample_w, g_sample_b.unsqueeze(2)], dim=2)
        else:
            g_sample_aug = g_sample_w

        temp = torch.einsum("oj,bjk->bok", inv_G[name], g_sample_aug)
        preconditioned = torch.einsum("bok,ki->boi", temp, inv_A[name])

        if module.bias is not None and hasattr(module.bias, "grad_sample"):
            new_w_grad = preconditioned[:, :, :-1]
            new_b_grad = preconditioned[:, :, -1]
        else:
            new_w_grad = preconditioned
            new_b_grad = None

        if isinstance(module, nn.Conv2d):
            new_w_grad = new_w_grad.view_as(module.weight.grad_sample)

        module.weight.grad_sample = new_w_grad
        if new_b_grad is not None and module.bias is not None:
            module.bias.grad_sample = new_b_grad

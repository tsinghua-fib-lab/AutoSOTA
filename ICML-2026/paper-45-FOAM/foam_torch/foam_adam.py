# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version
import numpy as np
import torch.nn.functional as F

def fold_tensor(x, level):
    factor = 2 ** level
    original_size = x.shape[-1]
    exp_x = x ** 2
    pad_size = (factor - original_size % factor) % factor

    if pad_size > 0:
        exp_x =  F.pad(exp_x, (0, pad_size), mode='constant', value=0)
        x = F.pad(x, (0, pad_size), mode='constant', value=0)

    x = x.view(*x.shape[:-1], x.shape[-1] // factor, factor)
    exp_x = exp_x.view(*exp_x.shape[:-1], exp_x.shape[-1] // factor, factor)
    
    return x.mean(dim=-1), exp_x.mean(dim=-1) ,original_size, pad_size

def unfold_tensor(x, original_size, pad_size, level):
    factor = 2 ** level
    unfolded = x.repeat_interleave(factor, dim=-1)
    return unfolded[..., :original_size] if pad_size > 0 else unfolded


class FOAM(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        res_scale: float = 1.0,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, \
                    "res_scale": res_scale}
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "level" in group:
                    folded, folded_exp, original_size, pad_size = fold_tensor(p.grad, level=group["level"])
                    unfold = unfold_tensor(folded, original_size, pad_size, level=group["level"])
                    res = grad - unfold
                # State initialization
                if "exp_avg" not in state:
                    if "level" in group:
                        state["exp_avg"] = torch.zeros_like(folded)
                        state["exp_avg_sq"] = torch.zeros_like(folded)
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if "level" in group:
                    res_scale = group["res_scale"]
                    exp_avg.mul_(beta1).add_(folded, alpha=(1.0 - beta1))
                    update_m = unfold_tensor(exp_avg, original_size, pad_size, level=group["level"]).add_(res, alpha=res_scale)
                    exp_avg_sq.mul_(beta2).add_(folded_exp, alpha=(1.0 - beta2))
                    update_v = unfold_tensor(exp_avg_sq, original_size, pad_size, level=group["level"]).add_(res.pow(2), alpha=res_scale**2)

                    denom = update_v.sqrt().add_(group["eps"])
                    norm_grad = group["scale"] * update_m / denom

                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    norm_grad = exp_avg / denom

                step_size = group["lr"]
                
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
    
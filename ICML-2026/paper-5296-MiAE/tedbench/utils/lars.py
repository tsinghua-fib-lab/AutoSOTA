# Adapted from https://github.com/facebookresearch/mae/blob/main/util/lars.py
from typing import Callable

import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        weight_decay: weight decay (L2 penalty)
        momentum: momentum factor
        trust_coefficient: trust coefficient for computing the adaptive lr
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0,
        momentum: float = 0.9,
        trust_coefficient: float = 0.001,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient value: {trust_coefficient}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue

                dp = p.grad

                # Apply weight decay and trust coefficient only for parameters > 1D
                if p.ndim > 1:
                    # Add weight decay
                    if g["weight_decay"] != 0:
                        dp = dp.add(p, alpha=g["weight_decay"])

                    # Compute adaptive learning rate
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)

                    # Compute trust ratio
                    if param_norm > 0.0 and update_norm > 0.0:
                        trust_ratio = g["trust_coefficient"] * param_norm / update_norm
                        dp = dp.mul(trust_ratio)

                # Get or initialize momentum buffer
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mu = param_state["mu"]

                # Update momentum buffer: mu = momentum * mu + dp
                mu.mul_(g["momentum"]).add_(dp)

                # Update parameters: p = p - lr * mu
                p.add_(mu, alpha=-g["lr"])

        return loss

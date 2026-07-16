"""Baseline DP optimizers for comparison: DP-AdamBC and DiSK.

DP-AdamBC: Tang et al., "DP-AdamBC: Your DP-Adam Is Actually DP-SGD
(Unless You Apply Bias Correction)", AAAI 2024.

DiSK: Zhang et al., "DiSK: Differentially Private Optimizer with
Simplified Kalman Filter for Noise Reduction", ICLR 2025.
"""

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn


class DPAdamBC(torch.optim.Optimizer):
    """Adam with DP bias correction (DP-AdamBC).

    Corrects the upward bias in Adam's second moment estimate caused by
    DP noise injection. After standard Adam updates m and v, subtracts
    the known noise variance from the bias-corrected v_hat.

    Requires that ``clip_and_noise_gradients`` is called with
    ``store_summed_grad=True`` so that ``p.summed_grad`` is available.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        eps_root: float = 1e-4,
        noise_multiplier: float = 0.0,
        max_grad_norm: float = 1.0,
        batch_size: int = 256,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, eps_root=eps_root)
        super().__init__(params, defaults)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Known noise variance per coordinate: (sigma * C / B)^2
        sigma_sq = (
            self.noise_multiplier * self.max_grad_norm / self.batch_size
        ) ** 2

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            eps_root = group["eps_root"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad  # noised gradient

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]

                m, v = state["m"], state["v"]

                # Standard Adam moment updates using noised gradient
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad.square(), alpha=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Subtract known noise variance and clamp
                v_corr = (v_hat - sigma_sq).clamp(min=eps_root ** 2)

                # Parameter update
                p.data.addcdiv_(m_hat, v_corr.sqrt().add_(eps), value=-lr)

                # Clean up temporary attribute
                if hasattr(p, "summed_grad"):
                    del p.summed_grad

        return loss


class DiSKFilter:
    """Simplified Kalman filter for DP gradient denoising (DiSK).

    Maintains an exponential moving average of noised gradients per
    parameter. Call ``apply()`` after ``clip_and_noise_gradients()``
    and before ``optimizer.step()`` to replace ``p.grad`` with the
    filtered estimate.

    This is the simplified variant (Algorithm 3 from the paper) without
    the Hessian-vector product prediction step, which already captures
    the main denoising benefit while requiring no extra forward pass.
    """

    def __init__(self, kappa: float = 0.1):
        self.kappa = kappa
        self._state: Dict[int, torch.Tensor] = {}

    def apply(self, model: nn.Module) -> None:
        """Filter all parameter gradients in-place."""
        for p in model.parameters():
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                self._state[pid] = p.grad.clone()
            else:
                self._state[pid].mul_(1 - self.kappa).add_(
                    p.grad, alpha=self.kappa
                )
            p.grad = self._state[pid].clone()

    def reset(self) -> None:
        """Clear all filter state (call between training runs)."""
        self._state.clear()

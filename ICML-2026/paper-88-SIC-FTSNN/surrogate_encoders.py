# -*- coding: utf-8 -*-
"""
surrogate_encoders.py

A differentiable Poisson/Bernoulli spike encoder for SNN training.

Why this file?
- The vanilla Poisson encoder often samples spikes with a hard compare (e.g., rand <= p),
  which blocks gradients to upstream modules (e.g., learnable fragmentation).
- This module provides surrogate / reparameterized variants so gradients can flow.

Design notes:
- SpikingJelly-style "Poisson encoding" in discrete time is commonly implemented as
  sampling a spike at each time step with probability p in [0,1].
- We provide 4 modes:
    * "expected": return p (deterministic; no sampling)
    * "ste":      hard Bernoulli sample + straight-through gradient
    * "sigmoid":  soft threshold relaxation using sigmoid((p-u)/tau), optionally hard via STE
    * "concrete": Binary Concrete / Gumbel-Sigmoid reparameterization, optionally hard via STE

The encoder is stateless: call it once per time step to generate one spike sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


GradMode = Literal["expected", "ste", "sigmoid", "concrete"]
ProbFrom = Literal["clamp", "sigmoid", "raw"]


@dataclass
class EncoderDebug:
    """Optional: returned for debugging if you want to inspect internals."""
    p: torch.Tensor
    y_soft: Optional[torch.Tensor] = None
    y_hard: Optional[torch.Tensor] = None


class SurrogatePoissonEncoder(nn.Module):
    """
    Differentiable Poisson/Bernoulli spike encoder.

    Parameters
    ----------
    grad_mode:
        - "expected": return p (no sampling). Best for stable gradients, but not binary spikes.
        - "ste": hard Bernoulli sample; backward uses straight-through identity gradient.
        - "sigmoid": soft threshold relaxation with Uniform noise; optionally hard output via STE.
        - "concrete": Binary Concrete (Gumbel-Sigmoid / Logistic noise) relaxation; optionally hard via STE.

    temperature:
        Softness of the relaxation. Smaller -> harder spikes, but can make gradients noisier.

    hard:
        If True (recommended for SNN), forward outputs {0,1} spikes, but backward uses the soft sample
        (straight-through trick). If False, returns soft spikes in [0,1].

    prob_from:
        How to map input x to a probability p in (0,1):
        - "clamp": treat x as probability and clamp to [0,1]
        - "sigmoid": treat x as (unnormalized) logit/rate and map with sigmoid(x)
        - "raw": bypass the encoder and return x unchanged (useful for debugging / sanity checks)

        If your input is not guaranteed in [0,1] (e.g., after normalization / fragmentation),
        "sigmoid" is usually the safer choice to keep gradients.

    clip_prob:
        If True, clamp p into [eps, 1-eps] for numerical stability (especially important for "concrete").

    use_abs:
        If True, use abs(x) before mapping to probability. Useful if upstream can output negative values.

    eps:
        Numerical epsilon.
    """

    def __init__(
        self,
        *,
        grad_mode: GradMode = "concrete",
        temperature: float = 1.0,
        hard: bool = True,
        prob_from: ProbFrom = "sigmoid",
        clip_prob: bool = True,
        use_abs: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.grad_mode = grad_mode
        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.prob_from = prob_from
        self.clip_prob = bool(clip_prob)
        self.use_abs = bool(use_abs)
        self.eps = float(eps)

    def _to_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_abs:
            x = x.abs()

        if self.prob_from == "sigmoid":
            p = torch.sigmoid(x)
        elif self.prob_from == "clamp":
            p = x
        elif self.prob_from == "raw":
            # NOTE: raw bypass is handled in forward(); keep this for completeness.
            p = x
        else:
            raise ValueError(f"Unknown prob_from={self.prob_from}")

        if self.clip_prob:
            p = p.clamp(min=self.eps, max=1.0 - self.eps)
        else:
            # still avoid NaNs in logit for concrete
            p = p.clamp(min=0.0, max=1.0)
        return p

    @staticmethod
    def _straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        """
        Straight-through trick: forward uses hard, backward uses soft.
        Returns: hard - soft.detach() + soft
        """
        return hard - soft.detach() + soft

    def forward(self, x: torch.Tensor, *, return_debug: bool = False):
        """
        Forward produces ONE time-step of spikes given x.

        x: Tensor [...], recommended float.
        """
        # "raw": bypass encoder and return input unchanged.
        # This is deterministic and preserves full gradient to upstream modules.
        if self.prob_from == "raw":
            y = x.abs() if self.use_abs else x
            if return_debug:
                dbg = EncoderDebug(p=y, y_soft=y, y_hard=None)
                return y, dbg
            return y

        p = self._to_prob(x)

        debug = EncoderDebug(p=p)

        if self.grad_mode == "expected":
            y = p
            debug.y_soft = y

        elif self.grad_mode == "ste":
            # Hard sample; backward approximated as identity through p
            u = torch.rand_like(p)
            y_hard = (u < p).to(dtype=p.dtype)
            if self.hard:
                # Pass-through gradient as if y = p
                y = self._straight_through(y_hard, p)
            else:
                # If not hard, just return p (same as expected)
                y = p
            debug.y_hard = y_hard
            debug.y_soft = p

        elif self.grad_mode == "sigmoid":
            # Soft threshold: sigma((p-u)/tau)
            u = torch.rand_like(p)
            tau = max(self.temperature, self.eps)
            y_soft = torch.sigmoid((p - u) / tau)
            if self.hard:
                y_hard = (y_soft > 0.5).to(dtype=p.dtype)
                y = self._straight_through(y_hard, y_soft)
                debug.y_hard = y_hard
            else:
                y = y_soft
            debug.y_soft = y_soft

        elif self.grad_mode == "concrete":
            # Binary Concrete / Gumbel-Sigmoid:
            # y = sigmoid((logit(p) + logistic_noise) / tau)
            # logistic_noise = log(u) - log(1-u),  u~Uniform(0,1)
            u = torch.rand_like(p)
            # logistic noise
            logistic = torch.log(u.clamp(min=self.eps)) - torch.log((1.0 - u).clamp(min=self.eps))
            logit_p = torch.log(p) - torch.log(1.0 - p)
            tau = max(self.temperature, self.eps)
            y_soft = torch.sigmoid((logit_p + logistic) / tau)

            if self.hard:
                y_hard = (y_soft > 0.5).to(dtype=p.dtype)
                y = self._straight_through(y_hard, y_soft)
                debug.y_hard = y_hard
            else:
                y = y_soft
            debug.y_soft = y_soft

        else:
            raise ValueError(f"Unknown grad_mode={self.grad_mode}")

        if return_debug:
            return y, debug
        return y

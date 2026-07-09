# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.
#
# Modifications copyright (c) 2025 Huawei Noah's Ark Lab.
# This file contains code adapted from the Toto project and modified for use in Mantis.

import torch
from einops import rearrange


class RMSNorm(torch.nn.Module):
    """
    Wraps xFormers' rms_norm for eval/frozen mode, and does a Python fallback for train mode.
    """

    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        if include_weight:
            self.scale=torch.nn.Parameter(torch.ones(dim))
        else:
            self.scale = None

    def forward(self, x: torch.Tensor):
        x_normed = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # Scale the normalized input
        return x_normed if self.scale is None else x_normed * self.scale

    def increment_and_forward_(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: torch.Tensor, y: torch.Tensor

        If you need the fused addition with RMS norm, do the same check here.
        """
        if (not self.training) or (self.scale is not None and not self.scale.requires_grad):
            return rms_norm_add(x, y, self.scale, self.eps)

        # Fallback: x += y; then do RMS Norm
        return self.forward(x + y)

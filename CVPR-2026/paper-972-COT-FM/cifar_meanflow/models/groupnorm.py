"""
groupnorm.py

See https://github.com/pytorch/pytorch/issues/152185

This module provides an explicit implementation of Group Normalization for PyTorch models, 
motivated by the need to avoid using `torch.nn.functional.group_norm` (F.group_norm), 
which may fail during PyTorch compilation or tracing in certain environments. 
By defining both a custom `GroupNorm` module and a functional `group_norm` operation, 
this file ensures compatibility and flexibility for models that require group normalization 
without relying on PyTorch's internal functional API.

Motivation:
-----------
PyTorch's `F.group_norm` may not be compatible with certain compilation or export workflows 
(e.g., TorchScript, ONNX export, or custom backends), leading to runtime errors or 
unsupported operation exceptions. By implementing group normalization explicitly, 
this module provides a robust alternative that is less dependent on PyTorch's internal 
functional APIs and more amenable to compilation and export.

Usage:
------
- Replace instances of `nn.GroupNorm` or `F.group_norm` in your codebase with the 
  provided `GroupNorm` module or `group_norm` function to ensure compatibility 
  with compilation and export tools.
- The API and behavior closely follow the official PyTorch implementation, 
  including support for affine parameters and numerical stability via epsilon.

References:
-----------
- Group Normalization paper: https://arxiv.org/abs/1803.08494
- PyTorch GroupNorm documentation: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
"""

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn

import unittest
from torch.testing import assert_close


class GroupNorm(nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        return group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


def group_norm(input: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    N, C, *rest = input.shape
    assert C % num_groups == 0, "num_channels must be divisible by num_groups"

    # Reshape to (N, num_groups, C // num_groups, *spatial)
    reshaped = input.view(N, num_groups, C // num_groups, *rest)

    # Compute mean and variance over (C // G, *spatial)
    dim = tuple(range(2, reshaped.dim()))
    mean = reshaped.mean(dim=dim, keepdim=True)
    var = reshaped.var(dim=dim, keepdim=True, unbiased=False)

    # Normalize
    normalized = (reshaped - mean) / torch.sqrt(var + eps)
    normalized = normalized.view(N, C, *rest)

    # Apply affine if provided
    shape = [1, -1] + [1] * (normalized.dim() - 2)
    if weight is not None:
        normalized = normalized * weight.view(*shape)
    if bias is not None:
        normalized = normalized + bias.view(*shape)

    return normalized


# ---- Unit test ----
# Run: python -m models.groupnorm
class TestGroupNorm(unittest.TestCase):
    def _test_equivalence(self, shape, num_groups, dtype=torch.float32, eps=1e-5):
        torch.manual_seed(0)
        N, C, *rest = shape
        assert C % num_groups == 0

        x = torch.randn(shape, dtype=dtype, requires_grad=True)
        x2 = x.clone().detach().requires_grad_()

        base = nn.GroupNorm(num_groups, C, affine=True, eps=eps)
        custom = GroupNorm(num_groups, C, affine=True, eps=eps)

        # Match weights
        custom.weight.data.copy_(base.weight.data)
        custom.bias.data.copy_(base.bias.data)

        # Forward
        y1 = base(x)
        y2 = custom(x2)
        assert_close(y1, y2, rtol=1e-5, atol=1e-6)

        # Backward
        grad = torch.randn_like(y1)
        y1.backward(grad)
        y2.backward(grad)

        assert_close(x.grad, x2.grad, rtol=1e-5, atol=1e-6)
        assert_close(base.weight.grad, custom.weight.grad, rtol=1e-5, atol=1e-6)
        assert_close(base.bias.grad, custom.bias.grad, rtol=1e-5, atol=1e-6)

    def test_2d_input(self):
        self._test_equivalence((4, 8, 16, 16), num_groups=4)

    def test_1d_input(self):
        self._test_equivalence((2, 6, 32), num_groups=3)

    def test_3d_input(self):
        self._test_equivalence((2, 12, 8, 8, 8), num_groups=6)

if __name__ == '__main__':
    unittest.main()
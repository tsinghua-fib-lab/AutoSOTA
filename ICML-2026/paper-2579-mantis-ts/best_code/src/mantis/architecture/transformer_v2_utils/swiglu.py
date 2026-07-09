# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.
#
# Modifications copyright (c) 2025 Huawei Noah's Ark Lab.
# This file contains code adapted from the Toto project and modified for use in Mantis.

import torch
import torch.nn.functional as F


class SwiGLU(torch.nn.Module):
    """
    https://arxiv.org/abs/2002.05202
    NOTE: x should be 2x the size you want
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note this ordering is unusual, but is done so to match xFormers
        gate, x = x.chunk(2, dim=-1)
        return F.silu(gate) * x

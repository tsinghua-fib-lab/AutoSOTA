"""Layer handlers.

A handler describes, for one layer type, (a) how to flatten the layer's input
to ``[N, in_dim]`` for covariance accumulation, and (b) how the layer's stored
weight maps to/from the canonical ``[out, in]`` matrix used by the engram
formula. Generalizing this here keeps ``EngramEditor.compute_engram_weights``
free of per-type branches.

Bias absorption (homogeneous coordinates): when ``absorb_bias`` is set and the
layer has a bias, the input is augmented with a constant ``1`` column and the
weight with the bias column, so an affine layer ``y = Wx + b`` is treated as the
exact linear map ``y = [W | b] @ [x ; 1]``. The covariance then becomes
``(in+1) x (in+1)`` and the engram result's last column is the bias engram.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn


def get_conv1d_class() -> Optional[type]:
    """Return HuggingFace's ``Conv1D`` class if importable, else ``None``.

    GPT-2 / original-GPT family implement their projections with
    ``transformers.pytorch_utils.Conv1D`` (a transposed linear: ``weight`` is
    ``[in, out]`` and the forward is ``x @ weight``), not ``nn.Linear``. Hooking
    only ``nn.Linear`` would miss every block projection in those models.
    """
    try:
        from transformers.pytorch_utils import Conv1D

        return Conv1D
    except Exception:
        return None


def handler_for(
    registry: Dict[Type[nn.Module], "LayerHandler"], module: nn.Module
) -> Optional["LayerHandler"]:
    """Return the first registry handler whose type matches ``module``."""
    return next((h for cls, h in registry.items() if isinstance(module, cls)), None)


def _augment_ones(x: torch.Tensor) -> torch.Tensor:
    """Append a constant-1 column: ``[N, d] -> [N, d+1]`` (homogeneous coords)."""
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1)


class LayerHandler(ABC):
    """Abstract per-layer-type handler.

    ``absorb_bias`` is threaded through so a single handler covers both the
    plain (``y = Wx``) and bias-absorbed (``y = [W|b][x;1]``) cases.
    """

    @abstractmethod
    def get_input_dim(self, module: nn.Module, absorb_bias: bool = False) -> int:
        """Covariance dimension ``D`` (``in`` or ``in+1`` when absorbing)."""

    @abstractmethod
    def reshape_input(
        self, module: nn.Module, inputs: Any, absorb_bias: bool = False
    ) -> torch.Tensor:
        """Flatten the forward-pre-hook input to ``[N, D]`` (augmented if absorbing)."""

    @abstractmethod
    def weight_matrix(self, module: nn.Module, absorb_bias: bool = False) -> torch.Tensor:
        """Return the weight as canonical ``[out, D]`` (``[W | b]`` if absorbing)."""

    @abstractmethod
    def to_weight_shape(self, w: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Map a canonical ``[out, in]`` weight part back to ``module.weight``'s shape."""


class LinearHandler(LayerHandler):
    """Handler for ``nn.Linear`` (weight already stored as ``[out, in]``)."""

    def get_input_dim(self, module: nn.Linear, absorb_bias: bool = False) -> int:
        return module.in_features + (1 if absorb_bias else 0)

    def reshape_input(self, module: nn.Linear, inputs: Any, absorb_bias: bool = False) -> torch.Tensor:
        x = inputs[0].reshape(-1, module.in_features)
        return _augment_ones(x) if absorb_bias else x

    def weight_matrix(self, module: nn.Linear, absorb_bias: bool = False) -> torch.Tensor:
        if absorb_bias and module.bias is not None:
            return torch.cat([module.weight, module.bias.reshape(-1, 1)], dim=1)
        return module.weight

    def to_weight_shape(self, w: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        return w


class Conv1DHandler(LayerHandler):
    """Handler for HuggingFace ``Conv1D`` (GPT-2). Weight is stored ``[in, out]``.

    The effective linear operator is ``weight.T`` (shape ``[out, in]``), so the
    canonical matrix is ``weight.t()`` and the engram weight part is transposed
    back to ``[in, out]`` to match ``module.weight``.
    """

    def get_input_dim(self, module: nn.Module, absorb_bias: bool = False) -> int:
        return int(module.weight.shape[0]) + (1 if absorb_bias else 0)

    def reshape_input(self, module: nn.Module, inputs: Any, absorb_bias: bool = False) -> torch.Tensor:
        x = inputs[0].reshape(-1, int(module.weight.shape[0]))
        return _augment_ones(x) if absorb_bias else x

    def weight_matrix(self, module: nn.Module, absorb_bias: bool = False) -> torch.Tensor:
        w = module.weight.t()  # [out, in]
        if absorb_bias and getattr(module, "bias", None) is not None:
            return torch.cat([w, module.bias.reshape(-1, 1)], dim=1)
        return w

    def to_weight_shape(self, w: torch.Tensor, module: nn.Module) -> torch.Tensor:
        return w.t()

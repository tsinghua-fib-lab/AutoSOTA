"""Configuration for the engram editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class EditorConfig:
    """Configuration for engram extraction.

    Attributes:
        storage_device: Device for accumulating/holding covariance matrices.
            ``None`` (default) follows the model's device — fastest, with no
            per-batch GPU->CPU transfer. Set to ``"cpu"`` when the ``D x D``
            covariances do not fit in VRAM (large/wide models): collection is
            slower (each batch's ``D x D`` is copied to CPU) but feasible.
        absorb_bias: When ``True`` (default, *automatic*), layers that have a
            bias are treated as the affine map ``y = Wx + b`` via homogeneous
            coordinates: the input is augmented with a constant ``1`` and the
            weight with the bias column, so the engram also corrects ``b``.
            Bias-free layers (e.g. Llama/Mistral projections) are unaffected.
            Set ``False`` to edit ``W`` only (the original behavior).
    """

    storage_device: Optional[Union[str, torch.device]] = None
    absorb_bias: bool = True

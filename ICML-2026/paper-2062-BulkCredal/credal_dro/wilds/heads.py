from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class HeadConfig:
    """Configuration for an embedding head.

    kind:
      - "linear": single Linear layer
      - "mlp": 1-hidden-layer MLP with ReLU
    """
    kind: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 256          # used only if kind == "mlp"
    dropout: float = 0.0           # used only if kind == "mlp"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_head(input_dim: int, num_classes: int, cfg: HeadConfig):
    """Construct a torch.nn.Module head.

    Notes
    -----
    - We keep this tiny and dependency-free beyond torch.
    - No batchnorm: embeddings are assumed reasonably scaled already.
    """
    try:
        import torch
        from torch import nn
    except Exception as e:
        raise ImportError(
            "PyTorch is required for real-world embedding experiments. "
            "Install your real-world extras (e.g. torch/torchvision/wilds)."
        ) from e

    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if num_classes <= 1:
        raise ValueError(f"num_classes must be >=2 for classification, got {num_classes}.")

    if cfg.kind == "linear":
        return nn.Linear(input_dim, num_classes)

    if cfg.kind == "mlp":
        if cfg.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive for MLP, got {cfg.hidden_dim}.")
        if not (0.0 <= cfg.dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {cfg.dropout}.")

        return nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(cfg.hidden_dim, num_classes),
        )

    raise ValueError(f"Unknown HeadConfig.kind={cfg.kind!r}.")

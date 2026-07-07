"""Lightweight exponential moving average helper for model parameters."""

from __future__ import annotations

import logging
from typing import Dict, MutableMapping

import torch
from torch import nn


logger = logging.getLogger(__name__)


class EMA:
    """Track an exponential moving average of a model's trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")

        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()

    def update(self) -> None:
        """Update the moving averages with the model's current parameters."""

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if name not in self.shadow:
                    logger.warning("Skipping EMA update for unexpected parameter: %s", name)
                    self.shadow[name] = param.detach().clone()
                    continue
                if tuple(self.shadow[name].shape) != tuple(param.shape):
                    logger.warning(
                        "Resetting EMA shadow for shape-mismatched parameter %s: checkpoint=%s current=%s",
                        name,
                        tuple(self.shadow[name].shape),
                        tuple(param.shape),
                    )
                    self.shadow[name] = param.detach().clone()
                    continue

                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.detach().clone()

    def apply_shadow(self) -> None:
        """Swap the model weights with their EMA counterparts."""

        self.backup.clear()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if name not in self.shadow:
                    logger.warning("EMA shadow missing parameter %s; skipping swap", name)
                    continue

                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore the original (non-EMA) model weights."""

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if name not in self.backup:
                    logger.warning("EMA backup missing parameter %s; skipping restore", name)
                    continue

                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> MutableMapping[str, torch.Tensor]:
        """Return a copy of the EMA weights for checkpointing."""

        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: MutableMapping[str, torch.Tensor]) -> None:
        """Load EMA weights from a checkpoint state dictionary."""
        param_by_name = dict(self.model.named_parameters())
        loaded = {}
        for name, tensor in state_dict.items():
            tensor = tensor.detach()
            param = param_by_name.get(name)
            if param is not None:
                if tuple(tensor.shape) != tuple(param.shape):
                    logger.warning(
                        "Skipping EMA checkpoint tensor with shape mismatch: %s checkpoint=%s current=%s",
                        name,
                        tuple(tensor.shape),
                        tuple(param.shape),
                    )
                    continue
                tensor = tensor.to(device=param.device, dtype=param.dtype)
            loaded[name] = tensor.clone()
        current = self.shadow
        current.update(loaded)
        self.shadow = current

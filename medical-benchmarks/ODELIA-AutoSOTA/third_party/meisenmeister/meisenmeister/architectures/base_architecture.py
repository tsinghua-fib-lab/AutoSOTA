from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn


class BaseArchitecture(nn.Module, ABC):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Run a forward pass."""

    def get_init_kwargs(self) -> dict:
        return {}

    def get_grad_cam_target_layer(self) -> nn.Module:
        raise NotImplementedError(
            f"Grad-CAM is not available for architecture '{self.__class__.__name__}'"
        )

    def load_initial_weights(
        self,
        *,
        path: Path,
        device: torch.device,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weights file does not exist: {path}")

        payload = torch.load(path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise TypeError(
                f"Unsupported pretrained weights payload type: {type(payload).__name__}"
            )

        self.load_state_dict(state_dict)

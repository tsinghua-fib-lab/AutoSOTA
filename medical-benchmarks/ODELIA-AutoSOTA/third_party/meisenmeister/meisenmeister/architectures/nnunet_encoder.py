from __future__ import annotations

import re
from pathlib import Path

import torch
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_dropout,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from meisenmeister.architectures.base_architecture import BaseArchitecture


class _ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResidualEncoderClsNetwork(BaseArchitecture):
    ENCODER_INPUT_DIVISIBILITY = (16, 32, 32)

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        final_layer_dropout: float = 0.0,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder = ResidualEncoder(
            input_channels=in_channels,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[
                [1, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[
                [1, 1, 1],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            block=BasicBlockD,
            bottleneck_channels=None,
            return_skips=False,
            disable_default_stem=False,
            stem_channels=None,
            squeeze_excitation=False,
            squeeze_excitation_reduction_ratio=1.0 / 16,
        )
        self.cls_head = _ClassificationHead(320, num_classes)
        self.final_layer_dropout = get_matching_dropout(
            dimension=convert_conv_op_to_dim(nn.Conv3d)
        )(p=final_layer_dropout)
        self.apply(self.initialize)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.final_layer_dropout(encoded)
        return self.cls_head(encoded)

    def get_grad_cam_target_layer(self) -> nn.Module:
        return self.encoder.stages[-2]

    @staticmethod
    def initialize(module) -> None:
        InitWeights_He(1e-2)(module)

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
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "network_weights" in payload:
            state_dict = payload["network_weights"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise TypeError(
                f"Unsupported pretrained weights payload type: {type(payload).__name__}"
            )

        filtered_weights = self._filter_pretrained_encoder_weights(state_dict)
        if filtered_weights:
            self.load_state_dict(filtered_weights, strict=False)
            return

        self.load_state_dict(state_dict)

    def _filter_pretrained_encoder_weights(
        self, pretrained_state_dict: dict
    ) -> dict[str, torch.Tensor]:
        current_state_dict = self.state_dict()
        filtered_weights: dict[str, torch.Tensor] = {}

        for key, value in pretrained_state_dict.items():
            if not self._is_encoder_weight(key):
                continue
            mapped_key = self._map_pretrained_encoder_key(key)
            if mapped_key not in current_state_dict:
                continue
            if current_state_dict[mapped_key].shape != value.shape:
                continue
            filtered_weights[mapped_key] = value

        return filtered_weights

    @staticmethod
    def _is_encoder_weight(key: str) -> bool:
        patterns_to_keep = [
            r"^network\.encoder\.",
            r"^encoder\.",
        ]
        patterns_to_skip = [
            r"decoder",
            r"seg_layers",
            r"seg_head",
            r"output",
        ]
        for pattern in patterns_to_skip:
            if re.search(pattern, key, re.IGNORECASE):
                return False
        return any(re.search(pattern, key) for pattern in patterns_to_keep)

    @staticmethod
    def _map_pretrained_encoder_key(key: str) -> str:
        mapped_key = key.replace("network.encoder.", "encoder.")
        if mapped_key.startswith("model.encoder."):
            return mapped_key.replace("model.", "", 1)
        return mapped_key

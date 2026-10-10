from __future__ import annotations

from typing import Tuple

import torch
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn

from meisenmeister.architectures.base_architecture import BaseArchitecture


class _PrimusClassificationHead(nn.Module):
    def __init__(
        self, embed_dim: int, num_classes: int, classifier_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.mean(dim=1))
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc(x)


class PrimusMClsNetwork(BaseArchitecture):
    ENCODER_INPUT_DIVISIBILITY = (8, 8, 8)
    EMBED_DIM = 864
    EVA_DEPTH = 16
    EVA_NUM_HEADS = 12

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        patch_embed_size: Tuple[int, int, int] = ENCODER_INPUT_DIVISIBILITY,
        classifier_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        init_values: float = 0.1,
        scale_attn_inner: bool = True,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.input_shape = tuple(int(axis) for axis in input_shape)
        self.patch_embed_size = tuple(int(axis) for axis in patch_embed_size)
        self.classifier_dropout = float(classifier_dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.patch_drop_rate = float(patch_drop_rate)
        self.init_values = float(init_values)
        self.scale_attn_inner = bool(scale_attn_inner)
        if len(self.input_shape) != 3:
            raise ValueError(
                f"PrimusMClsNetwork expects a 3D input_shape, got {self.input_shape}"
            )
        if len(self.patch_embed_size) != 3:
            raise ValueError(
                "PrimusMClsNetwork expects a 3D patch_embed_size, "
                f"got {self.patch_embed_size}"
            )
        if any(axis <= 0 for axis in self.input_shape):
            raise ValueError(f"input_shape must be positive, got {self.input_shape}")
        if any(axis <= 0 for axis in self.patch_embed_size):
            raise ValueError(
                f"patch_embed_size must be positive, got {self.patch_embed_size}"
            )
        if any(
            axis % patch != 0
            for axis, patch in zip(self.input_shape, self.patch_embed_size, strict=True)
        ):
            raise ValueError(
                "PrimusMClsNetwork requires input_shape divisible by "
                f"{list(self.patch_embed_size)}, got {list(self.input_shape)}"
            )

        self.down_projection = PatchEmbed(
            self.patch_embed_size,
            in_channels,
            self.EMBED_DIM,
        )
        self.eva = Eva(
            embed_dim=self.EMBED_DIM,
            depth=self.EVA_DEPTH,
            num_heads=self.EVA_NUM_HEADS,
            ref_feat_shape=tuple(
                axis // patch
                for axis, patch in zip(
                    self.input_shape, self.patch_embed_size, strict=True
                )
            ),
            num_reg_tokens=0,
            use_rot_pos_emb=True,
            use_abs_pos_emb=True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            rope_impl=RotaryEmbeddingCat,
            rope_kwargs=None,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
        self.cls_head = _PrimusClassificationHead(
            self.EMBED_DIM,
            num_classes,
            classifier_dropout=self.classifier_dropout,
        )
        self.down_projection.apply(InitWeights_He(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = tuple(int(axis) for axis in x.shape[2:])
        if spatial_shape != self.input_shape:
            raise ValueError(
                f"PrimusMClsNetwork expects input shape {self.input_shape}, "
                f"got {spatial_shape}"
            )
        x = self.down_projection(x)
        x = rearrange(x, "b c w h d -> b (w h d) c")
        x, _ = self.eva(x)
        return self.cls_head(x)

    def get_init_kwargs(self) -> dict:
        return {
            "input_shape": self.input_shape,
            "patch_embed_size": self.patch_embed_size,
            "classifier_dropout": self.classifier_dropout,
            "drop_path_rate": self.drop_path_rate,
            "patch_drop_rate": self.patch_drop_rate,
            "init_values": self.init_values,
            "scale_attn_inner": self.scale_attn_inner,
        }

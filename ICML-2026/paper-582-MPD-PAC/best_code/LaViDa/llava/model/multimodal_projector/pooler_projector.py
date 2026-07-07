import torch
import torch.nn as nn

import math

from transformers.models.clip.modeling_clip import CLIPVisionModel


class PoolerProjector(nn.Module):
    def __init__(self, config, vision_cfg,pooler_ratio=2):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.ratio = pooler_ratio

        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=self.ratio, stride=self.ratio)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "pooler"}

import torch
import torch.nn.functional as F
from torch import nn
from typing import List
from .clip import CLIPVisionTransformer
from .reins import LoRAReins
from .peft import set_requires_grad, set_train, get_pyramid_feature


class ReinsCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(
        self,
        reins_config=None,
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11],
        pretrained=None,
        get_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            input_resolution,
            patch_size,
            width,
            layers,
            heads,
            output_dim,
            drop_path_rate,
            out_indices,
            pretrained,
            get_embeddings,
            **kwargs,
        )
        self.reins = LoRAReins(**reins_config)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:,]
            .reshape(1, self.spatial_size, self.spatial_size, C)
            .permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            x = self.reins.forward(x, i, batch_first=False, has_cls_token=True)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())
        return get_pyramid_feature(features)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])


def get_std_reins_clip_large(checkpoint_file=None):
    reins_config = dict(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        lora_dim=16,
    )
    return ReinsCLIPVisionTransformer(
        reins_config=reins_config,
        patch_size=16,
        width=1024,
        output_dim=512,
        get_embeddings=False,
        drop_path_rate=0.1,
        layers=24,
        input_resolution=512,
        style="pytorch",
        out_indices=[7, 11, 15, 23],
        heads=16,
        pretrained=checkpoint_file,
    )

from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from .clip import CLIPVisionTransformer
from .utils import set_requires_grad, set_train
import torch
import torch.nn.functional as F


@BACKBONES.register_module()
class ReinsCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)

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
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = (
                x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
            )  # B C H W

            features.append([global_embedding, visual_embedding])
        # features[0] = F.interpolate(
        #     features[0], scale_factor=4, mode="bilinear", align_corners=False
        # )
        # features[1] = F.interpolate(
        #     features[1], scale_factor=2, mode="bilinear", align_corners=False
        # )
        # features[3] = F.interpolate(
        #     features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        # )
        return self.reins.return_auto(tuple(features))

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["fpn", "reins"])
        set_train(self, ["fpn", "reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if ("rein" not in k) and ('fpn' not in k)]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state

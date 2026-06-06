from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from .sam_vit import SAMViT
from .utils import set_requires_grad, set_train
import torch
import torch.nn.functional as F


@BACKBONES.register_module()
class ReinsSAMViT(SAMViT):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rein_enabled_layers: list = kwargs.get("global_attn_indexes")
        self.reins: Reins = MODELS.build(reins_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if self.pos_embed is not None:
            x = x + self.pos_embed
        features = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            if idx in self.rein_enabled_layers:
                x = self.reins.forward(
                    x.view(B, -1, C),
                    self.rein_enabled_layers.index(idx),
                    batch_first=True,
                    has_cls_token=False,
                ).view(B, H, W, C)
            # 4,32,32,768
            if idx in self.out_indices:
                features.append(x.permute(0, 3, 1, 2))
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        return self.reins.return_auto(tuple(features))

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state

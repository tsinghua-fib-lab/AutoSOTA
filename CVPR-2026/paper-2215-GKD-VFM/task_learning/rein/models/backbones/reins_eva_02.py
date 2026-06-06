from .eva_02 import EVA2
from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class ReinsEVA2(EVA2):
    def __init__(self, reins_config=None, **kwargs):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        return self.reins.return_auto(features)

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

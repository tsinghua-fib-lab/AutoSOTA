from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from mmseg.models.backbones import ResNetV1c
from .utils import set_requires_grad, set_train
from typing import List, Dict
import torch.nn as nn


@BACKBONES.register_module()
class ReinsResNetV1c(ResNetV1c):
    def __init__(
        self,
        distinct_cfgs: List[Dict] = None,
        reins_config: Dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: List[Reins] = nn.ModuleList()
        for cfgs in distinct_cfgs:
            reins_config.update(cfgs)
            self.reins.append(MODELS.build(reins_config))

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            for idx_sublayer, sublayer in enumerate(res_layer):
                x = sublayer(x)
                B, C, H, W = x.shape
                x = (
                    self.reins[i]
                    .forward(
                        x.flatten(-2, -1).permute(0, 2, 1),
                        idx_sublayer,
                        batch_first=True,
                        has_cls_token=False,
                    )
                    .permute(0, 2, 1)
                    .reshape(B, C, H, W)
                )
            if i in self.out_indices:
                outs.append(self.reins[i].return_auto(x))
        return [f1 for f1, _ in outs], sum([f2 for _, f2 in outs])

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

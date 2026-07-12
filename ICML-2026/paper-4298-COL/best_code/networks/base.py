import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import clip


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.backbone == 'vitB16':  # no bn, relu, maxpool after last convolution
            b, _ = clip.load("ViT-B/16", device="cuda", jit=False)
            b = b.visual
            backbone = b.to(torch.float)
            self.encoder = backbone
        else:
            raise ValueError(f'Not supported backbone architecture {cfg.backbone}')

    def forward(self, x_base, x_ref=None):
        # feature extraction
        base_embs = self.encoder(x_base)
        if x_ref is not None:
            ref_embs = self.encoder(x_ref)
            out = self._forward(base_embs, ref_embs)
            return out, base_embs, ref_embs
        else:
            out = self._forward(base_embs)
            return out

    def _forward(self, base_embs, ref_embs=None):
        raise NotImplementedError('Suppose to be implemented by subclass')

import torch
import torch.nn as nn
from timm.models import create_model

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
import timm.models.vision_transformer

backbone = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)

class TinyViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=16,
        emb_dim=192,
        depth=12,
        num_heads=3,
        use_pretrained=False,
        use_global_pool=False,
        vit_name='vit_tiny_patch16_224'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_pretrained = use_pretrained

        
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from dinov3.layers import SelfAttentionBlock, SwiGLUFFN
from dinov3.models.vision_transformer import init_weights_vit
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")


class VisionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        blocks_drop_path: float,
        use_class_token: bool,
        use_patch_tokens: bool,
        use_linear_projection: bool,
    ):
        super().__init__()
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        if num_blocks > 0:
            block_list = [
                SelfAttentionBlock(
                    input_dim,
                    num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=blocks_drop_path,
                )
                for _ in range(num_blocks)
            ]
            self.ln_final = nn.LayerNorm(input_dim)
        self.blocks = nn.ModuleList(block_list)
        self.num_blocks = num_blocks

        multiplier = 2 if use_class_token and use_patch_tokens else 1
        self.linear_projection = nn.Identity()
        if multiplier * input_dim != embed_dim or use_linear_projection:
            logger.info(
                f"Vision Tower: Using a linear projection from {input_dim} to {embed_dim}"
            )
            assert embed_dim % multiplier == 0, (
                f"Expects {embed_dim} to be divisible by {multiplier}"
            )
            self.linear_projection = nn.Linear(
                input_dim, embed_dim // multiplier, bias=False
            )
        
        self.patch_size = 21


    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block = self.blocks[i]
                named_apply(init_weights_vit, block)
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(
                self.linear_projection.weight,
                std=self.linear_projection.in_features**-0.5,
            )

    def forward(self, image_tokens, dino_feat) -> torch.Tensor:

        for block in self.blocks:
            image_tokens = image_tokens + block.ls1(self.proxy_attn(block.attn, block.norm1(image_tokens), dino_feat))
            image_tokens =  image_tokens + block.ls2(block.mlp(block.norm2(image_tokens)))

        image_tokens = self.ln_final(image_tokens)

        return self.linear_projection(image_tokens)

    def proxy_attn(self, attn_layer, x, ex_feats):

        num_heads = attn_layer.num_heads
        bsz, _, embed_dim = x.size()
        head_dim = embed_dim // num_heads

        global_tokens = x[:,:5].clone()
        x = x[:,5:]
        
        B, N, C = x.shape
        qkv = attn_layer.qkv(x).reshape(B, N, 3, num_heads, C // num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        q = q.squeeze(0).contiguous()
        k = k.squeeze(0).contiguous()
        v = v.squeeze(0).contiguous()


        # proxy feat attn
        ex_feats = ex_feats.permute(0,2,1)
        q_k = F.normalize(ex_feats, dim=1)
        similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)
        similarity = (similarity - torch.mean(similarity) * 1.5) * 0.5

        # DINO mask attention
        similarity[similarity <= 0.0] = float('-inf')

        mask = similarity.to(q.dtype).unsqueeze(1).repeat(1, num_heads, 1, 1)
        mask = mask.reshape(bsz * num_heads, mask.shape[2], mask.shape[3])
        attn_weights = F.softmax(mask, dim=-1)

        v = v.reshape(bsz*num_heads, self.patch_size, self.patch_size, head_dim).permute(0, 3, 1, 2)
        v = F.interpolate(v, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
        v = v.permute(0, 2, 3, 1).reshape(bsz*num_heads, self.patch_size*self.patch_size, head_dim)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, -1, embed_dim)
        attn_output = attn_layer.proj_drop(attn_layer.proj(attn_output))
        return torch.cat((global_tokens, attn_output), dim=1)



class VisionTower(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        embed_dim: int,
        num_head_blocks: int,
        head_blocks_block_drop_path: float,
        use_class_token: bool,
        use_patch_tokens: bool,
        patch_token_layer: int,
        patch_tokens_pooler_type: str,
        use_linear_projection: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_class_token = use_class_token
        self.use_patch_tokens = use_patch_tokens
        self.patch_token_layer = patch_token_layer
        self.patch_tokens_pooler_type = patch_tokens_pooler_type
        self.num_register_tokens = 0
        if hasattr(self.backbone, "num_register_tokens"):
            self.num_register_tokens = self.backbone.num_register_tokens
        elif hasattr(self.backbone, "n_storage_tokens"):
            self.num_register_tokens = self.backbone.n_storage_tokens
        backbone_out_dim = self.backbone.embed_dim
        logger.info(f"Visual backbone embedding dimension: {backbone_out_dim}")
        self.head = VisionHead(
            backbone_out_dim,
            embed_dim,
            self.backbone.num_heads,
            num_head_blocks,
            head_blocks_block_drop_path,
            use_class_token,
            use_patch_tokens,
            use_linear_projection,
        )

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()

    def get_backbone_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, _ = self.backbone.get_intermediate_layers(
            images,
            n=self.patch_token_layer,
            return_class_token=True,
            return_extra_tokens=True,
        )
        class_token = tokens[-1][1]
        patch_tokens = tokens[-1][0]
        register_tokens = tokens[-1][2]

        return class_token, patch_tokens, register_tokens

    def get_class_and_patch_tokens(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        class_token, patch_tokens, register_tokens = self.get_backbone_features(images)

        image_tokens = self.head(
            torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1), patch_tokens
        )
        return (
            image_tokens[:, 0],
            image_tokens[:, self.num_register_tokens + 1 :],
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_token, patch_tokens = self.get_class_and_patch_tokens(images)
        features = []
        if self.use_class_token:
            features.append(class_token)
        if self.use_patch_tokens:
            if self.patch_tokens_pooler_type == "mean":
                features.append(torch.mean(patch_tokens, dim=1))
            elif self.patch_tokens_pooler_type == "max":
                features.append(torch.max(patch_tokens, dim=1).values)
            else:
                raise ValueError(
                    f"Unknown patch tokens pooler type: {self.patch_tokens_pooler_type}"
                )
        return torch.cat(features, dim=-1), patch_tokens


def build_vision_model(
    embed_dim: int,
    backbone_model_config: str,
    freeze_backbone: bool,
    num_head_blocks: int,
    blocks_drop_path: float,
    use_class_token: bool,
    use_patch_tokens: bool,
    patch_token_layer: int,
    patch_tokens_pooler_type: str,
    use_linear_projection: bool,
    backbone: Optional[nn.Module] = None,
):
    if backbone is None:
        if backbone_model_config is not None:
            from omegaconf import OmegaConf

            from dinov3.models import build_model_from_cfg as build_vision_backbone

            cfg = OmegaConf.load(backbone_model_config)
            backbone, _ = build_vision_backbone(cfg, only_teacher=True)
        else:
            raise RuntimeError(
                "Failed to create, vision backbone, either backbone or backbone_model_config should be not None"
            )
    return VisionTower(
        backbone,
        freeze_backbone,
        embed_dim,
        num_head_blocks,
        blocks_drop_path,
        use_class_token,
        use_patch_tokens,
        patch_token_layer,
        patch_tokens_pooler_type,
        use_linear_projection,
    )

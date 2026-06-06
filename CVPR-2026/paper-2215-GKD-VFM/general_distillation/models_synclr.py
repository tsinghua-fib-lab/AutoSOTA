# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple
from timm.models.vision_transformer import Attention
from timm.models.layers import Mlp, DropPath
from timm.models.helpers import named_apply



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ffn_targets=False,
                 return_layer_targets=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # specify the targets for feature regression
        self.ffn_targets = ffn_targets
        self.return_layer_targets = return_layer_targets

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        x = x + self.drop_path(self.attn(self.norm1(x)))
        ffn_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(ffn_out)

        target = ffn_out if self.ffn_targets else x

        if self.return_layer_targets:
            return x, target
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ffn_targets=False, return_layer_targets=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            ffn_targets (bool): whether we use ffn output or block end as the feature targets
            return_layer_targets (bool): whether we return every layer targets
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.ffn_targets = ffn_targets
        self.return_layer_targets = return_layer_targets
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_targets=ffn_targets, return_layer_targets=return_layer_targets,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)



def compute_gather_ids(masks):
    unmask_indices = masks.logical_not().nonzero(as_tuple=False)
    ids_keep = unmask_indices[:, -1].reshape(masks.shape[0], -1)
    return ids_keep


class MaskedTransformer(VisionTransformer):
    """Inherit vision transformer from timm"""

    def __init__(self, mask_style='ibot', **kwargs):
        super().__init__(**kwargs)
        assert mask_style in ["ibot", "mae", "none"], "mask_style must be `ibot`, `mae`, or `none`"

        self.patch_size = self.patch_embed.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        nn.init.normal_(self.cls_token, std=1e-6)

        self.mask_style = mask_style
        if self.mask_style == "ibot":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            torch.nn.init.normal_(self.mask_token, std=.02)

    def interpolate_pos_encoding(self, x, w, h, npatch):
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        Args:
            x: data w/ shape [b, c, h, w]
            masks: shape [b, n], n is the number of tokens, 1 means masked, 0 means unmasked
        """
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            if self.mask_style == 'ibot':
                x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            elif self.mask_style == 'mae':  # only gather unmasked patches
                # add pos_embed before shuffle
                pos_embed = self.interpolate_pos_encoding(x, w, h, npatch=x.shape[1])
                x = x + pos_embed[:, 1:, :]
                ids_keep = compute_gather_ids(masks)
                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
                # x = x[masks.logical_not()]
                # x = x.reshape(b, -1, x.size(-1))
            else:
                raise NotImplementedError(f"mask style {self.mask_style} is not supported")

        if (masks is None) or (self.mask_style != "mae"):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h, npatch=x.shape[1]-1)
        else:
            # mae-style masking, only need to add cls tokens w/ pos embedding
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        num_data = len(x)
        if self.return_layer_targets:
            all_layer_results = [[] for _ in range(num_data)]
            for i, blk in enumerate(self.blocks):
                out = [blk(t) for t in x]
                x = [o[0] for o in out]
                # store layer targets
                for j in range(num_data):
                    all_layer_results[j].append(out[j][1])
            all_x = x
        else:
            all_x = [self.blocks(t) for t in x]
            all_layer_results = [None for _ in range(num_data)]

        output = []
        for x, masks, layer_results in zip(all_x, masks_list, all_layer_results):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm": x_norm,
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "masks": masks,
                    "layer_results": layer_results,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        if self.return_layer_targets:
            layer_results = []
            for i, blk in enumerate(self.blocks):
                x, lr = blk(x)
                layer_results.append(lr)
        else:
            x = self.blocks(x)
            layer_results = None

        x_norm = self.norm(x)
        return {
            "x_norm": x_norm,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "masks": masks,
            "layer_results": layer_results,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return ret["x_norm_clstoken"]


def vit_small(patch_size=16, teacher_path=None, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kwargs)

    if teacher_path is not None:
        checkpoint = torch.load(teacher_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        pretrained_dict = {k.replace("module.visual.", ""): v for k, v in pretrained_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, False)
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)
    
    return model


def vit_base(patch_size=16, teacher_path=None, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kwargs)

    if teacher_path is not None:
        checkpoint = torch.load(teacher_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        pretrained_dict = {k.replace("module.visual.", ""): v for k, v in pretrained_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, False)
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)

    return model


def vit_large(patch_size=14, teacher_path=None, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    
    if teacher_path is not None:
        checkpoint = torch.load(teacher_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        pretrained_dict = {k.replace("module.visual.", ""): v for k, v in pretrained_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, False)
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)

    return model


if __name__ == '__main__':
    import argparse
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    parser = argparse.ArgumentParser(description='PyTorch resnet Training')
    args = parser.parse_args()

    with torch.no_grad():
        model = vit_base(patch_size=14, num_classes=0, mask_style='ibot')
        
        # x = torch.randn(1, 3, 224, 224)
        # out = model(x)
        # print(out.shape)

        print(parameter_count_table(model))

        tensor = torch.rand(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, tensor)
        print("FLOPs: ", flops.total()/1e9)
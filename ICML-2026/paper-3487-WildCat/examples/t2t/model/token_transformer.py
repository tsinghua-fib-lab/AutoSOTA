# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""
import torch
import torch.nn as nn

from torchvision.ops import StochasticDepth

from einops import rearrange


# from src.models.modules.seq_common import Mlp

from itertools import repeat
import collections.abc

# Copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 act_fn=None, drop=0., device=None, dtype=None):
        """TD [2021-10-27] act_fn takes precedence over act_layer if set.
        This is to support Pytorch 1.10 Transformer interface that construct the activation
        *function*, not the activation *layer*.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.act = act_layer() if act_fn is None else act_fn
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class T2TAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_cfg=None):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim if in_dim is not None else dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if attn_cfg is None:
            self.attention_layer = None
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            # try:
            if attn_cfg['name'] == 'full':
                from model.attention.full_attention import FullAttention
                attn_cfg['softmax_temp'] = self.scale
                print(f"full all params: {attn_cfg}")
                self.attention_layer = FullAttention(**attn_cfg)

            elif attn_cfg['name'] == 'performer':
                from model.attention.performer import PerformerAttention
                attn_cfg['softmax_temp'] = self.scale
                print(f"performer all params: {attn_cfg}")
                self.attention_layer = PerformerAttention(**attn_cfg)

            elif attn_cfg['name'] == 'reformer':
                from model.attention.reformer import ReformerAttention
                attn_cfg['softmax_temp'] = self.scale
                print(f"reformer all params: {attn_cfg}")
                self.attention_layer = ReformerAttention(**attn_cfg)

            elif attn_cfg['name'] == 'kdeformer':
                from model.attention.kdeformer import RobustAttention
                # sample_size = attn_cfg['sample_size']
                # bucket_size = attn_cfg['bucket_size']
                attn_cfg['dim'] = 64
                attn_cfg['num_projs'] = 7
                attn_cfg['softmax_temp'] = self.scale
                print(f"kdeformer all params: {attn_cfg}")
                self.attention_layer = RobustAttention(**attn_cfg)

            elif attn_cfg['name'] == 'thinformer':
                from thinformer import ThinformerAttention
                attn_cfg['scale'] = self.scale
                print(f"thinformer all params: {attn_cfg}")
                self.attention_layer = ThinformerAttention(**attn_cfg)

            elif attn_cfg['name'] == 'scatterbrain':
                from model.attention.scatterbrain import SBLocalAttention
                attn_cfg['softmax_temp'] = self.scale
                print(f"scatterbrain all params: {attn_cfg}")
                self.attention_layer = SBLocalAttention(**attn_cfg)

            elif attn_cfg['name'] == 'wildcat':
                from wildcat import WildCat
                attn_cfg['scale'] = self.scale
                print(f"{attn_cfg['name']} all params: {attn_cfg}")
                self.attention_layer = WildCat(**attn_cfg)

            else:
                raise ValueError(f"Invalid attention method: {attn_cfg['name']}")

            # except Exception as e:
            #     print(f"Error: {e}")
            #     import pdb; pdb.set_trace()


    def forward(self, x):
        B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (B, N, D)

        v_og = v
        q, k, v = [rearrange(x, 'b n (n_head head_dim) -> b n n_head head_dim',
                             n_head=self.num_heads) for x in (q, k, v)]

        if self.attention_layer is None:  # Full attention
            q, k, v = [rearrange(x, 'b n n_head head_dim -> b n_head n head_dim') for x in (q, k, v)]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_output = (attn @ v).transpose(1, 2)
        elif self.attention_layer.__class__.__name__ == "WildCat":
            attn_output = self.attention_layer(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
            attn_output = attn_output.transpose(1,2)
        else:
            attn_output, _ = self.attention_layer(q, k, v)
        x = rearrange(attn_output, 'b n h d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        # because the original x has different size with current x, use v to do skip connection
        x = v_og + x
        return x


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = T2TAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            attn_cfg=attn_cfg,
        )
        self.drop_path = StochasticDepth(drop_path, mode='row')
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio),
                       out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from modeling.backbones.token_transformer import Token_transformer
from modeling.backbones.token_performer import Token_performer
from modeling.backbones.transformer_block import Block, get_sinusoid_encoding
import math


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution

        self.num_patches = (img_size[0] // (4 * 2 * 2)) * (
                img_size[1] // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, 64, 32)
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, 32, 16)
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class T2T_ViT(nn.Module):
    def __init__(self, img_size=(256, 128), tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., camera=0,
                 view=0, sie_xishu=3.0, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, camera_id=cam_label, view_id=view_label)
        cash_x = []
        cash_x.append(x)
        return cash_x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'state_dict_ema' in param_dict:
            param_dict = param_dict['state_dict_ema']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old resnest that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, 16, 8)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


@register_model
def t2t_vit_7(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_12(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_19(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_24(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
               attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_14(num_classes=1051, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                 attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(num_classes=num_classes, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_24(num_classes=1051, img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0,
                 attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(num_classes=num_classes, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu,
                    tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# model = t2t_vit_t_24()
# data = torch.randn(2, 3, 256, 128)
# model.load_param('/15127306268/wyh/UIS/pth/82.6_T2T_ViTt_24_pth.tar')
# model(data)

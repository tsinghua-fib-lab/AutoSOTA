from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from block import BlockRope
from attention import FlashAttentionRope
from block import BlockRope, CrossBlockRope
from pos_embed import RoPE2D, PositionGetter, DirectionGetter


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, attn_mask_cross: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_cross = nn.MultiheadAttention(d_model, n_head)
        self.ln_text = LayerNorm(d_model)
        self.ln_visual = LayerNorm(d_model)
        self.attn_mask_cross = attn_mask_cross

    def attention(self, x: torch.Tensor):

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cross_attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask_cross = self.attn_mask_cross.to(dtype=x.dtype, device=x.device) if self.attn_mask_cross is not None else None
        return self.attn_cross(x, y, y, need_weights=False, attn_mask=self.attn_mask_cross)[0]
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.cross_attention(self.ln_visual(x), self.ln_text(y))
        # IDEA-006: Add reverse cross-attention (text attends to image for refinement)
        y_refined = y + self.cross_attention(self.ln_text(y), self.ln_visual(x))
        # Fuse bidirectional features
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ScaleMapModel(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 text_channels: int, 
                 features: int, 
                 cross_attn_nhead: int, 
                 coef_scale: float, 
                 coef_shift: float,
                 pos_type:str = 'rope100'):
        super(ScaleMapModel, self).__init__()

        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 2
            
        if in_channels != text_channels:
            self.align_text_image = True
            self.align_mlp = nn.Sequential(OrderedDict([
                ("a_fc", nn.Linear(text_channels, in_channels)),
                ("gelu", QuickGELU())
            ]))
        else:
            self.align_text_image = False

        self.eps = 1e-4
        self.coef_scale = coef_scale
        self.coef_shift = coef_shift

        self.crossattn = CrossAttentionBlock(in_channels, cross_attn_nhead)

        self.projects = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        
        self.resize_layers = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0)
        
        self.layer1_rn = nn.Conv2d(
        out_channels, features, kernel_size=3, stride=1, padding=1, bias=False, groups=1
        )

        self.resConfUnit1 = ResidualConvUnit(features, nn.ReLU(False), False)
        self.out_conv1 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit2 = ResidualConvUnit(features, nn.ReLU(False), False)
        self.out_conv2 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        head_features_1 = features
        head_features_2 = 32

        self.output_conv_scale1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv_scale2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(True),
                # nn.Identity(),
            )

        self.output_conv_shift1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv_shift2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(True),
                # nn.Identity(),
            )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_features, text_features, patch_h, patch_w):
        if self.align_text_image:
             text_features = self.align_mlp(text_features)

        out_features = self.crossattn(img_features.permute(1, 0, 2), text_features.permute(1, 0, 2))
        x = out_features.permute(1, 0, 2) #B, patch_w * patch_c, C
        
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

        x = self.projects(x)
        x = self.resize_layers(x)
        x = self.layer1_rn(x)

        x = self.resConfUnit1(x)
        x = self.out_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.resConfUnit2(x)
        x = self.out_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        scale_pred_mid = self.output_conv_scale1(x)
        scale_pred = F.interpolate(scale_pred_mid, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        scale_pred = self.coef_scale * torch.exp(self.output_conv_scale2(scale_pred))

        shift_pred_mid = self.output_conv_shift1(x)
        shift_pred = F.interpolate(shift_pred_mid, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        shift_pred = self.coef_shift * torch.exp(self.output_conv_shift2(shift_pred))
        # shift_pred = self.coef_shift * F.leaky_relu(self.output_conv_shift2(shift_pred))

        return scale_pred, shift_pred, scale_pred_mid, shift_pred_mid
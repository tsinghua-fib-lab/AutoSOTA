"""UNet-style 3D decoder for MASS/Iris.

The decoder upsamples fused multi-scale features and produces segmentation
logits for the classes represented by the provided task embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from ..components.conv_blocks import BasicBlock

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.up_scale = up_scale

        block_list = []
        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        
        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        input_dtype = x1.dtype
        # Trilinear interpolation does not support bfloat16 on all PyTorch
        # builds, so upsample in FP32 and cast back for AMP training.
        x1 = F.interpolate(x1.float(), size=x2.shape[2:], mode='trilinear', align_corners=True)
        x1 = x1.to(input_dtype)
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out

class UNet_Decoder(nn.Module):
    """3D UNet decoder with optional prior fusion at the deepest scales."""

    def __init__(
        self, 
        in_ch, 
        base_ch=None,
        block='BasicBlock', 
        scale=[2,2,2,2], 
        num_block=[2,2,2,2], 
        norm='in', 
        kernel_size=[3,3,3,3], 
        num_prior_stage=1,
        channels: Optional[List[int]] = None
    ):
        super().__init__()
        from ..utils import get_block, get_norm
        from .fusion import PriorFusionLayer, MultiPriorFusionLayer, HierarchyPriorClassifier
        
        block = get_block(block)
        norm = get_norm(norm)
        self.num_prior_stage = num_prior_stage
        
        if channels is not None:
            assert len(channels) == 5, f"Expected 5 channel values, got {len(channels)}"
            ch = channels
        else:
            assert base_ch is not None, "Either channels or base_ch must be provided"
            ch = [base_ch, 2*base_ch, 4*base_ch, 8*base_ch, 16*base_ch]
        
        self.channels = ch
        
        self.prior_fuse = nn.ModuleList()
        for i in range(num_prior_stage):
            stage_idx = 4 - i  # maps i=0->4, i=1->3, i=2->2
            self.prior_fuse.append(MultiPriorFusionLayer(ch[stage_idx], ch[stage_idx], block_num=2))

        self.up1 = up_block(ch[4], ch[3], num_block=num_block[3], block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(ch[3], ch[2], num_block=num_block[2], block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(ch[2], ch[1], num_block=num_block[1], block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(ch[1], ch[0], num_block=num_block[0], block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        
    def forward(self, feat_list, prior_list=None):
        x5, x4, x3, x2, x1 = feat_list
        
        posterior_list = []
        
        if prior_list is not None and self.num_prior_stage >= 1:
            # Fuse priors before each corresponding upsampling stage.
            out, posterior5 = self.prior_fuse[0](x5, prior_list[0])
            # Token 0 is used as the stage summary for dynamic classification.
            posterior_list.append(posterior5[:, :, 0, :])
        else:
            out = x5
        
        out = self.up1(out, x4)
        
        if prior_list is not None and self.num_prior_stage >= 2:
            out, posterior4 = self.prior_fuse[1](out, prior_list[1])
            posterior_list.append(posterior4[:, :, 0, :])
        
        out = self.up2(out, x3)
        
        if prior_list is not None and self.num_prior_stage >= 3:
            out, posterior3 = self.prior_fuse[2](out, prior_list[2])
            posterior_list.append(posterior3[:, :, 0, :])
        
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        
        return out, posterior_list

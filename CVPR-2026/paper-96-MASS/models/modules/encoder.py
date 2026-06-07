"""UNet-style 3D image encoder for MASS/Iris.

The encoder converts target or reference volumes into multi-scale feature maps
that are later fused with task priors and decoded into segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from ..components.conv_blocks import BasicBlock


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        block_list = []

        if pool:
            block_list.append(nn.MaxPool3d(down_scale))
            block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))
        else:
            block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)
        
    def forward(self, x):
        return self.conv(x)


class UNet_Encoder(nn.Module):
    """3D UNet encoder that returns features from deepest to shallowest scale."""

    def __init__(
        self, 
        in_ch, 
        base_ch=None,
        block='BasicBlock', 
        scale=[2,2,2,2], 
        num_block=[2,2,2,2], 
        pool=True, 
        norm='in', 
        kernel_size=[3,3,3,3,3],
        channels: Optional[List[int]] = None
    ):
        super().__init__()
        from ..utils import get_block, get_norm
        
        block = get_block(block)
        norm = get_norm(norm)
        
        # Explicit channels make released checkpoints independent of base_ch conventions.
        if channels is not None:
            assert len(channels) == 5, f"Expected 5 channel values, got {len(channels)}"
            ch = channels
        else:
            assert base_ch is not None, "Either channels or base_ch must be provided"
            ch = [base_ch, 2*base_ch, 4*base_ch, 8*base_ch, 16*base_ch]
        
        self.channels = ch
        
        self.inc = inconv(in_ch, ch[0], block=block, kernel_size=kernel_size[0], norm=norm)
        self.down1 = down_block(ch[0], ch[1], num_block=num_block[0], block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(ch[1], ch[2], num_block=num_block[1], block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(ch[2], ch[3], num_block=num_block[2], block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(ch[3], ch[4], num_block=num_block[3], block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)
    
    def forward(self, img):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder and task-encoding code expect deep-to-shallow ordering.
        return (x5, x4, x3, x2, x1)

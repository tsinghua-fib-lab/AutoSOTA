import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class RelatedConv(nn.Module):
    def __init__(self, out_channel, kernel_size, dilation = 1):
        super(RelatedConv, self).__init__()
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list)
        in_channels = 1
        receptive_field = (kernel_size[-1] - 1) * dilation + 1
        padding = receptive_field // 2
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size,
                              padding = [0, padding],
                              dilation = dilation
                              )
        self.remove = 1 if receptive_field % 2 == 0 else 0

    def forward(self, x):
        x = x.unsqueeze(1)  # B * Ch * T => B * 1 * Ch * T
        x = self.conv(x)    # B * 1 * Ch * T => B * out * 1 * T
        if self.remove > 0:
            x = x[..., :-self.remove]
        return x.squeeze(2) # B * out * 1 * T => B * out * T


class RelatedEncoder(nn.Module):
    def __init__(self, out_channels, channel, kernel_size: int):
        super(RelatedEncoder, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(out_channels)

        out_channels.insert(0, channel)
        self.net = nn.Sequential(*[
            RelatedConv(out_channels[i], kernel_size = (out_channels[i - 1], kernel_size[i - 1]))
            for i in range(1, len(out_channels))
        ])

    def forward(self, x):
        return self.net(x)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
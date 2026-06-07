"""3D convolutional building blocks for MASS models.

The encoder and decoder use these residual, bottleneck, depthwise, and
normalization/activation blocks to build UNet-style feature hierarchies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_blocks import LayerNorm

__all__ = [
    'ConvNormAct',
    'BasicBlock',
    'Bottleneck',
    'DepthwiseSeparableConv',
    'SingleConv',
    'TransposedConvNormAct',
    'SEBlock',
    'MBConv',
    'FusedMBConv',
    'DropPath'
]


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN as IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch, eps=1e-4) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch, eps=1e-4) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x): 
    
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out 


class TransposedConvNormAct(nn.Module):
    """
    Layer grouping a transposed convolution, normalization and activation function
    """
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.transposed_conv = nn.ConvTranspose3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch, eps=1e-4) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch, eps=1e-4) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x): 
    
        if self.preact:
            out = self.transposed_conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.transposed_conv(x)))

        return out 


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, groups=1, dilation=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.expansion = 8
        self.conv1 = ConvNormAct(in_ch, out_ch//self.expansion, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//self.expansion, out_ch//self.expansion, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch//self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, list):
            padding = [i//2 for i in kernel_size]
        else:
            padding = kernel_size // 2

        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
                        nn.Conv3d(in_ch, in_ch//ratio, kernel_size=1),
                        act(),
                        nn.Conv3d(in_ch//ratio, in_ch, kernel_size=1),
                        nn.Sigmoid()
        )
    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out


class DropPath(nn.Module):
    
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]
        expanded = expansion * in_ch
        self.se = se

        self.expand_proj = nn.Identity() if (expansion==1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0, norm=norm, act=act, preact=True)

        self.depthwise = ConvNormAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=expanded, act=act, norm=norm, preact=True)

        if self.se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))


    def forward(self, x):
        residual = x

        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x += self.shortcut(residual)

        return x


class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size -1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]

        expanded = expansion * in_ch

        self.stride= stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x

# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate
from ptflops import get_model_complexity_info as info


ch = 64
n_blocks = 8


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return avg_out + max_out


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        # 添加通道注意力模块，输入通道数为 out_channels * 2
        self.channel_attn = ChannelAttention(out_channels * 2)
        self.conv_layer2 = torch.nn.Conv2d(out_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        height, width = ffted.shape[-2:]
        coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        # 添加通道注意力操作
        attn = self.channel_attn(ffted)
        ffted = ffted * attn  # 注意力加权
        ffted = self.conv_layer2(ffted)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        return output


class SpectralTransform(nn.Module):
    def __init__(self):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.fu = FourierUnit(ch // 2, ch // 2)
        self.conv2 = nn.Conv2d(ch, ch // 2, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(torch.cat([x, x2], dim=1))
        return x

class WindowStd(nn.Module):
    """
    实现窗口标准差统计的模块
    输入: (B, C, H, W) 形状的特征图
    输出: (B, C, H, W) 形状的特征图，与输入尺寸一致
    使用镜像padding保证边缘计算的准确性
    """
    def __init__(self, kernel_size=3, channels=None, eps=1e-5):
        """
        参数:
            kernel_size (int or tuple): 窗口大小，默认3x3
            channels (int): 输入特征图的通道数，用于初始化卷积核
                            若为None，需在第一次前向传播时自动推断
            eps (float): 数值稳定性参数，避免除以零或开方负数
        """
        super(WindowStd, self).__init__()
        # 统一kernel_size为元组格式
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2, "kernel_size必须是整数或二元组"
            self.kernel_size = kernel_size
        
        self.channels = channels
        self.eps = eps
        self.weight = None  # 用于存储均值卷积核（计算窗口内均值）
        
        # 计算padding大小（保证输出尺寸与输入一致）
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        
        # 初始化卷积核（如果通道数已知）
        if self.channels is not None:
            self._init_weight()

    def _init_weight(self):
        """初始化均值卷积核：每个通道的窗口内权重均为1/(k_h*k_w)"""
        kernel_h, kernel_w = self.kernel_size
        kernel_area = kernel_h * kernel_w
        # 创建单个通道的卷积核 (1, 1, k_h, k_w)
        single_kernel = torch.ones(1, 1, kernel_h, kernel_w) / kernel_area
        # 扩展到所有通道 (C, 1, k_h, k_w)，每个通道独立计算
        self.weight = single_kernel.repeat(self.channels, 1, 1, 1)
        # 将权重注册为非参数化缓冲区（不参与训练）
        self.register_buffer('mean_kernel', self.weight)

    def forward(self, x):
        """
        前向传播：对输入特征图应用窗口标准差计算
        
        标准差计算原理：
        std = sqrt( E[x²] - (E[x])² )
        其中E[·]表示窗口内的均值
        
        参数:
            x: 输入特征图，形状为 (B, C, H, W)
            
        返回:
            窗口标准差特征图，形状为 (B, C, H, W)
        """
        # 检查输入形状
        assert len(x.shape) == 4, "输入必须是4维张量 (B, C, H, W)"
        batch_size, channels, height, width = x.shape
        
        # 如果通道数未初始化，自动推断并初始化卷积核
        if self.channels is None:
            self.channels = channels
            self._init_weight()
        else:
            assert self.channels == channels, f"输入通道数({channels})与初始化通道数({self.channels})不匹配"
        
        # 镜像padding
        x_padded = F.pad(x, 
                        pad=(self.padding[1], self.padding[1],  # 左右padding
                             self.padding[0], self.padding[0]),  # 上下padding
                        mode='reflect')
        
        # 计算窗口内均值 E[x]
        mean = F.conv2d(x_padded, 
                       weight=self.mean_kernel, 
                       bias=None, 
                       stride=1, 
                       padding=0, 
                       groups=channels)
        
        # 计算窗口内平方的均值 E[x²]
        x_squared = x **2
        x_squared_padded = F.pad(x_squared, 
                                pad=(self.padding[1], self.padding[1],
                                     self.padding[0], self.padding[0]),
                                mode='reflect')
        mean_squared = F.conv2d(x_squared_padded, 
                               weight=self.mean_kernel, 
                               bias=None, 
                               stride=1, 
                               padding=0, 
                               groups=channels)
        
        # 计算标准差：sqrt(E[x²] - (E[x])² + eps)
        std = torch.sqrt(torch.clamp(mean_squared - mean** 2, min=self.eps))
        
        return std

class FFC(nn.Module):
    def __init__(self):
        super(FFC, self).__init__()
        self.convl2l = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.conv1 = nn.Conv2d(ch // 2, ch // 2, 1, bias=False)
        self.std = WindowStd(3, ch // 2)
        self.sig = nn.Sigmoid()
        # self.convl2g = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        # self.convg2l = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.convg2g = SpectralTransform()
    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        #
        feature = self.convl2l(x_l)
        std = self.std(x_l)
        weight = self.sig(self.conv1(std))
        out_xl = feature * weight

        out_xg = self.convg2g(x_g)

        return out_xl, out_xg



class FFC_BN_ACT(nn.Module):
    def __init__(self):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC()
        self.bn_l = nn.BatchNorm2d(ch // 2)
        self.bn_g = nn.BatchNorm2d(ch // 2)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FFC_BN_ACT()
        self.conv2 = FFC_BN_ACT()
        self.fuse = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        x_l, x_g = torch.split(x, (ch // 2, ch // 2), dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = torch.cat((x_l, x_g), dim=1)
        out = self.fuse(out)
        return out


class Net(nn.Module):
    def __init__(self, use_tta=True):
        super(Net, self).__init__()
        self.use_tta = use_tta
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(FFCResnetBlock())
        self.body = nn.Sequential(*self.blocks)
        self.head_conv = nn.Conv2d(1, ch, 3, 1, 1)
        self.tail_conv = nn.Conv2d(ch, 1, 3, 1, 1)

    def _forward_once(self, x):
        x = self.head_conv(x)
        shortcut = x
        x = self.body(x)
        x += shortcut
        x = self.tail_conv(x)
        return x

    def forward(self, x):
        if not self.use_tta or self.training:
            return self._forward_once(x)
        # 4x flip ensemble (original + h-flip + v-flip + hv-flip)
        out1 = self._forward_once(x)
        x_hf = torch.flip(x, [-1])
        out2 = torch.flip(self._forward_once(x_hf), [-1])
        x_vf = torch.flip(x, [-2])
        out3 = torch.flip(self._forward_once(x_vf), [-2])
        x_hvf = torch.flip(x_vf, [-1])
        out4 = torch.flip(torch.flip(self._forward_once(x_hvf), [-1]), [-2])
        # Weighted average: original gets 0.4, each flip gets 0.2
        return out1 * 0.4 + (out2 + out3 + out4) * 0.2


if __name__ == '__main__':
    flops, params = info(Net(), (1, 256, 256), as_strings=False,
                         print_per_layer_stat=False, verbose=False)
    print(flops, params)
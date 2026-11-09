from __future__ import annotations
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn1d import WaveBlock, SimplifiedWaveBlock
from .resnet1d import BasicBlock, MyConv1dPadSame, MyMaxPool1dPadSame


class WaveNetSpectrogram(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 base_filters: int | tuple = 128,
                 wave_layers: tuple = (10, 6, 2),
                 wave_block: str = 'simplified',
                 kernel_size: int = 3, 
                 downsample: int = 4,
                 sigmoid: bool = False, 
                 output_size: int = None, 
                 separate_channel: bool = False,
                 reinit: bool = True):

        super().__init__()

        if wave_block == 'simplified':
            wave_block = SimplifiedWaveBlock
        else:
            wave_block = WaveBlock

        self.out_chans = len(wave_layers)
        self.out_size = output_size
        self.sigmoid = sigmoid
        self.separate_channel = separate_channel
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])

        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_blocks = [wave_block(
                wave_layers[i], 
                in_channels, 
                base_filters[0], 
                kernel_size,
                downsample)]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_blocks = tmp_blocks + [
                        wave_block(
                            wave_layers[i], 
                            base_filters[j], 
                            base_filters[j+1], 
                            kernel_size,
                            downsample)
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_blocks))
            else:
                self.spec_conv.append(tmp_blocks[0])
        
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))
        
        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x): # x: (bs, ch, w)
        out = []
        if not self.separate_channel:
            for i in range(self.out_chans):
                out.append(self.spec_conv[i](x))
        else:
            for i in range(self.out_chans):
                out.append(self.spec_conv[i](x[:, i, :].unsqueeze(1)))
        out = torch.stack(out, dim=1)
        if self.out_size is not None:
            out = self.pool(out)
        if self.sigmoid:
            out = out.sigmoid()
        return out


class CNNSpectrogram(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 base_filters: int | tuple = 128, 
                 kernel_sizes: tuple = (32, 16, 4), 
                 stride: int = 4, 
                 sigmoid: bool = False, 
                 output_size: int = None,
                 conv: Callable = nn.Conv1d, 
                 disable_amp: bool = False, 
                 reinit: bool = True):

        super().__init__()

        self.out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        self.disable_amp = disable_amp
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                   in_channels, 
                   base_filters[0],
                   kernel_size=kernel_sizes[i],
                   stride=stride,
                   padding=(kernel_sizes[i]-1)//2,)]
            if len(base_filters) > 1:
                for j in range(len(base_filters)-1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j], 
                            base_filters[j+1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i]-1)//2,)
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])
        
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))
        
        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        if self.disable_amp: 
            with torch.cuda.amp.autocast(enabled=False):
                for i in range(self.out_chans):
                    out.append(self.spec_conv[i](x))
        else:
            for i in range(self.out_chans):
                out.append(self.spec_conv[i](x))
        out = torch.stack(out, dim=1)
        if self.out_size is not None:
            out = self.pool(out)
        if self.sigmoid:
            out = out.sigmoid()
        return out


class ResNetSpectrogram(nn.Module):

    def __init__(self, 
                 in_channels: int = 3, 
                 base_filters: int = 32, 
                 kernel_size: int = 3, 
                 stride: int = 2, 
                 groups: int = 32, 
                 n_block: int = 16,
                 downsample_gap: int = 8,
                 increasefilter_gap: int = 6,
                 transpose: bool = False,
                 reinit: bool = False):
        super().__init__()
        
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.activation = nn.ReLU
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.transpose = transpose

        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=base_filters, 
            kernel_size=self.kernel_size, 
            stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = self.activation(inplace=True)
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=downsample, 
                use_bn=True,  
                dropout=0.0, 
                activation=self.activation,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # init 
        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.first_block_conv(x)
        out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        for i_block in range(self.n_block):
            out = self.basicblock_list[i_block](out)
        
        out = out.unsqueeze(1)

        if self.transpose:
            out = out.transpose(2, 3)
        return out

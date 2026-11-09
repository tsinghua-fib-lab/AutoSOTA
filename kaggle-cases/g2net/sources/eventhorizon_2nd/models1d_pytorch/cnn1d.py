from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    '''
    '''
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1, 
                 hidden_dims: tuple = (8, 16, 32, 64), 
                 kernel_size: int = 3,
                 reinit: bool = False):

        super().__init__()

        # first conv
        self.features= nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=hidden_dims[0],
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(inplace=True))
        
        for i in range(len(hidden_dims)-1):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dims[i],
                          out_channels=hidden_dims[i+1],
                          kernel_size=kernel_size,
                          padding='same'),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.SiLU(inplace=True)
            )
            self.features.add_module(f'block{i}', conv_layer)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], num_classes))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)


'''
WaveNet variants
'''
class WaveBlock(nn.Module):

    def __init__(self, num_layers, in_channels, num_filters, kernel_size, downsample=False):
        super().__init__()
        dilation_rates = [2**i for i in range(num_layers)]
        if downsample:
            if isinstance(downsample, bool):
                first_stride = 2
            else:
                first_stride = downsample
            first_kernel_size = first_stride + 1
            first_padding = (first_kernel_size-1)//2
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=first_kernel_size,
                                        stride=first_stride,
                                        padding=first_padding)
        else:
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=1,
                                        padding='same')
        self.tanh_conv = nn.ModuleList()
        self.sigm_conv = nn.ModuleList()
        self.final_conv = nn.ModuleList()
        for _, dilation_rate in enumerate(dilation_rates):
            self.tanh_conv.append(nn.Sequential(
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          padding='same'),
                nn.Tanh(),
            ))
            self.sigm_conv.append(nn.Sequential(
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          padding='same'),
                nn.Sigmoid(),
            ))
            self.final_conv.append(nn.Conv1d(num_filters,
                                             num_filters,
                                             kernel_size=1,
                                             padding='same'))
    
    def forward(self, x):
        x = self.first_conv(x)
        res_x = x
        for i in range(len(self.tanh_conv)):
            tanh_out = self.tanh_conv[i](x)
            sigm_out = self.sigm_conv[i](x)
            x = tanh_out * sigm_out
            x = self.final_conv[i](x)
            res_x = res_x + x
        
        return res_x


class SimplifiedWaveBlock(nn.Module):

    def __init__(self, num_layers, in_channels, num_filters, kernel_size, downsample=False):
        super().__init__()
        dilation_rates = [2**i for i in range(num_layers)]
        if downsample:
            if isinstance(downsample, bool):
                first_stride = 2
            else:
                first_stride = downsample
            first_kernel_size = first_stride + 1
            first_padding = (first_kernel_size-1)//2
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=first_kernel_size,
                                        stride=first_stride,
                                        padding=first_padding)
        else:
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=1,
                                        padding='same')
        self.conv_act = nn.ModuleList()
        for _, dilation_rate in enumerate(dilation_rates):
            self.conv_act.append(nn.Sequential(
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          padding='same'),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=1,
                          padding='same'),
            ))
    
    def forward(self, x):
        x = self.first_conv(x)
        res_x = x
        for i in range(len(self.conv_act)):
            x = self.conv_act[i](x)
            res_x = res_x + x
        return res_x
        

class WaveNet1d(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: tuple = (16, 32, 64, 128), 
                 wave_layers: tuple = (12, 8, 4, 1),
                 wave_block: str = 'none',
                 kernel_size: int = 3,
                 num_classes: int = 1,
                 activation: nn.Module = nn.ReLU, 
                 downsample: bool | tuple = False,
                 reinit: bool = False):

        super().__init__()

        if wave_block == 'simplified':
            wave_block = SimplifiedWaveBlock
        else:
            wave_block = WaveBlock

        assert len(hidden_dims) == len(wave_layers)
        num_blocks = len(hidden_dims)
        if not isinstance(downsample, bool):
            assert len(downsample) == len(hidden_dims)
        else:
            downsample = tuple([downsample for _ in range(num_blocks)])

        self.features = nn.Sequential()
        self.features.add_module(
            'waveblock0', wave_block(
                wave_layers[0], 
                in_channels, 
                hidden_dims[0], 
                kernel_size,
                downsample[0]))
        self.features.add_module(
            'bn0', nn.BatchNorm1d(hidden_dims[0]))
        for i in range(num_blocks-1):
            self.features.add_module(
                f'waveblock{i+1}', wave_block(
                    wave_layers[i+1], 
                    hidden_dims[i], 
                    hidden_dims[i+1], 
                    kernel_size,
                    downsample[i+1]))
            self.features.add_module(
                f'bn{i+1}', nn.BatchNorm1d(hidden_dims[i+1]))
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.final_act = activation(inplace=True)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.features(x)
        return self.final_pool(self.final_act(x))
    
    def forward(self, x):
        features = self.forward_features(x)
        features = features.squeeze(-1)
        return self.classifier(features)

    def reset_classifier(self):
        self.classifier = nn.Identity()

    def get_classifier(self):
        return self.classifier

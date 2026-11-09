'''
DenseNet for 1-d signal data
Author: Hiroshi Yoshihara
'''
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    '''
    '''
    def __init__(self, in_channels, growth_rate, bottleneck_size, kernel_size):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0
        self.num_bottleneck_output_filters = growth_rate * bottleneck_size
        if self.use_bottleneck:
            self.bn2 = nn.BatchNorm1d(in_channels)
            self.act2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(
                in_channels, 
                self.num_bottleneck_output_filters,
                kernel_size=1,
                stride=1,
                dilation=1)
        self.bn1 = nn.BatchNorm1d(self.num_bottleneck_output_filters)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            self.num_bottleneck_output_filters,
            growth_rate,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            dilation=1)

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv2(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        return x


class DenseBlock(nn.ModuleDict):
    '''
    '''
    def __init__(self, num_layers, in_channels, growth_rate, kernel_size, bottleneck_size):
        super().__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.add_module(f'denselayer{i}', 
                DenseLayer(in_channels + i * growth_rate, 
                           growth_rate, 
                           bottleneck_size,
                           kernel_size))

    def forward(self, x):
        layer_outputs = [x]
        for _, layer in self.items():
            x = layer(x)
            layer_outputs.append(x)
            x = torch.cat(layer_outputs, dim=1)
        return x


class TransitionBlock(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, dilation=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
        

class DenseNet1d(nn.Module):

    def __init__(
        self, 
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        bottleneck_size: int = 4,
        first_kernel_size: int = 7,
        kernel_size: int = 3, 
        in_channels: int = 3,
        num_classes: int = 1,
        reinit: bool = True,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels, num_init_features, 
                kernel_size=first_kernel_size, stride=2, padding=3, dilation=1),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
            )
            self.features.add_module(f'denseblock{i}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(
                    in_channels=num_features,
                    out_channels=num_features // 2)
                self.features.add_module(f'transition{i}', trans)
                num_features = num_features // 2
        
        self.final_bn = nn.BatchNorm1d(num_features)
        self.final_act = nn.ReLU(inplace=True)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        
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

    def forward_features(self, x):
        out = self.features(x)
        out = self.final_bn(out)
        out = self.final_act(out)
        out = self.final_pool(out)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        features = features.squeeze(-1)
        out = self.classifier(features)
        return out

    def reset_classifier(self):
        self.classifier = nn.Identity()
    
    def get_classifier(self):
        return self.classifier


def densenet121_1d(in_chans=3, num_classes=1, **kwargs):
    return DenseNet1d(32, (6, 12, 24, 16), 64, in_channels=in_chans, num_classes=num_classes, **kwargs)


def densenet201_1d(in_chans=3, num_classes=1, **kwargs):
    return DenseNet1d(32, (6, 12, 48, 32), 64, in_channels=in_chans, num_classes=num_classes, **kwargs)

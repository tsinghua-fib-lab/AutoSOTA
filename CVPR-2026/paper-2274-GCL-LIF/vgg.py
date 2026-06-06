import torch.nn as nn
import torch
from modules import OnlineNeuron, LIFNeuron
from resnet import conv3x3, ECAAttention
from typing import Literal

cfg = {
    'VGG11': [
        [64],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64],
        [64, 'M', 128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 'MM']
    ],
    'VGG16': [
        [64],
        [64, 'M', 128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64],
        [64, 'M', 128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, T, vgg_name, use_ternary, num_classes, use_dvs, use_eca, use_mem_bn, mode: Literal['vanilla','compression']='vanilla', weight_mode: Literal['vanilla','compression']='compression', use_lif: Literal['vanilla','online','online-learnable','none']='none'):
        super(VGG, self).__init__()
        self.init_channels = 3 if use_dvs is False else 2
        self.T = T
        self.use_ternary = use_ternary
        self.use_eca = use_eca
        self.use_mem_bn = use_mem_bn
        self.mode = mode
        self.weight_mode = weight_mode
        self.use_lif = use_lif
        self.layer1 = self._make_layers(cfg[vgg_name][0], True if use_dvs is False else False)
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])

        if use_dvs is True:
            self.classifier = nn.Linear(512*3*3, num_classes)   
        else:
            self.classifier = nn.Linear(512, num_classes)     

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, first_conv=False):
        layers = []
        is_online = True if ('online' in self.use_lif) else False
        is_learnable = True if ('learnable' in self.use_lif) else False

        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            elif x == 'MM':
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            else:
                if first_conv is True:
                    layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=False))
                else:
                    layers.append(conv3x3(self.init_channels, x, self.use_ternary) if self.weight_mode == 'compression' else nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                if self.use_eca > 0:
                    layers.append(ECAAttention())
                layers.append(OnlineNeuron(self.T, self.use_mem_bn, x, False, self.mode) if self.use_lif == 'none' else LIFNeuron(self.T, online_training=is_online, is_learnable=is_learnable))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def vgg11(T, num_classes, use_ternary, use_dvs, use_eca, use_mem_bn, mode):
    return VGG(T, 'VGG11', use_ternary, num_classes, use_dvs, use_eca, use_mem_bn, mode)

def vgg13(T, num_classes, use_ternary, use_dvs, use_eca, use_mem_bn, mode):
    return VGG(T, 'VGG13', use_ternary, num_classes, use_dvs, use_eca, use_mem_bn, mode)

def vgg16(T, num_classes, use_ternary, use_dvs, use_eca, use_mem_bn, mode):
    return VGG(T, 'VGG16', use_ternary, num_classes, use_dvs, use_eca, use_mem_bn, mode)

def vgg19(T, num_classes, use_ternary, use_dvs, use_eca, use_mem_bn, mode):
    return VGG(T, 'VGG19', use_ternary, num_classes, use_dvs, use_eca, use_mem_bn, mode)
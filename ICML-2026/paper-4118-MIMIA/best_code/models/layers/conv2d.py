import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias= False)

        # if bn == 'bn':
        #     self.bn = nn.BatchNorm2d(o)
        # else:
        #     self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

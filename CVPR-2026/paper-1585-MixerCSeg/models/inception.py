import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import BottConv


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        # self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        # self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # self.conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        
        self.conv = BottConv(out_channels * 4, out_channels, out_channels*2, kernel_size=1)
        self.branch1x1 = BottConv(in_channels, out_channels, in_channels//2, kernel_size=1)
        self.branch3x3 = BottConv(in_channels, out_channels, in_channels//2, kernel_size=3, padding=1)
        self.branch5x5 = BottConv(in_channels, out_channels, in_channels//2, kernel_size=5, padding=2)
        self.branch_pool = BottConv(in_channels, out_channels, in_channels//2, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv = BottConv(out_channels * 4, out_channels, out_channels*2, kernel_size=1)

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     # Custom initialization for each convolution layer
    #     nn.init.kaiming_normal_(self.branch1x1.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.branch3x3.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.branch5x5.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.branch_pool.weight, mode='fan_out', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(self.pool(x))

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        outputs = self.conv(outputs)  
        return outputs
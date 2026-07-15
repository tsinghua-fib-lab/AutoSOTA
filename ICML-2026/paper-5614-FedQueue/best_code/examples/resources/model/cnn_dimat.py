"""
CNN for DIMAT integration.
Same architecture as cnn.py but each layer is a unique nn.Module instance
so DIMAT's graph can register hooks on each one independently.
"""

import math
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_channel=1, num_classes=10, num_pixel=28):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.act1 = nn.ReLU(inplace=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.act2 = nn.ReLU(inplace=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.act3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.act2(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

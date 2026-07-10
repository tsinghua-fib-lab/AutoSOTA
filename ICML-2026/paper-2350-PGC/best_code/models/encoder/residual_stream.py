import torch
import torch.nn as nn


class ResidualStream(nn.Module):

    def __init__(self, out_channels: int = 256):
        super().__init__()

        self.out_channels = out_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        x = self.block1(residual)
        x = self.block2(x)
        x = self.block3(x)
        return x

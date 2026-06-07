import torch
from torch import nn


class Conv2dForXYZ(nn.Module):
    """Depth encoder that processes XYZ positional embeddings (192-dim sincos + 1 mask) via Conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, xyz_values: torch.FloatTensor) -> torch.Tensor:
        return self.conv(xyz_values)

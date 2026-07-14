"""
ResNet18 building blocks for General EFC.

Defines the network as a sequence of independent nn.Module blocks.
Each block boundary is a point where EFC places teaching signals
during Dynamical Inversion.

Block decomposition for ResNet18:
  [0] Stem        : Conv7x7 + BN + ReLU + MaxPool  (or Conv3x3 for small inputs)
  [1] BasicBlock  : 64  -> 64
  [2] BasicBlock  : 64  -> 64
  [3] BasicBlock  : 64  -> 128  (stride=2, with shortcut)
  [4] BasicBlock  : 128 -> 128
  [5] BasicBlock  : 128 -> 256  (stride=2, with shortcut)
  [6] BasicBlock  : 256 -> 256
  [7] BasicBlock  : 256 -> 512  (stride=2, with shortcut)
  [8] BasicBlock  : 512 -> 512
  [9] GlobalAvgPool + Flatten
  [10] Linear classifier
"""

import torch.nn as nn
import torch.nn.functional as F


class Stem(nn.Module):
    """ResNet stem: conv + bn + relu + maxpool."""

    def __init__(self, in_channels=3, out_channels=64, small_input=False):
        super().__init__()
        if small_input:
            # CIFAR-style: 3x3 conv, stride 1, no pooling
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )
        else:
            # ImageNet / TinyImageNet-style
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with optional downsampling shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=False)
        return out


class GlobalAvgPoolFlat(nn.Module):
    """Global average pooling followed by flatten to (B, C)."""

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_resnet18_blocks(num_classes, in_channels=3, small_input=False):
    """
    Build ResNet18 as a list of nn.Module blocks for General EFC.

    Args:
        num_classes: Total number of output classes (across all tasks).
        in_channels: Number of input channels (3 for RGB).
        small_input: If True, use CIFAR-style stem (no maxpool, stride-1 conv).

    Returns:
        List[nn.Module]: Ordered blocks ready for GeneralEFCNetwork.
    """
    return [
        Stem(in_channels, 64, small_input=small_input),   # [0]
        BasicBlock(64, 64),                                # [1]
        BasicBlock(64, 64),                                # [2]
        BasicBlock(64, 128, stride=2),                     # [3]
        BasicBlock(128, 128),                              # [4]
        BasicBlock(128, 256, stride=2),                    # [5]
        BasicBlock(256, 256),                              # [6]
        BasicBlock(256, 512, stride=2),                    # [7]
        BasicBlock(512, 512),                              # [8]
        GlobalAvgPoolFlat(),                               # [9]
        nn.Linear(512, num_classes),                       # [10]
    ]

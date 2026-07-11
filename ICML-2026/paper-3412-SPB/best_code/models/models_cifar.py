# Defines a baseline CNN and an equivariant CNN (using escnn/e2cnn) for rotated MNIST classification.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Equivariant G-CNN using e2cnn / escnn
try:
    from escnn import gspaces
    from escnn import nn as enn
except Exception as e:
    raise ImportError(
        "e2cnn (escnn) not found. Install with `pip install escnn` or see https://github.com/QUVA-Lab/e2cnn`."
    )


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EquivariantCNN(nn.Module):
    def __init__(self, num_classes=10, N=8):
        super().__init__()

        self.r2_act = gspaces.rot2dOnR2(N)

        in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.trivial_repr] * 3
        )

        n_fields1 = max(1, 24 // N)
        n_fields2 = max(1, 48 // N)

        out_type1 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields1)
        out_type2 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields2)

        self.eq_conv1 = enn.R2Conv(in_type, out_type1, kernel_size=5, padding=2, bias=False)
        self.eq_relu1 = enn.ReLU(out_type1, inplace=True)
        self.eq_pool1 = enn.PointwiseMaxPool(out_type1, kernel_size=2, stride=2)

        self.eq_conv2 = enn.R2Conv(out_type1, out_type2, kernel_size=5, padding=2, bias=False)
        self.eq_relu2 = enn.ReLU(out_type2, inplace=True)
        self.eq_pool2 = enn.PointwiseMaxPool(out_type2, kernel_size=2, stride=2)

        out_channels = out_type2.size

        self.fc1 = nn.Linear(out_channels * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(
            x,
            enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * 3)
        )

        x = self.eq_conv1(x)
        x = self.eq_relu1(x)
        x = self.eq_pool1(x)

        x = self.eq_conv2(x)
        x = self.eq_relu2(x)
        x = self.eq_pool2(x)

        x = x.tensor
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
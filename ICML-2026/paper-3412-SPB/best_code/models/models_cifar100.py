import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 128, 8, 8]

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EquivariantCNN(nn.Module):
    def __init__(self, num_classes=100, N=8):
        super().__init__()

        self.r2_act = gspaces.rot2dOnR2(N)

        in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.trivial_repr] * 3
        )

        n_fields1 = max(1, 32 // N)
        n_fields2 = max(1, 64 // N)

        out_type1 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields1)
        out_type2 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields2)

        self.eq_conv1 = enn.R2Conv(in_type, out_type1, kernel_size=5, padding=2, bias=False)
        self.eq_relu1 = enn.ReLU(out_type1, inplace=True)
        self.eq_pool1 = enn.PointwiseMaxPool(out_type1, kernel_size=2, stride=2)

        self.eq_conv2 = enn.R2Conv(out_type1, out_type2, kernel_size=5, padding=2, bias=False)
        self.eq_relu2 = enn.ReLU(out_type2, inplace=True)
        self.eq_pool2 = enn.PointwiseMaxPool(out_type2, kernel_size=2, stride=2)

        out_channels = out_type2.size  # = n_fields2 * N

        self.fc1 = nn.Linear(out_channels * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

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
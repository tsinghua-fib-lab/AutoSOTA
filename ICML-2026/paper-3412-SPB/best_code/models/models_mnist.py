# Defines a baseline CNN and an equivariant CNN (using escnn/e2cnn) for rotated MNIST classification.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline (non-equivariant) CNN used as a capacity-matched reference.
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 max pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after two poolings: 28->14->7
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Equivariant G-CNN using e2cnn / escnn
try:
    from escnn import gspaces
    from escnn import nn as enn
except Exception as e:
    raise ImportError(
        "e2cnn (escnn) not found. Install with `pip install escnn` or see https://github.com/QUVA-Lab/e2cnn`."
    )


class EquivariantCNN(nn.Module):
    """
    A 2-layer equivariant CNN using e2cnn (R2Conv / FieldType).
    This architecture mirrors BaselineCNN capacity by using regular_repr fields whose
    channel counts match the baseline (approx).
    - Rotational group: Rot2dOnR2(N) where N = number of discrete rotations (e.g. 8)
    - Input is treated as a scalar field (trivial representation) with 1 channel.
    """
    def __init__(self, num_classes=10, N=8):
        """
        N: number of discrete rotations (e.g. N=8 -> 45-degree symmetry).
        """
        super().__init__()

        # --- group / gspace ---
        self.r2_act = gspaces.rot2dOnR2(N)  # cyclic rotations of order N

        # --- field types ---
        # input: 1 scalar channel (trivial representation)
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # choose number of *fields* so that total channel count matches baseline:
        # baseline conv1: 32 channels -> choose 4 regular fields when N=8 (4 * 8 = 32)
        # baseline conv2: 64 channels -> choose 8 regular fields when N=8 (8 * 8 = 64)
        n_fields1 = max(1, 32 // N)
        n_fields2 = max(1, 64 // N)

        # out types: use regular_repr repeated n_fields*
        out_type1 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields1)
        out_type2 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * n_fields2)

        # --- equivariant layers ---
        # First equivariant convolution: from scalar input to regular fields
        self.eq_conv1 = enn.R2Conv(in_type, out_type1, kernel_size=5, padding=2, bias=False)
        self.eq_relu1 = enn.ReLU(out_type1, inplace=True)
        self.eq_pool1 = enn.PointwiseMaxPool(out_type1, kernel_size=2, stride=2)

        # Second equivariant convolution: regular fields -> regular fields
        self.eq_conv2 = enn.R2Conv(out_type1, out_type2, kernel_size=5, padding=2, bias=False)
        self.eq_relu2 = enn.ReLU(out_type2, inplace=True)
        self.eq_pool2 = enn.PointwiseMaxPool(out_type2, kernel_size=2, stride=2)

        # After the equivariant pooling the representation is a GeometricTensor.
        # Use `.tensor` to get the underlying torch tensor before linear layers.
        out_channels = out_type2.size  # total number of channels after decomposition
        self.fc1 = nn.Linear(out_channels * 7 * 7, 128) # match baseline spatial flattening
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: standard torch Tensor [B, 1, H, W]
        # convert to GeometricTensor with proper FieldType
        x = enn.GeometricTensor(x, enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))

        x = self.eq_conv1(x)
        x = self.eq_relu1(x)
        x = self.eq_pool1(x)

        x = self.eq_conv2(x)
        x = self.eq_relu2(x)
        x = self.eq_pool2(x)

        # extract raw tensor and continue with standard linear layers
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
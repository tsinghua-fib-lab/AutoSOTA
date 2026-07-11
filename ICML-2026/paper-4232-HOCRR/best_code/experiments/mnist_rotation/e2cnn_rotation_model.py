"""
Rotation-equivariant CNN for angle regression using e2cnn.
Based on the e2cnn examples, adapted for (cos θ, sin θ) regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import e2cnn.nn as enn
from e2cnn import gspaces


class RotationEquivariantCNN(nn.Module):
    """
    Rotation-equivariant CNN for angle regression.
    Uses e2cnn to build a model that is equivariant to rotations.
    """
    
    def __init__(self, N=8, num_classes=2):  # 2 outputs for (cos θ, sin θ)
        super(RotationEquivariantCNN, self).__init__()
        
        # Define the group action: rotations by multiples of 2π/N
        self.gspace = gspaces.Rot2dOnR2(N=N)
        
        # Input: 1 channel (grayscale MNIST)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        
        # Build the network layers
        self._build_network()
        
    def _build_network(self):
        """Build the equivariant network layers."""
        
        # First conv layer: 1 -> 32 channels
        self.conv1_type = enn.FieldType(self.gspace, 32 * [self.gspace.regular_repr])
        self.conv1 = enn.R2Conv(self.in_type, self.conv1_type, kernel_size=5, padding=2)
        self.bn1 = enn.InnerBatchNorm(self.conv1_type)
        self.relu1 = enn.ReLU(self.conv1_type)
        self.pool1 = enn.PointwiseMaxPool(self.conv1_type, kernel_size=2)
        
        # Second conv layer: 32 -> 64 channels
        self.conv2_type = enn.FieldType(self.gspace, 64 * [self.gspace.regular_repr])
        self.conv2 = enn.R2Conv(self.conv1_type, self.conv2_type, kernel_size=5, padding=2)
        self.bn2 = enn.InnerBatchNorm(self.conv2_type)
        self.relu2 = enn.ReLU(self.conv2_type)
        self.pool2 = enn.PointwiseMaxPool(self.conv2_type, kernel_size=2)
        
        # Third conv layer: 64 -> 128 channels
        self.conv3_type = enn.FieldType(self.gspace, 128 * [self.gspace.regular_repr])
        self.conv3 = enn.R2Conv(self.conv2_type, self.conv3_type, kernel_size=3, padding=1)
        self.bn3 = enn.InnerBatchNorm(self.conv3_type)
        self.relu3 = enn.ReLU(self.conv3_type)
        self.pool3 = enn.PointwiseMaxPool(self.conv3_type, kernel_size=2)
        
        # Final layer: map to invariant features (trivial representation)
        self.final_type = enn.FieldType(self.gspace, 256 * [self.gspace.trivial_repr])
        self.final_conv = enn.R2Conv(self.conv3_type, self.final_type, kernel_size=3, padding=1)
        self.final_bn = enn.InnerBatchNorm(self.final_type)
        self.final_relu = enn.ReLU(self.final_type)
        
        # Global average pooling and final regression head
        self.global_pool = enn.PointwiseAdaptiveAvgPool(self.final_type, output_size=1)
        self.fc = nn.Linear(self.final_type.size, 2)  # 2 outputs for (cos θ, sin θ)
        
    def forward(self, x):
        """Forward pass through the network."""
        
        # Wrap input in GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Final conv block (to trivial representation)
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Extract tensor and flatten
        x = x.tensor
        x = x.view(x.size(0), -1)
        
        # Final regression head
        x = self.fc(x)
        
        return x


class RotationEquivariantCNN_Simple(nn.Module):
    """
    Simpler rotation-equivariant CNN for angle regression.
    Based on the Wide ResNet architecture but adapted for regression.
    """
    
    def __init__(self, N=8):
        super(RotationEquivariantCNN_Simple, self).__init__()
        
        # Define the group action
        self.gspace = gspaces.Rot2dOnR2(N=N)
        
        # Input type: 1 channel grayscale
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        
        # Build network
        self._build_network()
        
    def _build_network(self):
        """Build a simpler equivariant network."""
        
        # First layer: 1 -> 16 channels
        self.conv1_type = enn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])
        self.conv1 = enn.R2Conv(self.in_type, self.conv1_type, kernel_size=7, padding=3)
        self.bn1 = enn.InnerBatchNorm(self.conv1_type)
        self.relu1 = enn.ReLU(self.conv1_type)
        self.pool1 = enn.PointwiseMaxPool(self.conv1_type, kernel_size=2)
        
        # Second layer: 16 -> 32 channels
        self.conv2_type = enn.FieldType(self.gspace, 32 * [self.gspace.regular_repr])
        self.conv2 = enn.R2Conv(self.conv1_type, self.conv2_type, kernel_size=5, padding=2)
        self.bn2 = enn.InnerBatchNorm(self.conv2_type)
        self.relu2 = enn.ReLU(self.conv2_type)
        self.pool2 = enn.PointwiseMaxPool(self.conv2_type, kernel_size=2)
        
        # Third layer: 32 -> 64 channels
        self.conv3_type = enn.FieldType(self.gspace, 64 * [self.gspace.regular_repr])
        self.conv3 = enn.R2Conv(self.conv2_type, self.conv3_type, kernel_size=3, padding=1)
        self.bn3 = enn.InnerBatchNorm(self.conv3_type)
        self.relu3 = enn.ReLU(self.conv3_type)
        self.pool3 = enn.PointwiseMaxPool(self.conv3_type, kernel_size=2)
        
        # Final layer: map to invariant features
        self.final_type = enn.FieldType(self.gspace, 128 * [self.gspace.trivial_repr])
        self.final_conv = enn.R2Conv(self.conv3_type, self.final_type, kernel_size=3, padding=1)
        self.final_bn = enn.InnerBatchNorm(self.final_type)
        self.final_relu = enn.ReLU(self.final_type)
        
        # Global pooling and regression head
        self.global_pool = enn.PointwiseAdaptiveAvgPool(self.final_type, output_size=1)
        self.fc = nn.Linear(self.final_type.size, 2)  # (cos θ, sin θ)
        
    def forward(self, x):
        """Forward pass."""
        
        # Wrap input
        x = enn.GeometricTensor(x, self.in_type)
        
        # Network layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        
        # Regression head
        x = self.fc(x)
        
        return x


def angle_to_cos_sin(angles_deg):
    """Convert angles in degrees to (cos θ, sin θ) representation."""
    angles_rad = torch.tensor(angles_deg, dtype=torch.float32) * math.pi / 180.0
    cos_theta = torch.cos(angles_rad)
    sin_theta = torch.sin(angles_rad)
    return torch.stack([cos_theta, sin_theta], dim=-1)


def cos_sin_to_angle(cos_sin):
    """Convert (cos θ, sin θ) representation back to angles in degrees."""
    cos_theta, sin_theta = cos_sin[:, 0], cos_sin[:, 1]
    angles_rad = torch.atan2(sin_theta, cos_theta)
    angles_deg = angles_rad * 180.0 / math.pi
    return angles_deg


def circular_mae_loss(pred_cos_sin, target_cos_sin):
    """Compute circular MAE loss for angle regression."""
    # Convert to angles
    pred_angles = cos_sin_to_angle(pred_cos_sin)
    target_angles = cos_sin_to_angle(target_cos_sin)
    
    # Compute circular difference
    diff = pred_angles - target_angles
    diff = ((diff + 180) % 360) - 180  # Wrap to [-180, 180]
    
    # Return mean absolute error
    return torch.mean(torch.abs(diff))


if __name__ == "__main__":
    # Test the model
    print("Testing RotationEquivariantCNN...")
    
    # Create model
    model = RotationEquivariantCNN_Simple(N=8)
    model.eval()
    
    # Test input
    x = torch.randn(2, 1, 28, 28)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output (cos θ, sin θ): {output}")
        
        # Convert to angles
        angles = cos_sin_to_angle(output)
        print(f"Predicted angles: {angles}")
    
    print("Model test completed!")

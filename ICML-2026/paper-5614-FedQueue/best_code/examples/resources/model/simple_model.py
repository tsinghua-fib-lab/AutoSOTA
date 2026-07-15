"""
Tiny neural network for TES testing - very small to avoid gRPC size limits
"""

import torch.nn as nn


class TinyNet(nn.Module):
    """Ultra-simple neural network for TES integration testing"""

    def __init__(self, input_size=8, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 4)  # Very small hidden layer
        self.fc2 = nn.Linear(4, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

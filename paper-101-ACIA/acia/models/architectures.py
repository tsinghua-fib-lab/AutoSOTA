# acia/models/architectures.py

import torch.nn as nn
import torchvision.models as models


class CausalRepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )

        self.phi_H = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits

class ImprovedCausalRepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Low-level encoder (unchanged)
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )

        # IMPROVED: High-level encoder with bottleneck for better filtering
        self.phi_H = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(64, 16),  # Bottleneck layer - forces compression
            nn.ReLU(),
            nn.Linear(16, 128),  # Expand back
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits

class LowLevelEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.conv_layers(x)

class HighLevelEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

class RMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder for low-level representation - 1 channel input (grayscale)
        self.phi_L = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)
        )

        # Encoder for high-level representation
        self.phi_H = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # Ensure proper shape for grayscale images
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits

class BallCausalModel(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.phi_H = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.position_head = nn.Linear(latent_dim, 8)  # 4 balls * 2 coordinates

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        positions = self.position_head(z_H)
        return z_L, z_H, positions

class CamelyonModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.phi_L = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
        self.phi_H = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.dropout(self.phi_H(z_L))
        logits = self.classifier(z_H)
        return z_L, z_H, logits


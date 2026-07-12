import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, with_alpha=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        if with_alpha:
            self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze_weights(self):
        for p in self.parameters():
            p.requires_grad_(True)
        return self

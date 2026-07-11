"""
Use pre-trained RotNet model for MNIST rotation angle prediction.
This avoids training from scratch and provides a stronger baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import os
from urllib.parse import urlparse

class RotNet(nn.Module):
    """
    RotNet architecture for rotation angle prediction.
    Based on the RotNet paper: "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"
    """
    
    def __init__(self, num_classes=360):
        super(RotNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class RotNetRegressor(nn.Module):
    """
    RotNet adapted for continuous regression output.
    Uses the same feature extractor but outputs a single angle value.
    """
    
    def __init__(self, pretrained_rotnet_path=None):
        super(RotNetRegressor, self).__init__()
        
        # Load pre-trained RotNet if available
        if pretrained_rotnet_path and os.path.exists(pretrained_rotnet_path):
            print(f"Loading pre-trained RotNet from {pretrained_rotnet_path}")
            self.rotnet = RotNet(num_classes=360)
            checkpoint = torch.load(pretrained_rotnet_path, map_location='cpu')
            self.rotnet.load_state_dict(checkpoint['state_dict'])
            
            # Freeze feature extractor
            for param in self.rotnet.features.parameters():
                param.requires_grad = False
                
            # Replace classifier with regression head
            self.rotnet.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1)  # Single output for angle
            )
        else:
            print("No pre-trained weights found, initializing from scratch")
            self.rotnet = RotNet(num_classes=360)
            # Replace classifier with regression head
            self.rotnet.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1)  # Single output for angle
            )
    
    def forward(self, x):
        return self.rotnet(x)

def download_pretrained_rotnet(save_path='pretrained_rotnet.pth'):
    """
    Download pre-trained RotNet weights if available.
    This is a placeholder - in practice, you would download from the actual repository.
    """
    # For now, we'll create a mock download function
    # In practice, you would download from the RotNet repository
    print("Note: This is a placeholder for downloading pre-trained RotNet weights.")
    print("In practice, you would download from the RotNet repository:")
    print("https://github.com/d4nst/RotNet")
    return None

def create_pretrained_model(pretrained_path=None):
    """Create RotNet regressor with optional pre-trained weights."""
    
    if pretrained_path and os.path.exists(pretrained_path):
        model = RotNetRegressor(pretrained_path)
    else:
        print("Creating RotNet from scratch (no pre-trained weights available)")
        model = RotNetRegressor()
    
    return model

def test_pretrained_model():
    """Test the pre-trained model."""
    print("Testing RotNet regressor...")
    
    # Create model
    model = create_pretrained_model()
    
    # Test with a batch
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Test the pre-trained model
    model = test_pretrained_model()
    
    print("\nModel architecture:")
    print(model)

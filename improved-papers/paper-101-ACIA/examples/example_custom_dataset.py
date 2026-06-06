"""
Example: Using ACIA with Your Own Dataset
==========================================

This example shows how to adapt ACIA to your own dataset.
Key requirements:
1. Dataset returns (x, y, e) tuples
2. x: input data, y: labels, e: environment/domain indicators
3. Multiple environments for causal learning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from acia.models import CausalRepresentationNetwork
from acia.models.training import CausalOptimizer


class CustomImageDataset(Dataset):
    """Template for custom image datasets."""
    
    def __init__(self, images, labels, environments):
        """
        Args:
            images: torch.Tensor of shape (N, C, H, W)
            labels: torch.Tensor of shape (N,) - target labels
            environments: torch.Tensor of shape (N,) - domain/environment IDs
        """
        self.images = images
        self.labels = labels
        self.environments = environments
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.environments[idx]


class CustomFeatureDataset(Dataset):
    """Template for pre-extracted features or tabular data."""
    
    def __init__(self, features, labels, environments):
        """
        Args:
            features: torch.Tensor of shape (N, feature_dim)
            labels: torch.Tensor of shape (N,)
            environments: torch.Tensor of shape (N,)
        """
        self.features = features
        self.labels = labels
        self.environments = environments
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.environments[idx]


class CustomEncoder(nn.Module):
    """Custom encoder for non-image data."""
    
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.phi_L = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.phi_H = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.classifier = nn.Linear(128, 10)  # Adjust output dim
    
    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits


def example_image_data():
    """Example with image data."""
    # Generate synthetic multi-environment image data
    n_samples_per_env = 1000
    n_classes = 10
    
    # Environment 1: Clean images
    images_e1 = torch.randn(n_samples_per_env, 3, 28, 28)
    labels_e1 = torch.randint(0, n_classes, (n_samples_per_env,))
    envs_e1 = torch.zeros(n_samples_per_env)
    
    # Environment 2: Images with different corruption
    images_e2 = torch.randn(n_samples_per_env, 3, 28, 28)
    labels_e2 = torch.randint(0, n_classes, (n_samples_per_env,))
    envs_e2 = torch.ones(n_samples_per_env)
    
    # Combine
    all_images = torch.cat([images_e1, images_e2])
    all_labels = torch.cat([labels_e1, labels_e2])
    all_envs = torch.cat([envs_e1, envs_e2])
    
    dataset = CustomImageDataset(all_images, all_labels, all_envs)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Use standard ACIA model
    model = CausalRepresentationNetwork()
    optimizer = CausalOptimizer(model, batch_size=64)
    
    # Training
    for epoch in range(5):
        for x, y, e in loader:
            metrics = optimizer.train_step(x, y, e)
        print(f"Epoch {epoch+1}: Loss = {metrics['total_loss']:.4f}")
    
    return model


def example_tabular_data():
    """Example with tabular/feature data."""
    # Generate synthetic tabular data
    n_samples_per_env = 500
    feature_dim = 50
    n_classes = 5
    
    # Environment 1
    features_e1 = torch.randn(n_samples_per_env, feature_dim)
    labels_e1 = torch.randint(0, n_classes, (n_samples_per_env,))
    envs_e1 = torch.zeros(n_samples_per_env)
    
    # Environment 2
    features_e2 = torch.randn(n_samples_per_env, feature_dim) + 0.5  # Domain shift
    labels_e2 = torch.randint(0, n_classes, (n_samples_per_env,))
    envs_e2 = torch.ones(n_samples_per_env)
    
    # Combine
    all_features = torch.cat([features_e1, features_e2])
    all_labels = torch.cat([labels_e1, labels_e2])
    all_envs = torch.cat([envs_e1, envs_e2])
    
    dataset = CustomFeatureDataset(all_features, all_labels, all_envs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Use custom encoder
    model = CustomEncoder(input_dim=feature_dim)
    optimizer = CausalOptimizer(model, batch_size=32)
    
    # Training
    for epoch in range(5):
        for x, y, e in loader:
            metrics = optimizer.train_step(x, y, e)
        print(f"Epoch {epoch+1}: Loss = {metrics['total_loss']:.4f}")
    
    return model


if __name__ == '__main__':
    print("Example 1: Image Data")
    print("=" * 50)
    model1 = example_image_data()
    
    print("\nExample 2: Tabular Data")
    print("=" * 50)
    model2 = example_tabular_data()
    
    print("\nBoth examples completed successfully!")

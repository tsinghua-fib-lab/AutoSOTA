"""
Tiny random dataset for TES testing - very small to avoid gRPC size limits
"""

import torch
from torch.utils.data import Dataset


class TinyDataset(Dataset):
    """Ultra-simple random dataset for TES integration testing"""

    def __init__(self, num_samples=100, input_size=8, num_classes=2):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

        # Generate random data
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_tiny_data(num_clients=2, client_id=0, num_samples=100):
    """
    Get tiny random dataset for a specific client

    Args:
        num_clients: Total number of clients
        client_id: ID of this client (0-indexed)
        num_samples: Number of samples per client

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Create different random seed for each client
    torch.manual_seed(42 + client_id)

    # Create train and test datasets
    train_dataset = TinyDataset(num_samples=num_samples)
    test_dataset = TinyDataset(num_samples=num_samples // 4)  # Smaller test set

    return train_dataset, test_dataset

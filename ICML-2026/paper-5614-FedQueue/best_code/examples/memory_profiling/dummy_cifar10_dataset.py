"""
Dummy CIFAR-10 dataset for memory profiling focused on training, not data loading.

This creates very small random datasets with the same shape as CIFAR-10 to isolate
memory usage during training from memory usage during data loading/partitioning.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class DummyCIFAR10Dataset(Dataset):
    """
    Dummy CIFAR-10 dataset that generates random data with CIFAR-10 dimensions.

    Args:
        num_samples: Number of samples in the dataset (default: 64 for small memory footprint)
        num_classes: Number of classes (default: 10 for CIFAR-10)
        image_size: Image dimensions (default: (3, 32, 32) for CIFAR-10)
        seed: Random seed for reproducibility
    """

    def __init__(self, num_samples=64, num_classes=10, image_size=(3, 32, 32), seed=42):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

        # Set seed for reproducible random data
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate random images with CIFAR-10 dimensions (3, 32, 32)
        # Values normalized to [0, 1] range like real CIFAR-10
        self.data = torch.rand(num_samples, *image_size)

        # Generate random labels
        self.labels = torch.randint(0, num_classes, (num_samples,))

        print(
            f"Created dummy CIFAR-10 dataset: {num_samples} samples, shape {image_size}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dummy_cifar10(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    samples_per_client: int = 64,
    **kwargs,
):
    """
    Return dummy CIFAR-10 datasets for a given client.

    This bypasses all the memory-intensive data partitioning and creates
    small random datasets for memory profiling focused on training.

    Args:
        num_clients: Total number of clients (not used, for compatibility)
        client_id: The client ID (used for different random seeds)
        partition_strategy: Partition strategy (ignored, for compatibility)
        samples_per_client: Number of samples per client dataset
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Create small train dataset with different seed for each client
    train_dataset = DummyCIFAR10Dataset(
        num_samples=samples_per_client,
        seed=42 + client_id,  # Different seed per client
    )

    # Create small test dataset (shared across clients)
    test_dataset = DummyCIFAR10Dataset(
        num_samples=32,  # Even smaller test set
        seed=1000,  # Same seed for all clients (shared test set)
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test the dummy dataset
    train_ds, test_ds = get_dummy_cifar10(num_clients=2, client_id=0)

    # Verify data shapes
    sample_data, sample_label = train_ds[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label: {sample_label}")
    print(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")

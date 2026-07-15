"""
File-based dataset loader for TES volume mounting.

This dataset loader reads pre-generated data files from mounted volumes,
demonstrating how TES containers can access host filesystem data.
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class FileBasedDataset(Dataset):
    """Dataset that loads data from mounted CSV files."""

    def __init__(self, data_dir: str = "/data"):
        """
        Initialize dataset from mounted volume.

        Args:
            data_dir: Directory where CSV files are mounted (default: /data)
        """
        self.data_dir = data_dir

        # Load metadata
        metadata_file = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            self.metadata = json.load(f)

        # Load features
        features_file = os.path.join(data_dir, self.metadata["features_file"])
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")

        self.features_df = pd.read_csv(features_file)
        self.features = torch.tensor(self.features_df.values, dtype=torch.float32)

        # Load labels
        labels_file = os.path.join(data_dir, self.metadata["labels_file"])
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        self.labels_df = pd.read_csv(labels_file)
        self.labels = torch.tensor(self.labels_df["label"].values, dtype=torch.long)

        print(f"Loaded dataset from {data_dir}:")
        print(f"  Client ID: {self.metadata['client_id']}")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Number of classes: {self.metadata['num_classes']}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_sample_size(self):
        """Return the number of samples in this dataset."""
        return len(self.features)


def get_file_based_data(data_dir: str = "/data", **kwargs):
    """
    Create file-based datasets for federated learning.

    This function is called by APPFL clients to load data from mounted volumes.

    Args:
        data_dir: Directory where data files are mounted
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Load the full dataset
    print(f"Debug: Loading dataset from {data_dir}")
    dataset = FileBasedDataset(data_dir)

    # For simplicity, use 80% for training, 20% for testing
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split: {train_size} train, {test_size} test samples")

    return train_dataset, test_dataset

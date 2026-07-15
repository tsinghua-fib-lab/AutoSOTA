#!/usr/bin/env python3
"""
Generate test datasets for TES volume mounting demo.

This script creates random datasets for federated learning clients
and saves them to the filesystem for volume mounting into containers.
"""

import os
import numpy as np
import pandas as pd


def generate_client_dataset(
    client_id: int, num_samples: int = 100, num_features: int = 8, num_classes: int = 2
):
    """Generate a random dataset for a specific client."""
    np.random.seed(42 + client_id)  # Different seed per client

    # Generate random features
    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # Generate random labels
    y = np.random.randint(0, num_classes, size=(num_samples,))

    # Create simple linear relationship with some noise
    weights = np.random.randn(num_features)
    linear_output = X @ weights
    y = (linear_output > np.median(linear_output)).astype(int)

    return X, y


def save_dataset(X, y, output_dir: str, client_id: int):
    """Save dataset as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save features
    features_file = os.path.join(output_dir, f"client_{client_id}_features.csv")
    pd.DataFrame(X).to_csv(
        features_file, index=False, header=[f"feature_{i}" for i in range(X.shape[1])]
    )

    # Save labels
    labels_file = os.path.join(output_dir, f"client_{client_id}_labels.csv")
    pd.DataFrame({"label": y}).to_csv(labels_file, index=False)

    print(f"Client {client_id} dataset saved:")
    print(f"  Features: {features_file} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"  Labels: {labels_file} ({len(y)} labels)")

    return features_file, labels_file


def main():
    """Generate datasets for all clients."""
    base_data_dir = "/tmp/tes-data"
    num_clients = 2

    print(f"Generating test datasets in {base_data_dir}/")
    print("=" * 50)

    for client_id in range(num_clients):
        # Create client-specific directory
        client_dir = os.path.join(base_data_dir, f"client_{client_id}")

        # Generate different sized datasets for variety
        num_samples = 100 + client_id * 50  # Client 0: 100, Client 1: 150

        X, y = generate_client_dataset(
            client_id=client_id, num_samples=num_samples, num_features=8, num_classes=2
        )

        features_file, labels_file = save_dataset(X, y, client_dir, client_id)

        # Create a metadata file
        metadata = {
            "client_id": client_id,
            "num_samples": num_samples,
            "num_features": X.shape[1],
            "num_classes": len(np.unique(y)),
            "data_type": "synthetic_random",
            "features_file": os.path.basename(features_file),
            "labels_file": os.path.basename(labels_file),
        }

        print(f"Debug: Metadata for client {client_id}: {metadata}")

        metadata_file = os.path.join(client_dir, "metadata.json")
        import json

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {metadata_file}")
        print()

    print("Dataset generation complete!")
    print("Data structure:")
    print(f"{base_data_dir}/")
    for client_id in range(num_clients):
        print(f"├── client_{client_id}/")
        print(f"│   ├── client_{client_id}_features.csv")
        print(f"│   ├── client_{client_id}_labels.csv")
        print("│   └── metadata.json")

    print("\nTo use with TES volume mounting:")
    print(
        f"Set volume_mounts in client configs to mount {base_data_dir}/client_X to /data"
    )


if __name__ == "__main__":
    main()

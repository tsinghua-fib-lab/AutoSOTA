"""
CIFAR-10 dataset for DIMAT paper replication.
Uses the same normalization as the paper:
  mean = [0.4914, 0.4822, 0.4465]
  std  = [0.2470, 0.2435, 0.2616]
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    class_noniid_partition,
)


def get_cifar10(
    num_clients: int,
    client_id: int,
    num_classes_per_client: int = 2,
    **kwargs,
):
    """
    Return the CIFAR-10 dataset for a given client with paper-matching normalization.
    Non-IID: each client gets exactly `num_classes_per_client` classes (default 2).
    """
    dir = os.getcwd() + "/datasets/RawData"

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    test_data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=test_transform
    )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    train_data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=True, transform=train_transform
    )

    # Non-IID partition: exactly num_classes_per_client classes per client
    c = num_classes_per_client
    # Remove keys that class_noniid_partition doesn't accept
    partition_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ("visualization", "output_dirname", "output_filename", "seed")
    }
    train_datasets = class_noniid_partition(
        train_data_raw,
        num_clients,
        Cmin={num_clients: c, "none": c},
        Cmax={num_clients: c, "none": c},
        **partition_kwargs,
    )

    return train_datasets[client_id], test_dataset

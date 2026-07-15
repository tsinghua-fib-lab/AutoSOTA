"""
CIFAR-10 dataset for DIMAT experiments with IID (class-balanced) partitioning.
Uses lazy transforms (applied on each __getitem__ call) matching the CIFAR-100
DIMAT setup.
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_cifar10(
    num_clients: int,
    client_id: int,
    **kwargs,
):
    """
    Return the CIFAR-10 dataset for a given client with paper-matching
    normalization. IID partition distributes each class equally across clients.
    """
    import os

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

    trainset = torchvision.datasets.CIFAR10(
        dir, download=True, train=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=test_transform
    )

    # Class-balanced IID partition
    labels = np.array(trainset.targets)
    classes = np.unique(labels)
    worker_indices = [[] for _ in range(num_clients)]
    for cls in classes:
        cls_indices = np.where(labels == cls)[0].tolist()
        samples_per_client = len(cls_indices) // num_clients
        for rank in range(num_clients):
            start = rank * samples_per_client
            end = start + samples_per_client
            worker_indices[rank].extend(cls_indices[start:end])

    train_subset = Subset(trainset, worker_indices[client_id])

    return train_subset, testset

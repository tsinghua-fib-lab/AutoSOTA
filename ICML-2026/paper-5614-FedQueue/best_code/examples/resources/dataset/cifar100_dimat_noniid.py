"""
CIFAR-100 dataset for DIMAT experiments with non-IID (class-based) partitioning.
Uses lazy transforms (applied on each __getitem__ call) matching the IID variant.
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_cifar100(
    num_clients: int,
    client_id: int,
    num_classes_per_client: int = 20,
    **kwargs,
):
    """
    Return the CIFAR-100 dataset for a given client with paper-matching
    normalization. Non-IID partition assigns each client a disjoint subset
    of classes.
    """
    import os

    dir = os.getcwd() + "/datasets/RawData"

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

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

    trainset = torchvision.datasets.CIFAR100(
        dir, download=True, train=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        dir, download=True, train=False, transform=test_transform
    )

    # Non-IID partition: each client gets num_classes_per_client classes
    labels = np.array(trainset.targets)
    classes = np.unique(labels)
    np.random.seed(42)
    np.random.shuffle(classes)

    # Assign classes to clients round-robin
    client_classes = [[] for _ in range(num_clients)]
    for i, cls in enumerate(classes):
        client_classes[i % num_clients].append(cls)

    # Gather indices for this client's classes
    my_classes = client_classes[client_id]
    worker_indices = []
    for cls in my_classes:
        cls_indices = np.where(labels == cls)[0].tolist()
        worker_indices.extend(cls_indices)

    train_subset = Subset(trainset, worker_indices)

    return train_subset, testset

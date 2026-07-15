"""
Proxy dataset for DIMAT aggregator with CIFAR-100 normalization.
Uses the full CIFAR-100 training set with lazy transforms (applied on each
__getitem__ call) matching the original DIMAT code's trainloader.
"""

import os
import torchvision
import torchvision.transforms as transforms


def get_cifar100_proxy(num_samples: int = None, **kwargs):
    """
    Return CIFAR-100 training set as proxy data.
    If num_samples is None, returns the full 50k dataset (matching original DIMAT).
    If num_samples is set, returns a subset of that size.
    """
    from torch.utils.data import Subset

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

    dataset = torchvision.datasets.CIFAR100(
        dir, download=True, train=True, transform=train_transform
    )

    if num_samples is not None and num_samples < len(dataset):
        return Subset(dataset, list(range(num_samples)))
    return dataset

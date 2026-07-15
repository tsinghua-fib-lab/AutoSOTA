"""
Proxy dataset for DIMAT aggregator with CIFAR-10 normalization.
Uses the same normalization as the paper training data.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset


def get_cifar10_proxy(num_samples: int = 500):
    """
    Return a small proxy dataset from CIFAR-10 test set with paper normalization.
    """
    dir = os.getcwd() + "/datasets/RawData"

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2470, 0.2435, 0.2616]

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=test_transform
    )

    num_samples = min(num_samples, len(data_raw))
    data_input = []
    data_label = []
    for idx in range(num_samples):
        data_input.append(data_raw[idx][0].tolist())
        data_label.append(data_raw[idx][1])

    return Dataset(torch.FloatTensor(data_input), torch.tensor(data_label))

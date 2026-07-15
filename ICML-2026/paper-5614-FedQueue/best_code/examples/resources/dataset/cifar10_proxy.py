"""
Proxy dataset for DIMAT aggregator.
Returns a small subset of CIFAR-10 for computing activation statistics
during the server-side model merging process.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset


def get_cifar10_proxy(num_samples: int = 500):
    """
    Return a small proxy dataset from CIFAR-10 test set.
    :param num_samples: number of samples to include (default 500)
    """
    dir = os.getcwd() + "/datasets/RawData"

    data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    num_samples = min(num_samples, len(data_raw))
    data_input = []
    data_label = []
    for idx in range(num_samples):
        data_input.append(data_raw[idx][0].tolist())
        data_label.append(data_raw[idx][1])

    return Dataset(torch.FloatTensor(data_input), torch.tensor(data_label))

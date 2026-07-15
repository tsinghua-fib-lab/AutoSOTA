"""
Proxy dataset for DIMAT aggregator with MNIST.
Uses a subset of the MNIST test set for activation statistics and BN reset.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset


def get_mnist_proxy(num_samples: int = 500):
    """
    Return a small proxy dataset from MNIST test set.
    """
    dir = os.getcwd() + "/datasets/RawData"

    data_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    num_samples = min(num_samples, len(data_raw))
    data_input = []
    data_label = []
    for idx in range(num_samples):
        data_input.append(data_raw[idx][0].tolist())
        data_label.append(data_raw[idx][1])

    return Dataset(torch.FloatTensor(data_input), torch.tensor(data_label))

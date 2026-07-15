import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset, iid_partition


def get_mnist():
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    # Partition the dataset
    train_datasets = iid_partition(train_data_raw, num_clients=1)

    return train_datasets[0], test_dataset

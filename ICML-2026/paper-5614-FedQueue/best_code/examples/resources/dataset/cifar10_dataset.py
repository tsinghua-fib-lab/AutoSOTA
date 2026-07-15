import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


def get_cifar10(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the CIFAR10 dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.CIFAR10(
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
    train_data_raw = torchvision.datasets.CIFAR10(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(
            train_data_raw, num_clients, **kwargs
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset

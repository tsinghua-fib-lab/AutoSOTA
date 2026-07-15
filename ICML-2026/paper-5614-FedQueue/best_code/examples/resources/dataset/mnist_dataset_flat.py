import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


class NumpyDataset:
    """Minimal dataset wrapper holding (X, y) numpy arrays.

    Supports len() and indexing so ClientAgent.get_sample_size() works correctly,
    and SklearnTrainer._to_numpy() can extract arrays without a DataLoader.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_mnist_flat(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    **kwargs,
):
    """
    Return flattened MNIST data (784-dim numpy arrays) for a given client.

    Args:
        num_clients: total number of FL clients.
        client_id: zero-based index of this client.
        partition_strategy: one of "iid", "class_noniid", or "dirichlet_noniid".
        **kwargs: forwarded to the partition helper.

    Returns:
        (train_dataset, test_dataset) as NumpyDataset instances.
    """
    dir = os.getcwd() + "/datasets/RawData"
    transform = transforms.ToTensor()

    # --- test set ---
    test_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transform
    )
    test_X = np.stack([test_raw[i][0].numpy().ravel() for i in range(len(test_raw))])
    test_y = np.array([test_raw[i][1] for i in range(len(test_raw))])

    # --- training set ---
    train_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transform
    )

    if partition_strategy == "iid":
        train_datasets = iid_partition(train_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(train_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy!r}")

    client_raw = train_datasets[client_id]
    train_X = np.stack(
        [client_raw[i][0].numpy().ravel() for i in range(len(client_raw))]
    )
    train_y = np.array([client_raw[i][1] for i in range(len(client_raw))])

    return NumpyDataset(train_X, train_y), NumpyDataset(test_X, test_y)

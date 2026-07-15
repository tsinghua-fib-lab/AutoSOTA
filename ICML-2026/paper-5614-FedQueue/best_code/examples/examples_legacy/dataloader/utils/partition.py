import torch
import numpy as np
from .plot import plot_distribution
from appfl.misc.data import Dataset


def iid_partition(train_data_raw, num_clients, visualization, output=None):
    label_counts = {}
    client_dataset_info = {}

    # training data for multiple clients
    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):
        client_dataset_info[i] = {}
        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

            label = train_data_raw[idx][1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            if label not in client_dataset_info[i]:
                client_dataset_info[i][label] = 0
            client_dataset_info[i][label] += 1
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    # Visualize the data distribution among clients
    if visualization:
        classes = []
        for i in range(len(label_counts)):
            classes.append(label_counts[i])
        res = np.zeros((len(classes), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                res[cls][i] = client_dataset_info[i][cls]
        plot_distribution(num_clients, classes, res, output)
    return train_datasets


def class_noiid_partition(
    train_data_raw, num_clients, visualization, output=None, seed=42, **kwargs
):
    np.random.seed(seed)
    # training data for multiple clients
    Cmin = {
        1: 10,
        2: 7,
        3: 6,
        4: 5,
        5: 5,
        6: 4,
        7: 4,
        "none": 3,
    }  # minimum sample classes for each client
    Cmax = {
        1: 10,
        2: 8,
        3: 8,
        4: 7,
        5: 6,
        6: 6,
        7: 5,
        "none": 5,
    }  # maximum sample classes for each client

    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_data_raw):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Obtain the way to partition the dataset
    while True:
        class_partition = {}  # number of partitions for each class of MNIST
        client_classes = {}  # sample classes for different clients
        for i in range(num_clients):
            cmin = Cmin[num_clients] if num_clients in Cmin else Cmin["none"]
            cmax = Cmax[num_clients] if num_clients in Cmax else Cmax["none"]
            cnum = np.random.randint(cmin, cmax + 1)
            classes = np.random.permutation(range(10))[:cnum]
            client_classes[i] = classes
            for cls in classes:
                if cls in class_partition:
                    class_partition[cls] += 1
                else:
                    class_partition[cls] = 1
        if len(class_partition) == 10:
            break

    # Calculate how to partition the dataset
    partition_endpoints = {}
    for label in labels:
        total_size = len(label_indices[label])

        # Partition the samples from the same class to different lengths
        partitions = class_partition[label]
        partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))

        # Scale the lengths so they add to the total length
        partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size

        # Calculate the endpoints of each subrange
        endpoints = np.cumsum(partition_lengths)
        endpoints = np.array(endpoints, dtype=np.int32)
        endpoints[-1] = total_size
        partition_endpoints[label] = endpoints

    # Start dataset partition
    partition_pointer = {}
    for label in labels:
        partition_pointer[label] = 0
    client_datasets = []
    client_dataset_info = {}
    for i in range(num_clients):
        client_dataset_info[i] = {}
        sample_indices = []
        client_class = client_classes[i]
        for cls in client_class:
            start_idx = (
                0
                if partition_pointer[cls] == 0
                else partition_endpoints[cls][partition_pointer[cls] - 1]
            )  # included
            end_idx = partition_endpoints[cls][partition_pointer[cls]]  # excluded
            sample_indices.extend(label_indices[cls][start_idx:end_idx])
            partition_pointer[cls] += 1
            client_dataset_info[i][cls] = (
                end_idx - start_idx
            )  # record the number for different classes
        client_datasets.append(sample_indices)

    # Visualize the data distribution among clients
    if visualization:
        classes = [len(label_indices[label]) for label in labels]
        res = np.zeros((len(classes), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                res[cls][i] = client_dataset_info[i][cls]
        plot_distribution(num_clients, classes, res, output)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in client_datasets[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets


def dirichlet_noiid_partition(
    train_data_raw,
    num_clients,
    visualization,
    output=None,
    seed=42,
    alpha1=8,
    alpha2=0.5,
    **kwargs,
):
    np.random.seed(seed)
    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_data_raw):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Shuffle the indices for different label
    for label in labels:
        np.random.shuffle(label_indices[label])

    p1 = [
        1 / num_clients for _ in range(num_clients)
    ]  # prior distribution for each client's number of elements
    p2 = [len(label_indices[label]) for label in labels]
    p2 = [
        p / sum(p2) for p in p2
    ]  # prior distribution for each class's number of elements

    q1 = [alpha1 * i for i in p1]
    q2 = [alpha2 * i for i in p2]

    weights = np.random.dirichlet(q1)  # the total number of elements for each client
    individuals = np.random.dirichlet(
        q2, num_clients
    )  # the number of elements from each class for each client

    classes = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(classes)):
            normalized_portions[i][j] = (
                weights[i]
                * individuals[i][j]
                / np.dot(weights, individuals.transpose()[j])
            )

    res = np.multiply(
        np.array([classes] * num_clients), normalized_portions
    ).transpose()

    for i in range(len(classes)):
        total = 0
        for j in range(num_clients - 1):
            res[i][j] = int(res[i][j])
            total += res[i][j]
        res[i][num_clients - 1] = classes[i] - total

    if visualization:
        plot_distribution(num_clients, classes, res, output)

    # number of elements from each class for each client
    num_elements = np.array(res.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i - 1][j]
            end = sum_elements[i][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(train_data_raw[idx][0].tolist())
                train_data_label.append(train_data_raw[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets

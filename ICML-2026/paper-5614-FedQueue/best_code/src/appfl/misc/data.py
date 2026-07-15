"""
Miscellaneous data classes and processing functions for federated learning.
"""

import os
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from typing import List, Optional
from .deprecation import deprecated
import pandas as pd
import random


class Dataset(data.Dataset):
    """
    This class provides a simple way to define client dataset for supervised learning.
    This is derived from ``torch.utils.data.Dataset`` so that can be loaded to ``torch.utils.data.DataLoader``.
    Users may also create their own dataset class derived from this for more data processing steps.

    An empty ``Dataset`` class is created if no argument is given (i.e., ``Dataset()``).

    :param data_input (`torch.FloatTensor`): optional data inputs
    :param data_label (`torch.Tensor`): optional data outputs (or labels)
    """

    def __init__(
        self,
        data_input: torch.FloatTensor = torch.FloatTensor(),
        data_label: torch.Tensor = torch.Tensor(),
    ):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
        """This returns the sample size."""
        return len(self.data_label)

    def __getitem__(self, idx):
        """This returns a sample point for given ``idx``."""
        return self.data_input[idx], self.data_label[idx]


def plot_distribution(
    num_clients: int,
    classes_samples: List[int],
    sample_matrix: np.ndarray,
    output_dirname: Optional[str],
    output_filename: Optional[str],
):
    """
    Visualize the data distribution among clients for different classes.
    :param num_clients: number of clients
    :param classes_samples: number of samples for each class
    :param sample_matrix: the number of samples for each class for each client with shape (num_classes, num_clients)
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    """
    _, ax = plt.subplots(figsize=(20, num_clients / 2 + 3))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]

    for i in range(len(classes_samples)):
        ax.barh(
            y=range(num_clients),
            width=sample_matrix[i],
            left=np.sum(sample_matrix[:i], axis=0) if i > 0 else 0,
            color=colors[i],
        )

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])
    output_dirname = "output" if output_dirname is None else output_dirname
    output_filename = (
        "data_distribution.pdf" if output_filename is None else output_filename
    )
    output_filename = (
        f"{output_filename}.pdf"
        if not output_filename.endswith(".pdf")
        else output_filename
    )
    if not os.path.exists(output_dirname):
        pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)
    unique = 1
    unique_filename = output_filename
    filename_base, ext = os.path.splitext(output_filename)
    while pathlib.Path(os.path.join(output_dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    plt.savefig(os.path.join(output_dirname, unique_filename))


def iid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
) -> List[data.Dataset]:
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in an IID manner.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    train_dataset_split_indices = np.array_split(range(len(train_dataset)), num_clients)
    train_dataset_partitioned = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in train_dataset_split_indices[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])
        train_dataset_partitioned.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_dataset_partitioned


def class_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    num_of_classes=10,
    Cmin=None,
    Cmax=None,
    **kwargs,
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in a
    non-IID manner by letting each client only have a subset of all classes.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    :param seed: the random seed
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    np.random.seed(seed)
    # training data for multiple clients
    if Cmin is None:
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
    if Cmax is None:
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
    for idx, (_, label) in enumerate(train_dataset):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Obtain the way to partition the dataset
    while True:
        class_partition = {}  # number of partitions for each class of CIFAR10
        client_classes = {}  # sample classes for different clients
        for i in range(num_clients):
            cmin = Cmin[num_clients] if num_clients in Cmin else Cmin["none"]
            cmax = Cmax[num_clients] if num_clients in Cmax else Cmax["none"]
            cnum = np.random.randint(cmin, cmax + 1)
            classes = np.random.permutation(range(num_of_classes))[:cnum]
            client_classes[i] = classes
            for cls in classes:
                if cls in class_partition:
                    class_partition[cls] += 1
                else:
                    class_partition[cls] = 1
        if len(class_partition) == num_of_classes:
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
        classes_samples = [len(label_indices[label]) for label in labels]
        sample_matrix = np.zeros((len(classes_samples), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                sample_matrix[cls][i] = client_dataset_info[i][cls]
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in client_datasets[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets


def class_noniid_partition_binary(
    train_dataset: data.Dataset,
    num_clients: int,
    min_class_ratio: float = 0.1,  # Minimum proportion for minority class (ensures both classes exist)
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    **kwargs,
):
    """
    Partition a dataset into `num_clients` in a class non-IID manner where
    each client has both binary classes (0 and 1), but with different class ratios.

    Parameters:
    - min_class_ratio: Ensures that both classes exist per client by bounding class ratio between [min, 1-min].
    """
    np.random.seed(seed)

    # Separate dataset by class
    labels = [0, 1]
    label_indices = {label: [] for label in labels}
    for idx, (_, label) in enumerate(train_dataset):
        label_indices[label].append(idx)

    # Shuffle indices
    for label in labels:
        np.random.shuffle(label_indices[label])

    # Track how many samples we've used
    label_ptr = {label: 0 for label in labels}

    # Total available per class
    label_total = {label: len(label_indices[label]) for label in labels}

    # Prepare outputs
    client_datasets = []
    client_dataset_info = {}

    # Equal total samples per client for simplicity
    total_per_client = (label_total[0] + label_total[1]) // num_clients

    for i in range(num_clients):
        # Randomly choose a class ratio, making sure both classes are included
        class_0_ratio = np.random.uniform(min_class_ratio, 1 - min_class_ratio)

        num_class_0 = int(class_0_ratio * total_per_client)
        num_class_1 = total_per_client - num_class_0

        # Pull samples from class 0
        start_0 = label_ptr[0]
        end_0 = min(start_0 + num_class_0, label_total[0])
        indices_0 = label_indices[0][start_0:end_0]
        label_ptr[0] = end_0

        # Pull samples from class 1
        start_1 = label_ptr[1]
        end_1 = min(start_1 + num_class_1, label_total[1])
        indices_1 = label_indices[1][start_1:end_1]
        label_ptr[1] = end_1

        # Combine and shuffle
        client_indices = indices_0 + indices_1
        np.random.shuffle(client_indices)

        # Store
        inputs = [train_dataset[idx][0].tolist() for idx in client_indices]
        labels_ = [train_dataset[idx][1] for idx in client_indices]

        client_datasets.append(
            Dataset(
                torch.FloatTensor(inputs),
                torch.tensor(labels_),
            )
        )

        client_dataset_info[i] = {0: len(indices_0), 1: len(indices_1)}

    # Visualization
    if visualization:
        sample_matrix = np.zeros((2, num_clients))
        for i in range(num_clients):
            sample_matrix[0][i] = client_dataset_info[i][0]
            sample_matrix[1][i] = client_dataset_info[i][1]

        classes_samples = [label_total[0], label_total[1]]
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    return client_datasets


def dirichlet_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    alpha1: int = 8,
    alpha2: int = 0.5,
    seed: int = 42,
    **kwargs,
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks
    using two Dirichlet distributions: one for the number of elements for each client
    and the other for the number of elements from each class for each client.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    :param alpha1: the concentration parameter for the Dirichlet distribution for the number of elements for each client
    :param alpha2: the concentration parameter for the Dirichlet distribution for the number of elements from each class for each client
    :param seed: the random seed
    """
    np.random.seed(seed)
    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
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

    classes_samples = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(classes_samples)):
            normalized_portions[i][j] = (
                weights[i]
                * individuals[i][j]
                / np.dot(weights, individuals.transpose()[j])
            )

    sample_matrix = np.multiply(
        np.array([classes_samples] * num_clients), normalized_portions
    ).transpose()

    for i in range(len(classes_samples)):
        total = 0
        for j in range(num_clients - 1):
            sample_matrix[i][j] = int(sample_matrix[i][j])
            total += sample_matrix[i][j]
        sample_matrix[i][num_clients - 1] = classes_samples[i] - total

    if visualization:
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    # number of elements from each class for each client
    num_elements = np.array(sample_matrix.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i - 1][j]
            end = sum_elements[i][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(train_dataset[idx][0].tolist())
                train_data_label.append(train_dataset[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets


def iid_partition_df(
    train_df: pd.DataFrame,
    num_clients: int,
) -> List[pd.DataFrame]:
    """
    Partition a pandas DataFrame into `num_clients` chunks in an IID manner.

    :param train_df: the training DataFrame (assumes it has columns like 'features', 'label', etc.)
    :param num_clients: number of clients
    :return: A list of DataFrames, one per client
    """
    # Split the indices evenly among the clients
    indices_split = np.array_split(range(len(train_df)), num_clients)

    partitions = []
    for indices in indices_split:
        # Create a sub-DataFrame for this client
        sub_df = train_df.iloc[indices].copy()
        partitions.append(sub_df)

    return partitions


def class_noniid_partition_df(
    train_df: pd.DataFrame,
    num_clients: int,
    label_col: str = "label",
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    Cmin=None,
    Cmax=None,
    **kwargs,
) -> List[pd.DataFrame]:
    """
    Partition a pandas DataFrame into `num_clients` in a non-IID manner by
    letting each client only have a subset of all classes.

    :param train_df: the training DataFrame (assumes it has columns like 'features' and 'label')
    :param num_clients: number of clients
    :param label_col: the name of the label column
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot (if visualizing)
    :param output_filename: the filename to save the plot (if visualizing)
    :param seed: the random seed
    :return: A list of DataFrames, one per client
    """
    np.random.seed(seed)

    # Control how many classes each client gets
    # (You can change these if your problem differs from the CIFAR-like example)
    if Cmin is None:
        Cmin = {1: 10, 2: 7, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, "none": 3}
    if Cmax is None:
        Cmax = {1: 10, 2: 8, 3: 8, 4: 7, 5: 6, 6: 6, 7: 5, "none": 5}

    # Gather indices for each label
    label_indices = {}
    labels = []
    for i in range(len(train_df)):
        label = train_df.iloc[i][label_col]
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(i)
    labels.sort()

    # Randomly assign how many classes each client will get
    while True:
        class_partition = {}  # how many clients want each class
        client_classes = {}  # which classes go to each client

        for i in range(num_clients):
            cmin = Cmin[num_clients] if num_clients in Cmin else Cmin["none"]
            cmax = Cmax[num_clients] if num_clients in Cmax else Cmax["none"]
            cnum = np.random.randint(cmin, cmax + 1)
            classes = np.random.permutation(labels)[
                :cnum
            ]  # pick random classes for this client
            client_classes[i] = classes
            for cls in classes:
                class_partition[cls] = class_partition.get(cls, 0) + 1

        # Ensure all classes end up assigned at least once (e.g., 10 if it’s CIFAR-10)
        if len(class_partition) == len(labels):
            break

    # For each class, figure out how to split among clients that want that class
    partition_endpoints = {}
    for label in labels:
        total_size = len(label_indices[label])

        # Number of clients that want this class
        partitions = class_partition[label]

        # Randomly generate partition lengths, then normalize to sum up to total_size
        partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))
        partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size

        # Convert to integer endpoints
        endpoints = np.cumsum(partition_lengths).astype(np.int32)
        endpoints[-1] = total_size  # ensure we use all samples
        partition_endpoints[label] = endpoints

    partition_pointer = {label: 0 for label in labels}

    # Assign actual indices per client
    client_datasets_indices = [[] for _ in range(num_clients)]
    client_dataset_info = [{} for _ in range(num_clients)]

    for i in range(num_clients):
        for cls in client_classes[i]:
            start_idx = (
                0
                if partition_pointer[cls] == 0
                else partition_endpoints[cls][partition_pointer[cls] - 1]
            )
            end_idx = partition_endpoints[cls][partition_pointer[cls]]

            sub_indices = label_indices[cls][start_idx:end_idx]
            client_datasets_indices[i].extend(sub_indices)

            count_in_class = end_idx - start_idx
            client_dataset_info[i][cls] = count_in_class
            partition_pointer[cls] += 1

    if visualization:
        classes_samples = [len(label_indices[label]) for label in labels]
        sample_matrix = np.zeros((len(classes_samples), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                sample_matrix[cls][i] = client_dataset_info[i][cls]
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    # Build the list of sub-DataFrames for each client
    client_dataframes = []
    for indices in client_datasets_indices:
        sub_df = train_df.iloc[indices].copy()
        client_dataframes.append(sub_df)

    return client_dataframes


def dirichlet_noniid_partition_df(
    train_df: pd.DataFrame,
    num_clients: int,
    label_col: str = "label",
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    alpha1: float = 8,
    alpha2: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> List[pd.DataFrame]:
    """
    Partition a pandas DataFrame into `num_clients` using two Dirichlet distributions:
      1) one for how many total samples each client gets,
      2) one for how many samples of each class each client gets.

    :param train_df: the training DataFrame
    :param num_clients: number of clients
    :param label_col: the name of the label column
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot (if visualizing)
    :param output_filename: the filename to save the plot (if visualizing)
    :param alpha1: the Dirichlet parameter for number of elements per client
    :param alpha2: the Dirichlet parameter for number of elements per class per client
    :param seed: the random seed
    :return: A list of DataFrames, one per client
    """
    np.random.seed(seed)

    # Gather indices for each label
    label_indices = {}
    labels = []
    for i in range(len(train_df)):
        label = train_df.iloc[i][label_col]
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(i)
    labels.sort()

    # Shuffle each label's indices
    for label in labels:
        np.random.shuffle(label_indices[label])

    # Base distributions
    p1 = [1 / num_clients for _ in range(num_clients)]  # prior for total # of elements
    p2 = [len(label_indices[label]) for label in labels]
    p2 = [p / sum(p2) for p in p2]  # prior for class distribution

    q1 = [alpha1 * x for x in p1]
    q2 = [alpha2 * x for x in p2]

    # Dirichlet samples:
    weights = np.random.dirichlet(q1)  # how to split total # of elements among clients
    individuals = np.random.dirichlet(
        q2, num_clients
    )  # how to split each class among clients

    classes_samples = [len(label_indices[label]) for label in labels]

    # Re-normalize so that sample counts per class match exactly
    normalized_portions = np.zeros_like(individuals)
    for i in range(num_clients):
        for j in range(len(classes_samples)):
            # This line effectively normalizes so that sum over i for individuals[i][j]
            # matches the fraction of class j in the entire dataset
            normalized_portions[i][j] = (
                weights[i] * individuals[i][j] / np.dot(weights, individuals[:, j])
            )

    sample_matrix = np.multiply(
        np.array([classes_samples] * num_clients), normalized_portions
    ).T  # shape: (#classes, #clients)

    # Convert to integer counts per class-client combination
    for j in range(len(classes_samples)):
        total = 0
        for i in range(num_clients - 1):
            sample_matrix[j][i] = int(sample_matrix[j][i])
            total += sample_matrix[j][i]
        sample_matrix[j][num_clients - 1] = classes_samples[j] - total

    if visualization:
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    # Construct final splits
    num_elements = sample_matrix.astype(np.int32).T  # shape: (#clients, #classes)
    sum_elements = np.cumsum(
        num_elements, axis=0
    )  # cumulative sum down each column (class)

    client_dataframes = []
    for i in range(num_clients):
        sub_indices = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i - 1][j]
            end = sum_elements[i][j]
            sub_indices.extend(label_indices[label][start:end])

        sub_df = train_df.iloc[sub_indices].copy()
        client_dataframes.append(sub_df)

    return client_dataframes


def column_based_partition_df(
    train_df: pd.DataFrame,
    num_clients: int,
    label_col: str,
    partition_col: str,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    random_seed: Optional[int] = 42,
    **kwargs,
) -> List[pd.DataFrame]:
    """
    Partition a dataframe into a list of dataframes based on the unique values in `partition_col`.
    If #classes >= #clients, distribute classes across clients as evenly as possible.
    If #classes < #clients, split the largest partition (by number of rows) repeatedly
    until we reach #clients partitions or cannot split any more.

    Args:
        :param train_df: the training DataFrame
        :param num_clients: number of clients
        :param label_col: the name of the label column
        :param partition_col (str): The column containing the class or partition labels.
        :param visualization: whether to visualize the data distribution among clients
        :param output_dirname: the directory to save the plot (if visualizing)
        :param output_filename: the filename to save the plot (if visualizing)
        :param random_seed (int, optional): If provided, sets a seed for reproducible random splitting.
        **kwargs: Additional keyword arguments for future extensibility.

    Returns:
        List[pd.DataFrame]: A list of dataframes, each corresponding to one client's partition.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # group by class, one partition per unique class initially
    grouped = train_df.groupby(partition_col)
    partitions = [group.copy() for _, group in grouped]
    unique_classes = sorted(train_df[partition_col].unique())
    num_partitions = len(partitions)

    if num_partitions >= num_clients:
        # assign classes as evenly
        base_classes_per_client = num_partitions // num_clients
        remainder = num_partitions % num_clients

        client_dfs = []
        current_idx = 0
        for i in range(num_clients):
            c_count = base_classes_per_client + (1 if i < remainder else 0)
            if c_count == 0:
                client_dfs.append(pd.DataFrame(columns=train_df.columns))
                continue
            these_classes = unique_classes[current_idx : current_idx + c_count]
            current_idx += c_count
            client_df = train_df[train_df[partition_col].isin(these_classes)].copy()
            client_dfs.append(client_df)

    else:
        # split the largest partitions until we reach #clients or can't split further
        while len(partitions) < num_clients:
            largest_idx = max(range(len(partitions)), key=lambda i: len(partitions[i]))
            largest_df = partitions[largest_idx]
            size = len(largest_df)
            if size <= 1:
                break
            split_size = random.randint(1, size - 1)
            subset_a = largest_df.sample(n=split_size, replace=False, random_state=None)
            subset_b = largest_df.drop(subset_a.index)
            partitions.pop(largest_idx)
            partitions.append(subset_a)
            partitions.append(subset_b)

        # We assign each partition to a different client
        client_dfs = []
        for i in range(num_clients):
            if i < len(partitions):
                client_dfs.append(partitions[i])
            else:
                client_dfs.append(pd.DataFrame(columns=train_df.columns))

    if visualization:
        all_unique_classes = sorted(train_df[partition_col].unique())
        num_classes = len(all_unique_classes)
        sample_matrix = np.zeros((num_classes, num_clients), dtype=int)

        for j, df_j in enumerate(client_dfs):
            class_counts = df_j[partition_col].value_counts()
            for i, cls in enumerate(all_unique_classes):
                sample_matrix[i, j] = class_counts.get(cls, 0)

        classes_samples = list(sample_matrix.sum(axis=1))

        plot_distribution(
            num_clients=num_clients,
            classes_samples=classes_samples,
            sample_matrix=sample_matrix,
            output_dirname=output_dirname,
            output_filename=output_filename,
        )

    return client_dfs


@deprecated(silent=True)
def data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel):
    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = data.DataLoader(train_datasets[0], batch_size=64, shuffle=False)

    for input, label in train_dataloader:
        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel

    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    for input, label in test_dataloader:
        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel

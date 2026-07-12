"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import math
import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils import get_mask

DATA_PATH = "data"


def keep_only_largest_connected_component(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
    )
    dataset.data = data

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_train_val_test_split_im(seed: int, data: Data, dataset_name: str, split_idx: int = None) -> Data:

    if dataset_name in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
    ]:
        if dataset_name == "Cora":
            num_train = 140
            num_val = 1360
        elif dataset_name == "CiteSeer":
            num_train = 120
            num_val = 1300
        elif dataset_name == "PubMed":
            num_train = 60
            num_val = 1440
        elif dataset_name == "Photo":
            num_train = 160
            num_val = 1340
        elif dataset_name == "Computers":
            num_train = 200
            num_val = 1300


        data = set_uniform_train_val_test_split_by_num(
            seed=seed, data=data, num_train = num_train, num_val = num_val
        )

    return data

def set_train_val_test_split(seed: int, data: Data, dataset_name: str, split_idx: int = None) -> Data:

    if dataset_name in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers"
    ]:
        # Use split from "Diffusion Improves Graph Learning" paper, which selects 20 nodes for each class to be in the training set
        num_val = 5000 if dataset_name == "CoauthorCS" else 1500
        data = set_per_class_train_val_test_split(
            seed=seed, data=data, num_val=num_val, num_train_per_class=20, split_idx=split_idx,
        )
    elif dataset_name in ["OGBN-Arxiv"]:
        # OGBN datasets have pre-assigned split
        data.train_mask = split_idx["train"]
        data.val_mask = split_idx["valid"]
        data.test_mask = split_idx["test"]
    else:
        raise ValueError(f"We don't know how to split the data for {dataset_name}")

    return data


def set_per_class_train_val_test_split(
    seed: int, data: Data, num_val: int = 1500, num_train_per_class: int = 20, split_idx: int = None,
) -> Data:

    if split_idx is None:
        random.seed(seed)
        rnd_state = np.random.RandomState(seed)
        # rnd_state = np.random.RandomState(development_seed)
        num_nodes = data.y.shape[0]
        development_idx = rnd_state.choice(num_nodes, num_val, replace=False)
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []

        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_train_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]

        data.train_mask = get_mask(train_idx, num_nodes)
        data.val_mask = get_mask(val_idx, num_nodes)
        data.test_mask = get_mask(test_idx, num_nodes)

    else:
        data.train_mask = split_idx["train"]
        data.val_mask = split_idx["valid"]
        data.test_mask = split_idx["test"]

    return data


def set_uniform_train_val_test_split(seed: int, data: Data, train_ratio: float = 0.5, val_ratio: float = 0.25) -> Data:
    rnd_state = np.random.RandomState(seed)
    # np.random.seed(seed)
    num_nodes = data.y.shape[0]

    # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)

    idxs = list(range(num_labeled_nodes))
    # Shuffle in place
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train : num_train + num_val]
    test_idx = idxs[num_train + num_val :]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)

    # Set labels of unlabeled nodes to 0, otherwise there is an issue in label propagation (which does one-hot encoding of all labels)
    # This labels are not used since these nodes are excluded from all masks, do it doesn't affect any results
    data.y[data.y == -1] = 0

    return data


def set_uniform_train_val_test_split_by_num(seed: int, data: Data, num_train: int = 1, num_val: int = 1) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]

    # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]

    idxs = list(range(num_labeled_nodes))
    # Shuffle in place
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train : num_train + num_val]
    test_idx = idxs[num_train + num_val :]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)
    data.y[data.y == -1] = 0

    return data


# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (Amazon, Coauthor, Planetoid,
                                      TUDataset, Actor, WebKB, CitationFull)
from torch_geometric.utils import degree


def load_pyg(name, dataset_dir):
    """load_pyg
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif 'Coauthor' in name:
        dataset = Coauthor(dataset_dir, name=name[8:])
    elif 'Amazon' in name:
        dataset = Amazon(dataset_dir, name=name[6:])
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(dataset_dir, name)
    elif name == 'Actor':
        dataset = Actor(dataset_dir)
    elif name == 'DBLP':
        dataset = CitationFull(dataset_dir, name)
    elif name[:3] == 'TU_':  # Handling TUDataset for Graph Classification
        dataset = TUDataset(dataset_dir, name=name[3:])
        if name[3:] in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY']:
            initializeNodes(dataset)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset

# Taken from: https://github.com/lcicek/imdb-binary-gcn/blob/master/utility.py#L24
def initializeNodes(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

# Taken from: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py
# Found in: https://github.com/pyg-team/pytorch_geometric/discussions/3334
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


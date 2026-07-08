import os
from time import time
import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset
import json

### Code adapted from GraphEBM
def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])


def get_transform_fn(dataset, adj_scale):
    if dataset == 'QM9':
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F
            x_ = np.zeros((9, 5))
            indices = np.where(x >= 6, x - 6, 4)
            x_[np.arange(9), indices] = 1
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                    axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            adj = adj / adj_scale
            return x, adj

    elif dataset == 'ZINC250k':
        def transform(data):
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            x_ = np.zeros((38, 10), dtype=np.float32)
            for i in range(38):
                ind = zinc250k_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                                 axis=0).astype(np.float32)

            x = x[:, :-1]                               # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(adj.argmax(axis=0))      # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            adj = adj / adj_scale
            return x, adj

    return transform

def dataloader(config, get_graph_list=False):
    start_time = time()
    
    mols = load_mol(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    with open(os.path.join(config.data.dir, f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)
        
    if config.data.data == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
    
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data, config.model.adj_scale))
    test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data, config.model.adj_scale))

    if get_graph_list:
        train_mols_nx = [nx.from_numpy_array(np.array(adj)) for x, adj in train_dataset]
        test_mols_nx = [nx.from_numpy_array(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx

    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader

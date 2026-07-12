"""
version 1.0
date 2021/02/04
"""

import pickle as pkl
import sys

import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import scipy.sparse as sp
import torch

from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, dataset_str):
        if dataset_str in ['cora', 'citeseer']:
            data = load_planetoid_data(dataset_str)
        elif dataset_str in ['amaphoto', 'amacomp']:
            data = load_amazon_data(dataset_str)
        elif dataset_str in ['synthetic']:   
            d = torch.load('/home/ferrini/phd/gnn-missing-data/synthetic/pa_graph.pt')
            data = {
                'adj': d.adj,
                'edge_list': d.edge_index,
                'features': d.x,
                'labels': d.y,
                'G': to_networkx(d),
            }
        else:
            raise ValueError("Dataset {0} does not exist".format(dataset_str))
        self.adj = data['adj']
        self.edge_list = data['edge_list']
        self.features = data['features']
        self.labels = data['labels']
        self.G = data['G']
        self.num_features = self.features.size(1)

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)


class NodeClsData(Data):
    def __init__(self, dataset_str, data=None, seed=None):
        super(NodeClsData, self).__init__(dataset_str)
        if dataset_str in ['cora', 'citeseer']:
            train_mask, val_mask, test_mask = split_planetoid_data(dataset_str, self.labels.size(0))
        elif dataset_str in ['amaphoto', 'amacomp']:  # in ['amaphoto', 'amacomp']
            train_mask, val_mask, test_mask = split_amazon_data(dataset_str, self.labels.size(0))
        elif dataset_str in ['synthetic']:
            # Split: train 70%, val 10%, test 20%
            idx = np.arange(data.num_nodes)
            idx_train_val, idx_test = train_test_split(
                idx, test_size=0.2, stratify=data.y.cpu().numpy(), random_state=seed
            )
            idx_train, idx_val = train_test_split(
                idx_train_val, test_size=0.125, stratify=data.y[idx_train_val].cpu().numpy(), random_state=seed
            )

            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            print(train_mask.shape)
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            train_mask[idx_train] = True
            val_mask[idx_val] = True
            test_mask[idx_test] = True
            
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = int(torch.max(self.labels)) + 1

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        super().to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


class LinkPredData(Data):
    def __init__(self, dataset_str, val_ratio=0.05, test_ratio=0.1, seed=None):
        super(LinkPredData, self).__init__(dataset_str)
        np.random.seed(seed)
        train_edges, val_edges, test_edges = split_edges(self.G, val_ratio, test_ratio)
        adjmat = torch.tensor(nx.to_numpy_matrix(self.G) + np.eye(self.features.size(0))).float()
        negative_edges = torch.stack(torch.where(adjmat == 0))

        # Update edge_list and adj to train edge_list, adj, and adjmat
        edge_list = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
        self.num_nodes = self.G.number_of_nodes()
        self.edge_list = add_self_loops(edge_list, self.num_nodes)
        self.adj = normalize_adj(self.edge_list)
        self.adjmat = torch.where(self.adj.to_dense() > 0, torch.tensor(1.), torch.tensor(0.))

        neg_idx = np.random.choice(negative_edges.size(1), val_edges.size(1) + test_edges.size(1), replace=False)

        self.val_edges = val_edges
        self.neg_val_edges = negative_edges[:, neg_idx[:val_edges.size(1)]]
        self.test_edges = test_edges
        self.neg_test_edges = negative_edges[:, neg_idx[val_edges.size(1):]]

        # For Link Prediction Training
        N = self.features.size(0)
        E = self.edge_list.size(1)
        self.pos_weight = torch.tensor((N * N - E) / E)
        self.norm = (N * N) / ((N * N - E) * 2)

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        super().to(device)
        self.val_edges = self.val_edges.to(device)
        self.neg_val_edges = self.neg_val_edges.to(device)
        self.test_edges = self.test_edges.to(device)
        self.neg_test_edges = self.neg_test_edges.to(device)
        self.adjmat = self.adjmat.to(device)


def load_planetoid_data(dataset_str):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (cora, citeseer)

    Returns
    -------
    data: dict

    """
    names = ['tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.tensor(out))

    tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1), dtype=torch.int)
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    G = nx.from_dict_of_lists(graph)
    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels,
        'G': G,
    }
    return data


def load_amazon_data(dataset_str):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (amaphoto, amacomp)

    Returns
    -------
    data: dict

    """
    with np.load('data/amazon/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)

    feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                shape=loader['attr_shape']).todense()
    features = torch.tensor(feature_mat)

    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edges = [(u, v) for u, v in zip(adj_mat.row.tolist(), adj_mat.col.tolist())]
    G = nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))
    G.add_edges_from(edges)

    edges = torch.tensor([[u, v] for u, v in G.edges()]).t()
    edge_list = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
    adj = normalize_adj(edge_list)

    labels = loader['labels']
    labels = torch.tensor(labels).long()

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels,
        'G': G,
    }
    return data


def split_planetoid_data(dataset_str, num_nodes):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (cora, citeseer)
    num_nodes: int
        the number of nodes

    Returns
    -------
    train_mask: torch.tensor
    val_mask: torch.tensor
    test_mask: torch.tensor

    """
    with open("data/planetoid/ind.{}.y".format(dataset_str), 'rb') as f:
        y = torch.tensor(pkl.load(f, encoding='latin1'))
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)
    return train_mask, val_mask, test_mask


def split_amazon_data(dataset_str, num_nodes):
    with np.load('data/amazon/' + dataset_str + '_mask.npz', allow_pickle=True) as masks:
        train_idx, val_idx, test_idx = masks['train_idx'], masks['val_idx'], masks['test_idx']
    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)
    return train_mask, val_mask, test_mask


def split_edges(G, val_ratio, test_ratio):
    edges = np.array([[u, v] for u, v in G.edges()])
    np.random.shuffle(edges)
    E = edges.shape[0]
    n_val_edges = int(E * val_ratio)
    n_test_edges = int(E * test_ratio)
    val_edges = torch.LongTensor(edges[:n_val_edges]).t()
    test_edges = torch.LongTensor(edges[n_val_edges: n_val_edges + n_test_edges]).t()
    train_edges = torch.LongTensor(edges[n_val_edges + n_test_edges:]).t()
    return train_edges, val_edges, test_edges


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.adjacency_matrix(G).tocoo()
    # coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row = edge_list[0]
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

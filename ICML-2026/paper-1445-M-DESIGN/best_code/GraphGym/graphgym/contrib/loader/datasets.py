import torch
from torch_geometric.transforms import ToUndirected, AddRandomWalkPE, AddLaplacianEigenvectorPE
import os.path as osp
import numpy as np
import pickle

from .neigh import get_multihop, get_knn
from .lepe import MyAddLaplacianEigenvectorPE


def add_node_attr_nc(data, dataset_dir=None, rw_dim=16, le_dim=16):
    if dataset_dir and osp.exists(osp.join(dataset_dir, 'node_attr_dict.pkl')):
        with open(osp.join(dataset_dir, 'node_attr_dict.pkl'), 'rb') as f:
            return pickle.load(f)

    data = ToUndirected()(data)

    rwpe = AddRandomWalkPE(walk_length=rw_dim)
    lepe = AddLaplacianEigenvectorPE(k=le_dim, is_undirected=True)
    data = rwpe(data)
    data = lepe(data)
    data.avg_degree = int(np.rint(data['edge_index'].size(1) / data['x'].size(0)))

    node_attr_dict = {
        'random_walk_pe': data.random_walk_pe,
        'laplacian_eigenvector_pe': data.laplacian_eigenvector_pe,
        'avg_degree': data.avg_degree,
        'edge_index_2hop': get_multihop(data, kmax=2, prune=True)[-1],
        'edge_index_knn': get_knn(data, attr='x'),
        'edge_index_knn_rwpe': get_knn(data, attr='random_walk_pe'),
        'edge_index_knn_lepe': get_knn(data, attr='laplacian_eigenvector_pe')
    }
    
    if dataset_dir:
        with open(osp.join(dataset_dir, 'node_attr_dict.pkl'), 'wb') as f:
            pickle.dump(node_attr_dict, f)

    return node_attr_dict


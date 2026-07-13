import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os.path as osp
try:
    import dgl
except ImportError:
    pass
import yaml
import os
import json
from scipy.sparse import coo_matrix
from torch_geometric.datasets import DBP15K
try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    pass

    

def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()

            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")




def load_douban(data_path):
    file_path = os.path.join(data_path, "douban.mat")
    x = loadmat(file_path)

    return (x['online_edge_label'][0][1],
            x['online_node_label'],
            x['offline_edge_label'][0][1],
            x['offline_node_label'],
            x['ground_truth'].T)



def load_ogbn_arxiv(root='./data'):
    """
    Load OGBN-Arxiv dataset and convert to required format.
    
    Returns:
        adj: Adjacency matrix as torch tensor
        features: Node features
        num_nodes: Number of nodes
    """
    print("Loading OGBN-Arxiv dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    
    # Convert to adjacency matrix
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Create symmetric adjacency matrix
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    
    # Make undirected
    row_all = np.concatenate([row, col])
    col_all = np.concatenate([col, row])
    values = np.ones(len(row_all))
    
    adj_sparse = coo_matrix((values, (row_all, col_all)), shape=(num_nodes, num_nodes))
    adj_dense = torch.from_numpy(adj_sparse.toarray()).int()
    
    # Remove self-loops and duplicates
    adj_dense = adj_dense - torch.diag(torch.diag(adj_dense))
    # adj_dense = (adj_dense > 0).int()
    
    # Node features
    # features = data.x.float()
    
    
    return adj_dense

 
def load_adj(dataset, data_path="../data"):
    if dataset == "celegans":
        S = torch.load(os.path.join(data_path, "celegans.pt"))

    elif dataset == "arenas":
        S = torch.load(os.path.join(data_path, "arenas.pt"))

    elif dataset == "douban":
        S = torch.load(os.path.join(data_path, "douban.pt"))

    elif dataset == "Online":
        S = torch.load(os.path.join(data_path, "online.pt"))

    elif dataset == "Offline":
        S = torch.load(os.path.join(data_path, "offline.pt"))

    elif dataset == "ACM":
        S = torch.load(os.path.join(data_path, "ACM.pt"))

    elif dataset == "DBLP":
        S = torch.load(os.path.join(data_path, "DBLP.pt"))

    elif dataset == "ogbn_arxiv":
        S = load_ogbn_arxiv(data_path)

    else:
        filepath = os.path.join(data_path, f"{dataset}.npz")
        loader = load_npz(filepath)

        data = loader["adj_matrix"]
        samples = data.shape[0]
        features = data.shape[1]

        values = data.data
        coo_data = data.tocoo()

        indices = torch.LongTensor([coo_data.row, coo_data.col])
        S = torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples, features]).to_dense()

        if not torch.all(S.transpose(0, 1) == S):
            S = S + S.transpose(0, 1)

        S = S.int()
        S = torch.where(S > 1, torch.ones_like(S), S)

    return S






def import_data(data, device, data_path="../data"):
    train_features = {}
    train_adj = {}
    if (data == "ACM_DBLP"):
        modals_name = ["ACM", "DBLP"]
        
        file_path = os.path.join(data_path, "ACM-DBLP.npz")
        b = np.load(file_path)
        
        train_features["ACM"] = [torch.from_numpy(b["x1"]).float()]
        train_features["DBLP"] = [torch.from_numpy(b["x2"]).float()]
        
        test_pairs = b['test_pairs'].astype(np.int32)
        for dataset in modals_name:
            train_adj[dataset] = [load_adj(dataset)]
                    
        
    elif (data == "Douban Online_Offline"):
        modals_name = ["Online", "Offline"]

        a1, f1, a2, f2, test_pairs = load_douban(data_path)
        f1 = f1.A
        f2 = f2.A
        test_pairs = torch.tensor(np.array(test_pairs, dtype=int)) - 1
        test_pairs = test_pairs.numpy()
        train_features["Online"] = [torch.from_numpy(f1).float()]
        train_features["Offline"] = [torch.from_numpy(f2).float()]
        # Use adjacency matrices directly from load_douban
        train_adj["Online"] = [torch.tensor(a1.todense(), dtype=torch.int32)]
        train_adj["Offline"] = [torch.tensor(a2.todense(), dtype=torch.int32)]
     
    return train_features, train_adj, test_pairs, modals_name
        
        
        


def load_config(config_path, dataset, args):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if dataset not in config:
        raise ValueError(f"Dataset {dataset} not found in config file.")

    dataset_config = config[dataset]
    for key, value in dataset_config.items():
        if getattr(args, key) is None:  # Only set if not already passed via CLI
            setattr(args, key, value)
    return args, config


 
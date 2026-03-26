import os
import numpy as np
import torch
import pickle
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, to_undirected
from torch_geometric.utils import degree

class Dataset(Dataset):
    def __init__(self, data_dir, dtype=torch.float16, num_samples=-1):
        super(Dataset, self).__init__()
        self.dtype = dtype
        self.files = glob.glob(os.path.join(data_dir,'*gpickle'))
        if num_samples>0:
            self.files = self.files[:num_samples]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        G = nx.read_gpickle(self.files[idx])
        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)
        degrees = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
        data.b = -degrees
        data.idx = idx
        return data

    
class Problem(object):
    def __init__(self, beta):
        self.beta = beta

    def energy_func(self, A, b, x, compute_grad):
        L = A@x
        if compute_grad:
            grad = 2*L+b
        else:
            grad = None
        loss = torch.sum(x*(L+b),dim=0)
        return loss, grad


    def decode_func(self, par, graph):
        A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
        batch_idx = torch.arange(par.shape[1], device=par.device).long()
        energy, grad = self.energy_func(A, graph.b, par, True)
        best_energy = energy.clone()
        delta = grad*(2*par-1)/2
         
        while not (delta<=0).all():
            value, index = torch.max(delta, dim=0)
            par[index, batch_idx] = torch.where(value>0, 1-par[index, batch_idx], par[index, batch_idx])
            best_energy, grad = self.energy_func(A, graph.b, par, True)
            delta = grad*(2*par-1)/2
        return best_energy

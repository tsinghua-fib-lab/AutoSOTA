import os
import numpy as np
import torch
import pickle
import glob
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, to_undirected

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
        data.edge_weight = -torch.ones_like(data.edge_index[0],  dtype=self.dtype)/2      
        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data

    
class Problem(object):
    def __init__(self, beta):
        self.beta = beta
        
    def energy_func(self, A, b, x, compute_grad):
        L = A@x
        set_size = torch.sum(x,dim=0)
        loss = (self.beta*set_size**2-torch.sum(x**2, dim=0))/2+torch.sum(x*(self.beta*L+b),dim=0)

        if compute_grad:
            grad = self.beta*(set_size-x)+2*self.beta*L+b
        else:
            grad = None

        return loss, grad
        
    def decode_func(self, par, graph):
        # par: shape of [n_nodes, n_solutions]
        # graph.edge_index: [n_nodes, n_nodes], adjacency matrix
        par = par.transpose(0, 1)
        mask = torch.ones_like(par).bool()
        solution = torch.zeros_like(par)

        while mask.any():
            prob = torch.where(mask, par, -torch.inf)
            next_node = torch.argmax(prob, 1)
            indices = torch.arange(par.shape[0]).to(par.device)
            flag = mask[indices, next_node]
            indices = indices[flag]
            next_node = next_node[flag]
            nnx_masks = graph.edge_index[0].unsqueeze(0) == next_node.unsqueeze(1)
            nnx_indices = torch.nonzero(nnx_masks).T # nnx_indices[0]: sol_number, nnx_indices[1]: neighbor_node_index

            temp = torch.zeros_like(solution[indices])
            temp[nnx_indices[0], graph.edge_index[1][nnx_indices[1]]] = 1
            feasible = torch.all(solution[indices] <= temp, dim=1) 
            solution[indices[feasible], next_node[feasible]] = 1

            # Remove selected node from further consideration
            mask[indices[feasible]] = mask[indices[feasible]] & temp.bool() 
            mask[indices, next_node] = False

        return -solution.sum(1)


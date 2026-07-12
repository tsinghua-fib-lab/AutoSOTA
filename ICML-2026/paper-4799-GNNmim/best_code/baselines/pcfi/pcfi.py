"""
Copyright 2023 Daeho Um
SPDX-License-Identifier: Apache-2.0
"""
import random
import numpy as np
import torch
import torch_geometric.utils
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add


def pcfi(edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    propagation_model = PCFI(num_iterations=num_iterations, alpha = alpha, beta=beta)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask, mask_type=mask_type)

class PCFI(torch.nn.Module):
    def __init__(self, num_iterations: int, alpha: float, beta: float):
        super(PCFI, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor, mask_type: str, edge_weight: OptTensor = None) -> Tensor:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        nv = x.shape[0]
        feat_dim = x.shape[1]
        out = x
        if mask_type == 'structural':
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type)
            adj_c = self.compute_edge_weight_c(edge_index, f_n2d, nv)
            if mask is not None:
                out = torch.zeros_like(x)
                out[mask] = x[mask]

            for _ in range(self.num_iterations):
                # Diffuse current features
                out = torch.sparse.mm(adj_c, out)
                out[mask] = x[mask]
            f_n2d = f_n2d.repeat(feat_dim,1)
        else:
            out = torch.zeros_like(x)
            if mask is not None:
                out[mask] = x[mask]
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, feat_dim)
            print('\n ==== propagation on {feat_dim} channels ===='.format(feat_dim=feat_dim))
            for i in range(feat_dim):
                adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], nv)
                for _ in range(self.num_iterations):
                    out[:,i] = torch.sparse.mm(adj_c, out[:,i].reshape(-1,1)).reshape(-1)
                    out[mask[:,i],i] = x[mask[:,i],i]
        cor = torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
        f_n2d = f_n2d.to(out.device)
        a_1 = (self.alpha ** f_n2d.T) * (out - torch.mean(out, dim=0))
        a_2 = torch.matmul(a_1, cor)
        out_1 = self.beta * (1 - (self.alpha ** f_n2d.T)) * a_2
        out = out + out_1
        return out

    def compute_f_n2d(self, edge_index, feature_mask, mask_type, feat_dim: OptTensor = None):
        nv = feature_mask.shape[0]
        if mask_type == 'structural':
            len_v_0tod_list = []
            f_n2d = torch.zeros(nv, dtype = torch.int)
            v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
            len_v_0tod_list.append(len(v_0))
            v_0_to_now = v_0
            f_n2d[v_0] = 0
            d = 1
            while True:
                v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                if len(v_d) == 0:
                    break
                f_n2d[v_d] = d
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                len_v_0tod_list.append(len(v_d))
                d += 1
        else:
            f_n2d = torch.zeros(feat_dim, nv)
            print('\n ==== compute f_n2d for {feat_dim} channels ===='.format(feat_dim=feat_dim))
            # for i in tqdm(range(feat_dim), mininterval=2):
            for i in range(feat_dim):
                v_0 = torch.nonzero(feature_mask[:,i]).view(-1)
                v_0_to_now = v_0
                f_n2d[i, v_0] = 0
                d=1
                while True:
                    v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
            print('\n ====== f_n2d is computed ======'.format(feat_dim=feat_dim))
        return f_n2d

    # Original
    def compute_edge_weight_c(self, edge_index, f_n2d, n_nodes):
        # edge_weight_c = torch.zeros(edge_index.shape[1], device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        # row: destination, col: source
        d_row = f_n2d[row]
        d_col = f_n2d[col]
        edge_weight_c = (self.alpha ** (d_col - d_row + 1)).to(edge_index.device)
        deg_W = scatter_add(edge_weight_c, row, dim_size= f_n2d.shape[0])
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse.FloatTensor(edge_index, values= A_Dinv, size=[n_nodes, n_nodes]).to(edge_index.device)

        return adj



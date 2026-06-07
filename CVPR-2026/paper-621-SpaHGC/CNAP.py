'''
code is based on https://github.com/Kevinz-code/CSRA

'''


import torch_geometric
import torch
import torch_geometric.nn as pyg
from torch_geometric.nn.dense.linear import Linear
from torch.nn import functional as F
from torch_geometric.utils.softmax import softmax
from torch_scatter import scatter_add


class CNAP(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, dim, dropout=0.0):
        super(CNAP, self).__init__()


        self.head = pyg.Linear(dim, dim)
        self.aggr = 'sum'
        self.dropout = dropout
        self.dim = dim
        heads = 4
        self.heads = heads
        self.scale = (dim // heads) ** 0.5

        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)

    def forward(self, x_dict, edge_index_dict):
        reference = x_dict['reference']
        target = self.head(x_dict['target'])

        edge_index = edge_index_dict[('reference', 'refer', 'target')]

        self.q = self.q_proj(target)
        self.k = self.k_proj(reference)
        self.v = self.v_proj(reference)

        q = self.q.view(-1, self.heads, self.dim // self.heads)
        k = self.k.view(-1, self.heads, self.dim // self.heads)
        v = self.v.view(-1, self.heads, self.dim // self.heads)

        src, dst = edge_index
        q_i = q[dst]
        k_j = k[src]
        v_j = v[src]

        attn = (q_i * k_j).sum(dim=-1) / self.scale
        attn = F.leaky_relu(attn, negative_slope=0.2)
        attn = softmax(attn, index=dst)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn.unsqueeze(-1) * v_j
        out = out.view(-1, self.dim)
        out_dist = scatter_add(out, dst, dim=0, out=torch.zeros(target.size(0), self.dim, device=out.device))


        return target +  out_dist




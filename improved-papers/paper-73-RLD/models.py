import torch
from torch import nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(MLP, self).__init__()   
        self.initial_project = nn.Linear(in_channels, hidden_channels)
        self.linear = nn.ModuleList()
        for i in range(num_layers):
            self.linear.append(nn.Linear(hidden_channels, hidden_channels))
        self.final_project = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        
    def reset_parameters(self):
        for linear in self.linear:
            linear.reset_parameters()
        
        self.initial_project.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x):
        x = self.initial_project(x)
        
        for linear in self.linear:
            x_ = self.activation(linear(x))
            x = x_ + x

        return self.final_project(x)
            

def edge2adj(edge_index, num_nodes):
    """Convert edge index to adjacency matrix"""
    tmp, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(tmp.size(1), dtype=None,
                                     device=edge_index.device)

    row, col = tmp[0], tmp[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(tmp, edge_weight, torch.Size((num_nodes, num_nodes))).to_sparse_csr()

class GCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        support = support.flatten(1)
        output = torch.spmm(adj, support)
        output = output.reshape(x.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(GCN, self).__init__()   
        self.initial_project = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()  
        self.linears = nn.ModuleList()
        
        for i in range(num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))       
        
        self.final_project = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for linear in self.linears:
            linear.reset_parameters()
        
        self.initial_project.reset_parameters()
        self.final_project.reset_parameters()
            
    def forward(self, x, edge_index): 
        adj = edge2adj(edge_index, x.shape[0])
        
        x = self.initial_project(x)
        for conv, linear in zip(self.convs,self.linears):
            nei = conv(x, adj)
            x_ = linear(x)
            x = self.activation(nei + x_) + x
        
        x = self.final_project(x)
        
        return x     

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fc
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, get_laplacian

class MyGCNConv(MessagePassing):
    """Custom GCN Convolutional Layer."""
    def __init__(self, in_channels, out_channels):
        super(MyGCNConv, self).__init__(aggr='add', node_dim=0)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_in = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, edge_index, edge_weight=None, norm=1):
        T = self.lin_in(x) + self.propagate(edge_index, x=self.lin_neigh(x),
                                            norm=norm, edge_weight=edge_weight)
        return T + self.bias[None, :]

    def message(self, x_j, edge_weight=None, norm=1):
        if edge_weight is not None:
            ew = norm * edge_weight
            return x_j * ew[:, None]
        if np.isscalar(norm):
            return norm * x_j
        else:
            return x_j * norm[:, None]

class MyGCN(nn.Module):
    """Standard GCN Architecture."""
    def __init__(self, dims, nonlin=Fc.relu, variant='inv', normalization='laplacian'):
        super(MyGCN, self).__init__()
        self.dims = dims
        self.num_layers = len(dims) - 2
        self.variant = variant
        self.normalization = normalization
        self.nonlin = nonlin

        self.layers = nn.ModuleList([MyGCNConv(dims[i], dims[i+1]) for i in range(self.num_layers)])
        self.output_lin = nn.Linear(dims[-2], dims[-1], bias=True)

    def forward(self, x, edge_index, edge_weight=None):
        # Apply Laplacian normalization if specified
        if self.normalization == 'laplacian':
            edge_id, norm = get_laplacian(edge_index, edge_weight, normalization='sym', num_nodes=x.shape[0])
            _, norm = remove_self_loops(edge_id, norm)
            norm = -norm

        for i in range(self.num_layers):
            x = self.nonlin(self.layers[i].forward(x, edge_index, norm=norm, edge_weight=edge_weight))

        x = self.output_lin(x)
        
        if self.variant == 'equi':
            return x
        else:
            return x.mean(axis=0)
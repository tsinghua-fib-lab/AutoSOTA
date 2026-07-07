

import torch
from functools import partial
from torch_geometric.nn import global_add_pool
from fastargs.decorators import param
from models.gcn_conv import GCNConv
from torch.nn import BatchNorm1d,LayerNorm

import torch.nn.functional as F
import pdb
from torch_geometric.nn.pool import global_mean_pool

class GCNPYG(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       hidden, num_conv_layers=3,
                       num_feat_layers=1, gfn=False, collapse=False, residual=False,
                       res_branch="BNConvReLU", dropout=0, 
                       edge_norm=True):
        super(GCNPYG, self).__init__()
        self.hidden_dim = hidden
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)#BN层很重要
        #self.bn_feat = LayerNorm(hidden_in)

        self.conv_feat = GCNConv(hidden_in, hidden, gfn=False) # linear transform #不要线性
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            if i:

                self.bns_conv.append(BatchNorm1d(hidden))
                #self.bns_conv.append(LayerNorm(hidden))
                self.convs.append(GConv(hidden, hidden))
            else:
                self.bns_conv.append(BatchNorm1d(hidden_in))
                #self.bns_conv.append(LayerNorm(hidden))
                self.convs.append(GConv(hidden_in, hidden))


        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
    
    def forward(self, data, edge_weight=None):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        #h = self.bn_feat(x)
        #h = F.leaky_relu(self.conv_feat(h, edge_index, edge_prompt = edge_prompt))
        h = x
        for i, conv in enumerate(self.convs):
            h = self.bns_conv[i](h)
            # h = F.relu(conv(h, edge_index))
            h = F.leaky_relu(conv(h, edge_index, edge_weight = edge_weight))
        
       

        return h

def get_model(num_features, hid_dim, num_conv_layers, dropout):
    return GCNPYG(num_features=num_features, hidden=hid_dim, num_conv_layers=num_conv_layers, dropout=dropout, gfn=False)


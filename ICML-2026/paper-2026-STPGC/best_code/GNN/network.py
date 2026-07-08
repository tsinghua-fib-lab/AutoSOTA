import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,APPNP
from torch.nn import ModuleList
from gcn_lib.sparse.torch_nn import norm_layer
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
#         super(GCN, self).__init__()
#         self.convs = ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, out_channels))
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x

class GCN_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t,val_test):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=not val_test)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
class net_gcn(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = torch.nn.ModuleList([GCNConv(embedding_dim[ln], embedding_dim[ln+1]) for ln in range(self.layer_num)])
        self.norm_layer = torch.nn.ModuleList([norm_layer('batch', embedding_dim[ln]) for ln in range(self.layer_num)])
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, x, edge_index, val_test=False):
        #adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            x = self.norm_layer[ln](x)
            x = self.net_layer[ln](x, edge_index)
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class APPNP_Net(torch.nn.Module):
    def __init__(self, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = torch.nn.Linear(args.num_features, args.hidden)
        self.lin2 = torch.nn.Linear(args.hidden,  args.num_classes)
        self.prop1 = APPNP(20, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1)


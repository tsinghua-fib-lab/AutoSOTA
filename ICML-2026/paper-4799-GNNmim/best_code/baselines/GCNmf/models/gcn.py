"""
version 1.0
date 2021/02/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5):
        super(GCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x

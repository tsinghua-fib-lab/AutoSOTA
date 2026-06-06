import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

class GCN(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature, hidden)
        self.conv2 = GCNConv(hidden, out_feature)
        self.dropout_rate = dropout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.sigmoid(x)
        return x.squeeze(), x.squeeze(), x.squeeze()


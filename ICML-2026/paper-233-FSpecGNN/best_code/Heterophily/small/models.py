import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, ChebConv
from Propagations import *


class Cheb_GAT(torch.nn.Module):
    """
    lin1 -> lin2 -> ChebConv -> (h + sigmoid(alpha) * GAT(h)) -> ChebConv
    """
    def __init__(self, dataset, args):
        super().__init__()
        hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.alpha_init = float(getattr(args, "alpha_init", -6.0))

        self.lin1 = nn.Linear(dataset.num_features, hidden)
        self.lin2 = nn.Linear(hidden, dataset.num_classes)

        self.conv1 = ChebConv(dataset.num_classes, dataset.num_classes, K=args.K)
        self.gat = GATConv(dataset.num_classes, dataset.num_classes, heads=args.heads, concat=False, dropout=self.dropout)
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        self.conv2 = ChebConv(dataset.num_classes, dataset.num_classes, K=args.K)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.gat.reset_parameters()
        self.alpha.data.fill_(self.alpha_init)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        h = F.dropout(h, p=self.dprate, training=self.training)
        h = self.conv1(h, edge_index)
        g = self.gat(h, edge_index)
        scale = torch.sigmoid(self.alpha)
        h = h + scale * g

        h = F.dropout(h, p=self.dprate, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)


class Bern_GAT(torch.nn.Module):
    """
    lin1 -> lin2 -> Bern_prop -> (h + sigmoid(alpha) * GAT(h)) -> Bern_prop
    """
    def __init__(self, dataset, args):
        super().__init__()

        hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.alpha_init = float(getattr(args, "alpha_init", -6.0))

        self.lin1 = nn.Linear(dataset.num_features, hidden)
        self.lin2 = nn.Linear(hidden, dataset.num_classes)
        self.prop1 = Bern_prop(K=args.K)
        self.gat = GATConv(dataset.num_classes, dataset.num_classes, heads=args.heads, concat=False, dropout=self.dropout)
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        self.prop2 = Bern_prop(K=args.K)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prop1.reset_parameters()
        self.prop2.reset_parameters()

        self.gat.reset_parameters()

        self.alpha.data.fill_(self.alpha_init)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        h = F.dropout(x, p=self.dprate, training=self.training)
        h = self.prop1(h, edge_index)
        g = self.gat(h, edge_index)
        scale = torch.sigmoid(self.alpha)
        h = h + scale * g

        h = F.dropout(h, p=self.dprate, training=self.training)
        h = self.prop2(h, edge_index)
        return F.log_softmax(h, dim=1)


class ChebII_GAT(torch.nn.Module):
    """
    lin1 -> lin2 -> ChebII_prop -> (h + sigmoid(alpha) * GAT(h)) -> ChebII_prop
    """
    def __init__(self, dataset, args):
        super().__init__()

        hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.alpha_init = float(getattr(args, "alpha_init", -6.0))

        self.lin1 = nn.Linear(dataset.num_features, hidden)
        self.lin2 = nn.Linear(hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(K=args.K)
        self.gat = GATConv(dataset.num_classes, dataset.num_classes, heads=args.heads, concat=False, dropout=self.dropout)
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        self.prop2 = ChebnetII_prop(K=args.K)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prop1.reset_parameters()
        self.prop2.reset_parameters()

        self.gat.reset_parameters()

        self.alpha.data.fill_(self.alpha_init)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        h = F.dropout(x, p=self.dprate, training=self.training)
        h = self.prop1(h, edge_index)
        g = self.gat(h, edge_index)
        scale = torch.sigmoid(self.alpha)
        h = h + scale * g

        h = F.dropout(h, p=self.dprate, training=self.training)
        h = self.prop2(h, edge_index)
        return F.log_softmax(h, dim=1)

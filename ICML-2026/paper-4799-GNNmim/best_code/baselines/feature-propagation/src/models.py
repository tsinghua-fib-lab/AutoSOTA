"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import (
    SGConv,
    SAGEConv,
    GCNConv,
    GATConv,
    ChebConv,
    JumpingKnowledge,
)
from torch_geometric.nn.models import LabelPropagation

from baselines.gcn_mf import GCNmfConv
from baselines.pa_gnn import PaGNNConv


def get_model(model_name, num_features, num_classes, edge_index, x, args, mask=None):
    if model_name in ["mlp", "cs"]:
        return MLP(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif model_name == "sgc":
        return SGC(
            num_features=num_features,
            num_classes=num_classes,
            K=args.num_layers,
            cached=args.filling_method not in ["parameterization", "learnable_diffusion"],
        )
    elif model_name in ["sage", "gcn", "gat"]:
        return GNN(
            num_features=num_features,
            num_classes=num_classes,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            conv_type=model_name,
            jumping_knowledge=args.jk,
        )
    elif model_name == "gcnmf":
        return GCNmf(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            x=x,
            edge_index=edge_index,
            dropout=args.dropout,
        )
    elif model_name == "pagnn":
        return PaGNN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            mask=mask,
            edge_index=edge_index,
        )
    elif model_name == "lp":
        return LabelPropagation(num_layers=50, alpha=args.lp_alpha)
    else:
        raise ValueError(f"Model {model_name} not supported")


class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=2, cached=False):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=K, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, conv_type="GCN", jumping_knowledge=False,
    ):
        super(GNN, self).__init__()

        self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        if jumping_knowledge:
            self.lin = Linear(hidden_dim, num_classes)
            self.jump = JumpingKnowledge(mode="max", channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge

    def forward(self, x, edge_index=None, adjs=None, full_batch=True):
        return self.forward_full_batch(x, edge_index) if full_batch else self.forward_sampled(x, adjs)

    def forward_full_batch(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)

    def forward_sampled(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=1)

    def inference(self, x_all, inference_loader, device):
        """Get embeddings for all nodes to be used in evaluation"""

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.lins = ModuleList([Linear(num_features, hidden_dim)])
        self.bns = ModuleList([BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
        self.lins.append(Linear(hidden_dim, num_classes))

    def forward(self, x, edge_index):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = bn(lin(x).relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return torch.nn.functional.log_softmax(x, dim=1)


class GCNmf(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, x, edge_index, dropout=0, n_components=5,
    ):
        super(GCNmf, self).__init__()
        self.gc1 = GCNmfConv(
            in_features=num_features,
            out_features=hidden_dim,
            x=x,
            edge_index=edge_index,
            n_components=n_components,
            dropout=dropout,
        )
        self.gc2 = GCNConv(hidden_dim, num_classes, dropout)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class PaGNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, mask=None, edge_index=None,
    ):
        super(PaGNN, self).__init__()
        # NOTE: It not specified in their paper (https://arxiv.org/pdf/2003.10130.pdf), but the only way for their model to work is to have only the first layer
        # to be what they describe and the others to be standard GCN layers. Otherwise, the feature matrix would change dimensionality, and it couldn't be
        # multiplied elmentwise with the mask anymore
        self.convs = ModuleList([PaGNNConv(num_features, hidden_dim, mask, edge_index)])
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.convs[-1](x, edge_index)

        return torch.nn.functional.log_softmax(out, dim=1)


def get_conv(conv_type, input_dim, output_dim):
    if conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gcn":
        return GCNConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "cheb":
        return ChebConv(input_dim, output_dim, K=4)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

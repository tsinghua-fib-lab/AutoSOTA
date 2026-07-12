"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    JumpingKnowledge,
)



def get_model(model_name, num_features, num_classes, edge_index, x, args, mask=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return GNN(
        num_features=num_features,
        num_classes=num_classes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        conv_type=model_name,
        jumping_knowledge=args.jk,
    )




class GNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=3, dropout=0, conv_type="GCN", jumping_knowledge=False,
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

def get_conv(conv_type, input_dim, output_dim):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True)
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
# -----------------------------------------------------------------------------#
# Models: GCN/Sage encoder, node classifier, link predictor
# -----------------------------------------------------------------------------#

import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GPSConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import ( Planetoid,
                                       WebKB,
                                       Actor,
                                       HeterophilousGraphDataset,
                                       Airports, )
from torch_geometric.utils import to_undirected, to_networkx

# -----------------------------------------------------------------------------#
# Encoders
# -----------------------------------------------------------------------------#
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5,):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs[0] = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5,):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        else:
            self.convs[0] = SAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -----------------------------------------------------------------------------#
# Models
# -----------------------------------------------------------------------------#
class NodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, enc="GCN"):
        super().__init__()

        if enc == "SAGE":
            self.encoder = SAGEEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else: # Fallback to GCN
            self.encoder = GCNEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.classifier(z)

class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src, dst = edge_pairs
        h = torch.cat([z[src], z[dst]], dim=-1)
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = self.lin2(h).squeeze(-1)
        return torch.sigmoid(h)

class LinkModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.5, enc="GCN"):
        super().__init__()

        if enc == "SAGE":
            self.encoder = SAGEEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else: # Fallback to GCN
            self.encoder = GCNEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.lp_head = LinkPredictor(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index_train: torch.Tensor, edge_pairs: torch.Tensor,) -> torch.Tensor:
        z = self.encoder(x, edge_index_train)
        return self.lp_head(z, edge_pairs)

# -----------------------------------------------------------------------------#
# Joint Head: shared encoder + NC head + LP head
# -----------------------------------------------------------------------------#
class JointNC_LP(nn.Module):
    """
    Shared encoder producing node embeddings z.
    NC head: linear classifier over z.
    LP head: LinkPredictor over z for edge pairs.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, num_layers: int = 2,
                 dropout: float = 0.5, lp_hidden: Optional[int] = None, enc="GCN"):
        super().__init__()

        if enc == "SAGE":
            self.encoder = SAGEEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )
        else: # Fallback to GCN
            self.encoder = GCNEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.nc_head = nn.Linear(hidden_channels, num_classes)
        self.lp_head = LinkPredictor(hidden_channels, lp_hidden or hidden_channels)

    """
    Returns:
        z: (N, hidden)
        nc_logits: (N, C)
        lp_scores: (E,) sigmoid scores if edge_pairs provided else None
    """
    def forward(self, x: torch.Tensor, edge_index_train: torch.Tensor, edge_pairs: Optional[torch.Tensor] = None,) \
            -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z = self.encoder(x, edge_index_train)
        nc_logits = self.nc_head(z)

        lp_scores = None
        if edge_pairs is not None:
            lp_scores = self.lp_head(z, edge_pairs)

        return z, nc_logits, lp_scores

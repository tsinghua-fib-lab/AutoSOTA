import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=3, dropout=0.3, use_bn=True):
        super().__init__()
        self.dropout = dropout
        self.input = GCNConv(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.stack = nn.ModuleList()
        self.bn_mid = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.stack.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_mid.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
        self.out = GCNConv(hidden_dim, embed_dim)
        self.bn_out = nn.BatchNorm1d(embed_dim) if use_bn else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input(x, edge_index, edge_weight)
        x = self.bn_in(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        for conv, bn in zip(self.stack, self.bn_mid):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        z = self.out(x, edge_index, edge_weight)
        z = self.bn_out(z)
        z = F.relu(z)
        return z


class Classifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim, dropout=0.3, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.pool_dim = embed_dim * 2 if pooling == "mean+max" else embed_dim
        self.head = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_dim),
        )

    def readout(self, z, batch):
        if self.pooling == "mean":
            return global_mean_pool(z, batch)
        if self.pooling == "max":
            return global_max_pool(z, batch)
        return torch.cat([global_mean_pool(z, batch), global_max_pool(z, batch)], dim=1)

    def forward(self, z, batch):
        g = self.readout(z, batch)
        return self.head(g), g


class GCNForCL(nn.Module):
    def __init__(self, in_dim, hidden, embed, out_dim=2, layers=4, dropout=0.3, use_bn=True, pooling="mean+max"):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden, embed, layers, dropout, use_bn)
        self.classifier = Classifier(embed, hidden, out_dim, dropout, pooling)
        self.pool_dim = self.classifier.pool_dim

    def forward_all(self, x, edge_index, batch, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        logits, g = self.classifier(z, batch)
        return logits, g


def build_model(cfg, in_dim: int) -> GCNForCL:
    device = torch.device(cfg.DEVICE)
    return GCNForCL(
        in_dim,
        cfg.HIDDEN,
        cfg.EMBED,
        out_dim=2,
        layers=cfg.LAYERS,
        dropout=cfg.DROPOUT,
        use_bn=cfg.USE_BN,
        pooling=cfg.POOLING,
    ).to(device)

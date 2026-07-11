import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzBaseline(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x:
            (B, N, 4)
        """
        # particle-wise features
        x = self.mlp1(x)

        # permutation invariant pooling
        x = x.max(dim=1).values

        # classification head
        x = self.mlp2(x)

        return x


def minkowski_dot(x, y):
    """
    x, y:
        (..., 4)

    metric:
        (+,-,-,-)
    """

    return (
        x[..., 0] * y[..., 0]
        - (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    )


def minkowski_distance_squared(xi, xj):

    diff = xi - xj

    return minkowski_dot(diff, diff)

class LorentzLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x):

        """
        h:
            (B, N, H)

        x:
            (B, N, 4)
        """

        B, N, H = h.shape

        xi = x.unsqueeze(2)  # (B,N,1,4)
        xj = x.unsqueeze(1)  # (B,1,N,4)

        dij2 = minkowski_distance_squared(xi, xj)
        dij2 = dij2.unsqueeze(-1)

        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([hi, hj, dij2], dim=-1)

        m_ij = self.edge_mlp(edge_input)

        # aggregate messages
        m_i = m_ij.sum(dim=2)

        node_input = torch.cat([h, m_i], dim=-1)

        h = h + self.node_mlp(node_input)

        return h

class LorentzInvariantModel(nn.Module):

    def __init__(
        self,
        num_classes=2,
        hidden_dim=64,
        num_layers=4
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            LorentzLayer(hidden_dim)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):

        """
        x:
            (B, N, 4)
        """

        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h, x)

        # permutation invariant pooling
        h = h.mean(dim=1)

        return self.head(h)
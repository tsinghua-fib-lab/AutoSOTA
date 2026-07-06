import torch
from torch import nn
from mace_layer import MACE_layer


class MACEProteinNet(nn.Module):
    def __init__(
        self,
        F_in: int,
        n_classes: int,
        cutoff: float = 8.0,
        avg_num_neighbors: float = 3.0,
        K_edge: int = 32,
        hidden: int = 256,
        num_layers: int = 2,
        max_ell: int = 3,
        correlation: int = 3,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.hidden = hidden
        self.num_layers = num_layers

        self.node_irreps_str = f"{hidden}x0e"
        self.edge_irreps_str = f"{K_edge}x0e"
        self.n_dims_in = F_in

        self.node_embed = nn.Linear(F_in, hidden)
        self.attr_proj = nn.Identity()

        self.K_edge = K_edge
        self.rbf_gamma = 10.0
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, K_edge))

        self.layers = nn.ModuleList(
            [
                MACE_layer(
                    max_ell=max_ell,
                    correlation=correlation,
                    n_dims_in=self.n_dims_in,
                    hidden_irreps=self.node_irreps_str,
                    node_feats_irreps=self.node_irreps_str,
                    edge_feats_irreps=self.edge_irreps_str,
                    avg_num_neighbors=avg_num_neighbors,
                    use_sc=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_classes),
        )

    def edge_rbf(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.rbf_gamma * (d[:, None] - self.rbf_centers[None, :]) ** 2
        )

    def forward(self, pos, x, edge_index, edge_vec, dist, batch_idx):
        node_feats = self.node_embed(x)
        node_attrs = self.attr_proj(x)
        edge_feats = self.edge_rbf(dist)

        h = node_feats
        for layer in self.layers:
            h = layer(
                edge_vec,
                h,
                node_attrs,
                edge_feats,
                edge_index,
            )

        B = int(batch_idx.max().item()) + 1
        h_graph = torch.zeros(B, self.hidden, device=h.device).index_add_(0, batch_idx, h)
        counts = torch.bincount(batch_idx, minlength=B).clamp_min(1).float().unsqueeze(-1)
        h_graph = h_graph / counts

        logits = self.readout(h_graph)
        return logits, h

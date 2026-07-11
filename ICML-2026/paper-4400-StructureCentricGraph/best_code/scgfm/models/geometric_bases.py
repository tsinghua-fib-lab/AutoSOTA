from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class GeometricBasesModel(nn.Module):
    """Learnable geometric bases used by SCGFM.

    This is a cleaned release version of the original GWN model. The objective
    and tensor operations are kept equivalent to the experimental code.
    """

    def __init__(
        self,
        K: int,
        M: int,
        feature_dim: int = 50,
        tau: float = 1.0,
        lambda_gw: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_div: float = 1.0,
        div_margin: float = 0.5,
        num_projections: int = 50,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.K = K
        self.M = M
        self.feature_dim = feature_dim
        self.tau = tau
        self.lambda_gw = lambda_gw
        self.lambda_recon = lambda_recon
        self.lambda_div = lambda_div
        self.div_margin = div_margin
        self.num_projections = num_projections
        self.device_name = str(device)

        self.bases_param = nn.Parameter(torch.randn(size=(K, M, M), device=device))

        # Fixed random projections for deterministic GW distance computation
        max_dim = 100
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.register_buffer("gw_theta", torch.randn(max_dim, num_projections, device=device))
        self.gw_theta.data = F.normalize(self.gw_theta.data, p=2, dim=0)
        torch.set_rng_state(rng_state)
        self.decoder = nn.Sequential(
            nn.Linear(K, 2 * K),
            nn.ReLU(),
            nn.BatchNorm1d(2 * K),
            nn.Linear(2 * K, K),
            nn.ReLU(),
            nn.Linear(K, feature_dim),
        )

    @property
    def device(self) -> torch.device:
        return self.bases_param.device

    def get_normalized_bases(self) -> torch.Tensor:
        sym = (self.bases_param + self.bases_param.transpose(1, 2)) / 2
        exp_sym = torch.sigmoid(sym)
        eye = torch.eye(self.M, device=self.device).unsqueeze(0)
        return exp_sym * (1 - eye)

    def compute_sgw_standard(self, adj1: torch.Tensor, adj2: torch.Tensor) -> torch.Tensor:
        n = adj1.shape[-1]
        m = adj2.shape[-1]
        max_dim = max(n, m)
        if n < max_dim:
            adj1 = F.pad(adj1, (0, max_dim - n, 0, max_dim - n))
        if m < max_dim:
            adj2 = F.pad(adj2, (0, max_dim - m, 0, max_dim - m))

        theta = self.gw_theta[:max_dim, :]
        proj1 = torch.matmul(adj1, theta)
        proj2 = torch.matmul(adj2, theta)
        sorted1, _ = torch.sort(proj1, dim=-2)
        sorted2, _ = torch.sort(proj2, dim=-2)
        return torch.mean((sorted1 - sorted2) ** 2)

    def compute_gw_distance_for_attention(self, x: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
        _, n, _ = x.shape
        k, m, _ = bases.shape
        del k
        max_dim = max(n, m)
        if n < max_dim:
            x = F.pad(x, (0, max_dim - n, 0, max_dim - n))
        if m < max_dim:
            bases = F.pad(bases, (0, max_dim - m, 0, max_dim - m))

        theta = self.gw_theta[:max_dim, :]
        x_proj = torch.matmul(x, theta)
        bases_proj = torch.matmul(bases, theta)
        x_sorted, _ = torch.sort(x_proj, dim=1)
        bases_sorted, _ = torch.sort(bases_proj, dim=1)
        return torch.mean((x_sorted - bases_sorted) ** 2, dim=(1, 2))

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def forward(self, batch_data):
        dense_adjs = to_dense_adj(batch_data.edge_index, batch_data.batch)
        if not hasattr(batch_data, "stats"):
            raise RuntimeError("Missing graph-level 'stats' features. Run precompute_graph_statistics first.")

        num_graphs = dense_adjs.shape[0]
        batch_f = batch_data.stats.view(num_graphs, -1)
        if batch_f.shape[1] != self.feature_dim:
            batch_f = batch_f[:, : self.feature_dim]

        bases = self.get_normalized_bases()
        loss_gw_list = []
        embedding_weights = []

        for i in range(num_graphs):
            c_i = dense_adjs[i]
            mask = ~torch.all(c_i == 0, dim=1)
            c_i = c_i[mask][:, mask]
            if c_i.shape[0] == 0:
                continue

            dists_tensor = self.compute_gw_distance_for_attention(c_i.unsqueeze(0), bases)
            w_i = F.softmax(-dists_tensor / self.tau, dim=0)
            embedding_weights.append(w_i)
            bases_mixed = torch.einsum("k,kij->ij", w_i, bases)
            loss_gw_list.append(self.compute_sgw_standard(c_i, bases_mixed))

        if not loss_gw_list:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        batch_w = torch.stack(embedding_weights)
        loss_gw = torch.stack(loss_gw_list).mean()
        batch_f_hat = self.decoder(batch_w)
        loss_recon = F.mse_loss(batch_f_hat, batch_f)

        bases_flat = bases.view(self.K, -1)
        pdist = torch.cdist(bases_flat, bases_flat, p=2)
        mask = torch.triu(torch.ones_like(pdist), diagonal=1).bool()
        div_dists = pdist[mask]
        loss_div = (
            torch.relu(self.div_margin - div_dists).mean()
            if div_dists.shape[0] > 0
            else torch.tensor(0.0, device=self.device)
        )

        total = self.lambda_gw * loss_gw + self.lambda_recon * loss_recon + self.lambda_div * loss_div
        return total, {
            "loss_gw": float(loss_gw.item()),
            "loss_rec": float(loss_recon.item()),
            "loss_div": float(loss_div.item()),
            "total": float(total.item()),
        }


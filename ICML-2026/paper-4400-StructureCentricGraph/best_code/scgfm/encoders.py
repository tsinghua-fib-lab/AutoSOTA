from __future__ import annotations

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


class SCGFMEncoder:
    """Frozen downstream encoder producing z(G) = [w, f(w), vec(H)]."""

    def __init__(
        self,
        model,
        tau: float | None = None,
        device: str | torch.device = "cpu",
        max_dim: int = 100,
        num_projections: int = 200,
        top_k: int = 8,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.tau = tau if tau is not None else model.tau
        self.max_dim = max_dim
        self.num_projections = num_projections
        self.top_k = top_k
        with torch.no_grad():
            self.bases = self.model.get_normalized_bases().detach()
        self.K, self.M, _ = self.bases.shape

        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.theta = torch.randn(max_dim, num_projections, device=self.device)
        self.theta = self.theta / (torch.norm(self.theta, dim=0, keepdim=True) + 1e-8)
        torch.set_rng_state(rng_state)

    @staticmethod
    def compute_distribution(adj_np: np.ndarray) -> np.ndarray:
        degrees = adj_np.sum(axis=1)
        total_deg = degrees.sum()
        if total_deg < 1e-6:
            return ot.unif(adj_np.shape[0])
        return degrees / total_deg

    def compute_sliced_gw_distance_fast(self, adj: torch.Tensor) -> torch.Tensor:
        n = adj.shape[0]
        # Normalize adjacency to remove scale effects
        adj_norm = self._normalize_adj(adj)
        if n < self.max_dim:
            adj_padded = F.pad(adj_norm.unsqueeze(0), (0, self.max_dim - n, 0, self.max_dim - n))
        else:
            adj_padded = adj_norm.unsqueeze(0)[:, : self.max_dim, : self.max_dim]
        bases_norm = F.normalize(self.bases.view(self.K, -1), p=2, dim=1).view(self.K, self.M, self.M)
        bases_padded = F.pad(bases_norm, (0, self.max_dim - self.M, 0, self.max_dim - self.M))
        adj_proj = torch.matmul(adj_padded, self.theta)
        bases_proj = torch.matmul(bases_padded, self.theta)
        adj_sorted, _ = torch.sort(adj_proj, dim=1)
        bases_sorted, _ = torch.sort(bases_proj, dim=1)
        return torch.mean((adj_sorted - bases_sorted) ** 2, dim=(1, 2))

    def compute_transport_matrix(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        c1_np = c1.detach().cpu().numpy().astype(np.float64)
        c2_np = c2.detach().cpu().numpy().astype(np.float64)
        p = self.compute_distribution(c1_np)
        q = ot.unif(self.M)
        try:
            transport = ot.gromov.gromov_wasserstein(c1_np, c2_np, p, q, "kl_loss", verbose=False, numItermax=50)
        except Exception:
            transport = np.outer(p, q)
        return torch.from_numpy(transport).to(self.device).float()

    @staticmethod
    def _normalize_adj(adj: "torch.Tensor") -> "torch.Tensor":
        """Frobenius-normalize adjacency to remove scale effects."""
        fnorm = torch.norm(adj, p="fro")
        if fnorm > 1e-8:
            return adj / fnorm
        return adj

    def encode_single(self, data) -> torch.Tensor:
        edge_index = data.edge_index.to(self.device)
        adj = to_dense_adj(edge_index, max_num_nodes=data.num_nodes)[0]
        n = data.num_nodes

        if getattr(data, "x", None) is not None:
            x = data.x.to(self.device).float()
        else:
            deg = adj.sum(dim=1).unsqueeze(1)
            x = deg / (deg.max() + 1e-6)

        adj_norm = self._normalize_adj(adj)
        dists = self.compute_sliced_gw_distance_fast(adj)
        w_i = F.softmax(-dists / self.tau, dim=0)
        f_hat = self.model.decoder(w_i.unsqueeze(0)).squeeze(0)

        h_accum = torch.zeros(self.M, x.shape[1], device=self.device)
        k_select = self.K if self.top_k is None or self.top_k <= 0 else min(self.top_k, self.K)
        top_vals, top_indices = torch.topk(w_i, k=k_select)
        top_vals = top_vals / top_vals.sum()
        for idx, weight in zip(top_indices, top_vals):
            basis = self.bases[idx.item()]
            transport = self.compute_transport_matrix(adj_norm, basis)
            h_accum += weight * (torch.matmul(transport.t(), x) * n)

        return torch.cat([w_i * 5.0, f_hat, h_accum.flatten()], dim=0)

    def encode_dataset(self, dataset, show_progress: bool = True):
        xs = []
        ys = []
        iterator = tqdm(dataset, desc="Encode graphs") if show_progress else dataset
        with torch.no_grad():
            for data in iterator:
                xs.append(self.encode_single(data).cpu())
                y = data.y.item() if data.y.numel() == 1 else data.y.view(-1)[0].item()
                ys.append(y)
        return torch.stack(xs), torch.tensor(ys).long()

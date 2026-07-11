# precompute_backbones.py
from __future__ import annotations

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def _spmm(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Sparse matmul (COO): out[col] += edge_weight * x[row]
    edge_index: [2, E], edge_weight: [E], x: [N, F] -> out: [N, F]
    """
    row, col = edge_index[0], edge_index[1]
    out = torch.zeros((num_nodes, x.size(1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, col, x[row] * edge_weight.unsqueeze(1))
    return out


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ar = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    self_loops = torch.stack([ar, ar], dim=0)
    return torch.cat([edge_index, self_loops], dim=1)


@torch.no_grad()
def precompute_sgc_x(x, edge_index, num_nodes, K):
    ei, w = gcn_norm(edge_index, None, num_nodes=num_nodes, add_self_loops=True, dtype=x.dtype)
    for _ in range(K):
        x = _spmm(ei, w, x, num_nodes)
    return x


@torch.no_grad()
def precompute_gin_linear_x(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, K: int) -> torch.Tensor:
    """
    GIN-Linear: x <- (A+I)^K x  (sum aggregation with self loops, no normalization)
    """
    K = int(K)
    if K <= 0:
        return x

    ei = _add_self_loops(edge_index, num_nodes)
    w = torch.ones(ei.size(1), device=ei.device, dtype=x.dtype)
    for _ in range(K):
        x = _spmm(ei, w, x, num_nodes)
    return x

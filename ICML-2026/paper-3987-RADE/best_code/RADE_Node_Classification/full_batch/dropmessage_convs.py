# dropmessage_convs.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import reset


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


class DropMessageGCNConv(nn.Module):
    """
    Baseline GCN with DropMessage:
      - add self-loops
      - symmetric normalization
      - during training: drop normalized messages (edge-wise) via dropout
    """
    supports_dropmessage = True

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        x = self.lin(x)
        N = x.size(0)

        row, col = edge_index[0], edge_index[1]
        # robustly remove self-loops if any, then add self-loops
        mask = row != col
        row, col = row[mask], col[mask]
        ar = torch.arange(N, device=x.device)
        row = torch.cat([row, ar], dim=0)
        col = torch.cat([col, ar], dim=0)

        w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
        deg = _scatter_add_scalar(w, col, N).clamp(min=1.0)
        inv_sqrt = deg.pow(-0.5)

        gamma = inv_sqrt[row] * inv_sqrt[col]  # [E]
        msg = gamma.unsqueeze(1) * x[row]      # [E,F]

        if self.training and drop_rate > 0.0:
            msg = F.dropout(msg, p=float(drop_rate), training=True)

        out = _scatter_add_rows(msg, col, N)
        if self.bias is not None:
            out = out + self.bias
        return out


class DropMessageGINConv(nn.Module):
    """
    Baseline GIN sum aggregation with DropMessage:
      - neigh_sum = sum_j dropout(x_j) (edge-wise)
      - out = MLP( (1+eps)x + neigh_sum )
    """
    supports_dropmessage = True

    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))

    def reset_parameters(self) -> None:
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]
        # remove self-loops from neighbor sum (GIN uses (1+eps)x for self contribution)
        mask = row != col
        row, col = row[mask], col[mask]

        msg = x[row]
        if self.training and drop_rate > 0.0:
            msg = F.dropout(msg, p=float(drop_rate), training=True)

        neigh_sum = _scatter_add_rows(msg, col, N)
        out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
        return self.mlp(out)


class DropMessageGATConv(nn.Module):
    """
    Baseline GAT with DropMessage-style attention/message dropout.

    PyG's GATConv applies dropout to normalized attention coefficients during
    training, which is the GAT analogue of dropping edge messages.
    """
    supports_dropmessage = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=0.0,
            add_self_loops=False,
            bias=bias,
        )

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        previous_dropout = float(self.conv.dropout)
        self.conv.dropout = float(drop_rate) if self.training else 0.0
        try:
            return self.conv(x, edge_index)
        finally:
            self.conv.dropout = previous_dropout

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset

from ..full_batch.rade_convs import RADEGATConv as _RADEGATConvBase


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}")
    return rade_variant


@dataclass
class BatchGraphCache:
    edge_index_clean_dir: torch.Tensor
    num_nodes: int
    deg_clean: torch.Tensor
    comp_size: torch.Tensor
    global_n_id: torch.Tensor
    global_num_nodes_total: int

    @staticmethod
    def from_edge_index(
        edge_index_clean: torch.Tensor,
        num_nodes: int,
        *,
        node_ids: Optional[torch.Tensor] = None,
        global_num_nodes_total: Optional[int] = None,
    ) -> "BatchGraphCache":
        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones)

        comp = (num_nodes - 1.0 - deg).clamp(min=0.0)
        if node_ids is None:
            global_n_id = torch.arange(num_nodes, device=col.device, dtype=torch.long)
        else:
            global_n_id = node_ids.to(device=col.device, dtype=torch.long)

        if global_num_nodes_total is None:
            total_nodes = int(global_n_id.max().item()) + 1 if global_n_id.numel() > 0 else int(num_nodes)
        else:
            total_nodes = int(global_num_nodes_total)

        return BatchGraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            num_nodes=num_nodes,
            deg_clean=deg,
            comp_size=comp,
            global_n_id=global_n_id,
            global_num_nodes_total=total_nodes,
        )

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, n)
        total_sum = x.sum(dim=0, keepdim=True)
        return total_sum - x - neigh_sum

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        total_w = w.sum()
        neigh_wx = _scatter_add_rows(wx[row], col, n)
        neigh_w = _scatter_add_scalar(w[row], col, n)
        num = total_wx - wx - neigh_wx
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        neigh_wx = _scatter_add_rows(wx[row], col, n)
        return total_wx - wx - neigh_wx


class RADEGATConvMB(_RADEGATConvBase):
    """
    Mini-batch/local-subgraph GAT RADE operator.

    The full-batch sampled attention moment implementation applies unchanged
    over the sampled local node set exposed by BatchGraphCache.
    """

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
        self.graph_cache = graph_cache


def _delta_E_inv_caseA(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float, eps: float = 1e-12) -> torch.Tensor:
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * d_minus + q * comp_size + 2.0
    var = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float, eps: float = 1e-12) -> torch.Tensor:
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * deg_clean + q * c_minus + 2.0
    var = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_sqrt_caseA(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    a = (1.0 - p) * d_minus + q * comp_size + 2.0
    b = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return a.pow(-0.5) + (3.0 / 8.0) * b * a.pow(-2.5)


def _delta_E_inv_sqrt_caseB(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    a = (1.0 - p) * deg_clean + q * c_minus + 2.0
    b = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return a.pow(-0.5) + (3.0 / 8.0) * b * a.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConvMB(nn.Module):
    """
    Mini-batch GCN symmetric normalization with analytic RADE correction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        correct_self_loop: bool = True,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.correct_self_loop = bool(correct_self_loop)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_eps = 1e-12
        self.graph_cache: Optional[BatchGraphCache] = None

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once per mini-batch."
        x = self.lin(x)
        n = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            ar = torch.arange(n, device=x.device)
            row = torch.cat([row, ar])
            col = torch.cat([col, ar])
            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, n)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
            gamma = inv_sqrt[row] * inv_sqrt[col]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, n)

            if use_ofs and (not self.training) and q > 0.0:
                deg_clean = self.graph_cache.deg_clean.to(x.device)
                comp_size = self.graph_cache.comp_size.to(x.device)
                t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                comp_wx = self.graph_cache.complement_sum_weighted(x, w=t_b)
                out = out + (q * t_b).unsqueeze(1) * comp_wx

            if self.bias is not None:
                out = out + self.bias
            return out

        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        ar = torch.arange(n, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])
        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, n).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        gamma_self = inv_sqrt_aug * inv_sqrt_aug

        if not self.ep_correction:
            out_k = _scatter_add_rows(gamma_k.unsqueeze(1) * x[row_k], col_k, n)
            out_a = (
                _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, n)
                if row_a.numel()
                else torch.zeros_like(out_k)
            )
            out = out_k + out_a + gamma_self.unsqueeze(1) * x
            if self.bias is not None:
                out = out + self.bias
            return out

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)
        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        t_a = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
        e_gamma_k = (1.0 - p) * t_a[row_k] * t_a[col_k]
        corr_k = alpha_k / e_gamma_k.clamp(min=self.ep_eps)
        out_k = _scatter_add_rows((gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, n)

        if row_a.numel():
            sum_a = _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, n)
            if use_ofs:
                out_a = sum_a
            else:
                t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                mu = self.graph_cache.complement_mean_weighted(x, w=t_b)
                sum_g = _scatter_add_scalar(gamma_a, col_a, n).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        if self.correct_self_loop:
            e_gamma_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_self = alpha_self / e_gamma_self.clamp(min=self.ep_eps)
            out_self = (gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out


class RADEGINConvMB(nn.Module):
    """
    Mini-batch GIN sum aggregation with analytic RADE correction.
    """

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
    ):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = float(eps)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_eps = 1e-12
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], dtype=torch.float32))
        else:
            self.register_buffer("eps", torch.tensor([self.initial_eps], dtype=torch.float32))
        self.graph_cache: Optional[BatchGraphCache] = None

    def set_graph(self, graph_cache: BatchGraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[torch.Tensor] = None,
        edge_index_add: Optional[torch.Tensor] = None,
        p: float = 0.0,
        q: float = 0.0,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once per mini-batch."
        n = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            neigh = _scatter_add_rows(x[row], col, n)
            out = (1.0 + self.eps) * x + neigh
            if use_ofs and (not self.training) and q > 0.0:
                out = out + q * self.graph_cache.complement_sum_unweighted(x)
            return self.mlp(out)

        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        neigh_keep = _scatter_add_rows(x[row_k], col_k, n)
        add_sum = (
            _scatter_add_rows(x[row_a], col_a, n)
            if row_a.numel()
            else torch.zeros_like(x)
        )

        if not self.ep_correction:
            out = (1.0 + self.eps) * x + neigh_keep + add_sum
            return self.mlp(out)

        keep_prob = max(self.ep_eps, 1.0 - float(p))
        neigh_corr = neigh_keep / keep_prob
        if use_ofs:
            add_corr = add_sum
        elif row_a.numel():
            mu = self.graph_cache.complement_sum_unweighted(x) / self.graph_cache.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
            add_count = _scatter_add_scalar(torch.ones(row_a.numel(), device=x.device, dtype=x.dtype), col_a, n).unsqueeze(1)
            add_corr = add_sum - add_count * mu
        else:
            add_corr = torch.zeros_like(x)

        out = (1.0 + self.eps) * x + neigh_corr + add_corr
        return self.mlp(out)

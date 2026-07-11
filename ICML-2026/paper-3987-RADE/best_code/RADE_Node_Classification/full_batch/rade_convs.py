from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Row-wise scatter-add.

    Args:
        values: [E, F]
        index:  [E]
        dim_size: number of destination rows

    Returns:
        out: [dim_size, F], where out[i] = sum_{e: index[e]=i} values[e]
    """
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scalar scatter-add.

    Args:
        values: [E]
        index:  [E]
        dim_size: number of destination entries

    Returns:
        out: [dim_size], where out[i] = sum_{e: index[e]=i} values[e]
    """
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_heads(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter-add for multi-head messages.

    Args:
        values: [E, H, F]
        index:  [E]

    Returns:
        out: [N, H, F]
    """
    if values.numel() == 0:
        return torch.zeros(
            (dim_size, values.size(1), values.size(2)),
            device=values.device,
            dtype=values.dtype,
        )
    flat = values.reshape(values.size(0), values.size(1) * values.size(2))
    out = _scatter_add_rows(flat, index, dim_size)
    return out.view(dim_size, values.size(1), values.size(2))


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(
            f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
        )
    return rade_variant


def _directed_edge_keys(row: torch.Tensor, col: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return row.to(torch.long) * int(num_nodes) + col.to(torch.long)


@dataclass
class GraphCache:
    """
    Cache derived from the clean graph G (without self-loops in edge_index_clean).

    Stored quantities:
      - edge_index_clean_dir: directed clean edges (both directions), no self-loops
      - deg_clean[i]: clean degree d_i in G (excluding self-loop)
      - comp_size[i]: |C(i)| = n - 1 - d_i, where C(i) is the clean non-neighbor set
    """
    edge_index_clean_dir: torch.Tensor  # [2, E_dir]
    num_nodes: int
    deg_clean: torch.Tensor             # [N]
    comp_size: torch.Tensor             # [N]

    @staticmethod
    def from_edge_index(edge_index_clean: torch.Tensor, num_nodes: int) -> "GraphCache":
        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones)  # for undirected graphs stored both ways: indegree == degree

        comp = (num_nodes - 1.0 - deg).clamp(min=0.0)
        return GraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            num_nodes=num_nodes,
            deg_clean=deg,
            comp_size=comp,
        )

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unweighted clean non-neighbor mean:
            mu_i = (1 / |C(i)|) sum_{j in C(i)} x_j

        Computed exactly in O(E + N) via global sums.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)   # sum_{j in N(i)} x_j
        total_sum = x.sum(dim=0, keepdim=True)          # sum_j x_j
        comp_sum = total_sum - x - neigh_sum            # sum_{j in C(i)} x_j

        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted clean non-neighbor mean:
            mu_i = (sum_{j in C(i)} w_j x_j) / (sum_{j in C(i)} w_j)

        This is used for the GCN Case-B weighted centering approximation, where the
        paper reduces the non-edge weights to node-wise terms w_j under the decoupled
        approximation.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        total_w = w.sum()

        neigh_wx = _scatter_add_rows(wx[row], col, N)   # sum_{j in N(i)} w_j x_j
        neigh_w = _scatter_add_scalar(w[row], col, N)   # sum_{j in N(i)} w_j

        num = total_wx - wx - neigh_wx                  # sum_{j in C(i)} w_j x_j
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unweighted clean non-neighbor sum:
            S_i = sum_{j in C(i)} x_j
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, N)
        total_sum = x.sum(dim=0, keepdim=True)
        comp_sum = total_sum - x - neigh_sum
        return comp_sum

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Weighted clean non-neighbor sum:
            S_i = sum_{j in C(i)} w_j x_j

        Used in the GCN RADE-OFS inference correction under the
        decoupled Case-B approximation.
        """
        N = self.num_nodes
        row, col = self.edge_index_clean_dir.to(x.device)

        wx = w.unsqueeze(1) * x
        total_wx = wx.sum(dim=0, keepdim=True)
        neigh_wx = _scatter_add_rows(wx[row], col, N)
        comp_wx = total_wx - wx - neigh_wx
        return comp_wx


def _delta_E_inv_caseA(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Case A (clean edge): delta-method approximation for E[X^{-1}],
    where X is the self-loop-included sampled degree term appearing in
    GCN normalization for a clean edge contribution.

    Mean and variance:
        mu  = (1-p)(d_i - 1) + q|C(i)| + 2
        var = (d_i - 1)(1-p)p + |C(i)| q(1-q)

    Approximation:
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3}
    """
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * d_minus + q * comp_size + 2.0
    var = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_caseB(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Case B (clean non-edge): delta-method approximation for E[X^{-1}],
    where X is the self-loop-included sampled degree term appearing in
    GCN normalization for a non-edge contribution.

    Mean and variance:
        mu  = (1-p)d_i + q(|C(i)| - 1) + 2
        var = d_i(1-p)p + (|C(i)| - 1) q(1-q)

    Approximation:
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3}
    """
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    mu = (1.0 - p) * deg_clean + q * c_minus + 2.0
    var = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    mu = mu.clamp(min=eps)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


def _delta_E_inv_sqrt_caseA(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
) -> torch.Tensor:
    """
    Case A (clean edge): delta-method approximation for E[X^{-1/2}],
    where X is the self-loop-included sampled degree term used in the
    GCN normalization factor for a clean edge contribution.

    Let
        A = (1-p)(d_i - 1) + q|C(i)| + 2
        B = (d_i - 1)(1-p)p + |C(i)| q(1-q)

    Then
        E[X^{-1/2}] ≈ A^{-1/2} + (3/8) B A^{-5/2}.
    """
    d_minus = (deg_clean - 1.0).clamp(min=0.0)
    A = (1.0 - p) * d_minus + q * comp_size + 2.0
    B = d_minus * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_sqrt_caseB(
    deg_clean: torch.Tensor,
    comp_size: torch.Tensor,
    p: float,
    q: float,
) -> torch.Tensor:
    """
    Case B (clean non-edge): delta-method approximation for E[X^{-1/2}],
    where X is the self-loop-included sampled degree term used in the
    GCN normalization factor for a non-edge contribution.

    Let
        A = (1-p)d_i + q(|C(i)| - 1) + 2
        B = d_i(1-p)p + (|C(i)| - 1) q(1-q)

    Then
        E[X^{-1/2}] ≈ A^{-1/2} + (3/8) B A^{-5/2}.
    """
    c_minus = (comp_size - 1.0).clamp(min=0.0)
    A = (1.0 - p) * deg_clean + q * c_minus + 2.0
    B = deg_clean * (1.0 - p) * p + c_minus * q * (1.0 - q)
    return A.pow(-0.5) + (3.0 / 8.0) * B * A.pow(-2.5)


def _delta_E_inv_self(deg_clean: torch.Tensor, comp_size: torch.Tensor, p: float, q: float) -> torch.Tensor:
    """
    Delta-method approximation for the self-loop normalization term.

    For
        X = 1 + Bin(d_i, 1-p) + Bin(|C(i)|, q),

    we use
        E[X^{-1}] ≈ mu^{-1} + var * mu^{-3},
    where
        mu  = 1 + (1-p)d_i + q|C(i)|,
        var = d_i(1-p)p + |C(i)|q(1-q).
    """
    mu = 1.0 + (1.0 - p) * deg_clean + q * comp_size
    var = deg_clean * (1.0 - p) * p + comp_size * q * (1.0 - q)
    return mu.pow(-1.0) + var * mu.pow(-3.0)


class RADEGCNConv(nn.Module):
    """
    GCN symmetric normalization with analytic RADE expectation preservation.
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
        self.graph_cache: Optional[GraphCache] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."

        x = self.lin(x)
        N = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            ar = torch.arange(N, device=x.device)
            row = torch.cat([row, ar])
            col = torch.cat([col, ar])

            w = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            deg = _scatter_add_scalar(w, col, N)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

            gamma = inv_sqrt[row] * inv_sqrt[col]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row], col, N)

            if use_ofs and (not self.training) and q > 0.0:
                deg_clean = self.graph_cache.deg_clean.to(x.device)
                comp_size = self.graph_cache.comp_size.to(x.device)
                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                comp_wx = self.graph_cache.complement_sum_weighted(x, w=tB)
                out = out + (q * tB).unsqueeze(1) * comp_wx

            if self.bias is not None:
                out = out + self.bias
            return out

        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        ar = torch.arange(N, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])

        w_all = torch.ones(row_all.numel(), device=x.device, dtype=x.dtype)
        deg_aug = _scatter_add_scalar(w_all, col_all, N).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        Gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        Gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        Gamma_self = inv_sqrt_aug * inv_sqrt_aug

        if not self.ep_correction:
            out_k = _scatter_add_rows(Gamma_k.unsqueeze(1) * x[row_k], col_k, N)
            out_a = (
                _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)
                if row_a.numel()
                else torch.zeros_like(out_k)
            )
            out_self = Gamma_self.unsqueeze(1) * x
            out = out_k + out_a + out_self
            if self.bias is not None:
                out = out + self.bias
            return out

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)

        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        tA = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
        E_Gamma_k = (1.0 - p) * tA[row_k] * tA[col_k]
        corr_k = alpha_k / E_Gamma_k.clamp(min=self.ep_eps)
        out_k = _scatter_add_rows((Gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, N)

        if row_a.numel():
            sum_a = _scatter_add_rows(Gamma_a.unsqueeze(1) * x[row_a], col_a, N)
            if use_ofs:
                out_a = sum_a
            else:
                tB = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                mu = self.graph_cache.complement_mean_weighted(x, w=tB)
                sum_g = _scatter_add_scalar(Gamma_a, col_a, N).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        if self.correct_self_loop:
            E_Gamma_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_self = alpha_self / E_Gamma_self.clamp(min=self.ep_eps)
            out_self = (Gamma_self * corr_self).unsqueeze(1) * x
        else:
            out_self = Gamma_self.unsqueeze(1) * x

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out

class RADEGATConv(nn.Module):
    """
    Full-batch GAT attention aggregation with RADE expectation preservation.

    This implementation follows the attention-normalization delta approximation:
        E[alpha_ij]  ~= pi_ij w_ij (D_ij^-1 + sigma_ij^2 D_ij^-3)
        E[alpha_ij^2] ~= pi_ij w_ij^2 (D_ij^-2 + 3 sigma_ij^2 D_ij^-4)
    where D_ij = w_ij + mu_ij and mu/sigma exclude j from i's random
    attention denominator. Self-loops are excluded, matching the provided
    k != i formulation.

    Sampled moments keep clean-neighbor contributions exact and estimate dense
    clean-complement terms by deterministic Monte Carlo samples. The scalable
    sampler draws non-neighbors with replacement.

    The forward operator supports multiple heads. The GradNorm variance proxy is
    intentionally restricted to heads=1 until a multi-head logit variance
    approximation is derived and validated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
        rade_variant: str = "rade-of",
        ep_correction: bool = True,
        ep_eps: float = 1e-12,
        moment_chunk_size: int = 1024,
        moment_samples: int = 256,
        nonedge_samples: int = 256,
        moment_seed: int = 0,
        ep_corr_clip: float = 0.0,
    ):
        super().__init__()
        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError(f"heads must be positive. Got {heads}")
        self.concat = bool(concat)
        if self.concat:
            if int(out_channels) % self.heads != 0:
                raise ValueError(
                    "For concat=True, out_channels must be divisible by heads. "
                    f"Got out_channels={out_channels}, heads={heads}."
                )
            self.out_per_head = int(out_channels) // self.heads
            self.out_channels = int(out_channels)
        else:
            self.out_per_head = int(out_channels)
            self.out_channels = int(out_channels)

        self.lin = nn.Linear(in_channels, self.heads * self.out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.heads, self.out_per_head))
        self.att_dst = nn.Parameter(torch.empty(self.heads, self.out_per_head))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.negative_slope = float(negative_slope)
        self.rade_variant = _validate_rade_variant(rade_variant)
        self.ep_correction = bool(ep_correction)
        self.ep_eps = float(ep_eps)
        self.moment_chunk_size = max(1, int(moment_chunk_size))
        self.moment_samples = max(1, int(moment_samples))
        self.nonedge_samples = max(1, int(nonedge_samples))
        self.moment_seed = int(moment_seed)
        self.ep_corr_clip = max(0.0, float(ep_corr_clip))

        self.graph_cache: Optional[GraphCache] = None
        self._last_h_for_moments: Optional[torch.Tensor] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
        self.graph_cache = graph_cache

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).view(x.size(0), self.heads, self.out_per_head)

    def _finalize_heads(self, out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            out = out.reshape(out.size(0), self.heads * self.out_per_head)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _node_attention_scores(
        self,
        h: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard GAT decomposes the pair logit into source and destination
        # terms: a_src^T h_j + a_dst^T h_i. Precomputing both lets the moment
        # code build all candidate weights w_ij = exp(leaky_relu(...)) cheaply.
        att_src = self.att_src.detach() if detach_params else self.att_src
        att_dst = self.att_dst.detach() if detach_params else self.att_dst
        src = (h * att_src.unsqueeze(0)).sum(dim=-1)
        dst = (h * att_dst.unsqueeze(0)).sum(dim=-1)
        return src, dst

    def _edge_logits(
        self,
        h: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> torch.Tensor:
        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        return torch.nn.functional.leaky_relu(
            score_src[row] + score_dst[col],
            negative_slope=self.negative_slope,
        )

    def _edge_softmax(self, logits: torch.Tensor, col: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if logits.numel() == 0:
            return logits
        return softmax(logits, col, num_nodes=num_nodes)

    def _attention_logit_shift(
        self,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
    ) -> torch.Tensor:
        N = score_src.size(0)
        if N <= 1:
            return torch.zeros((N, self.heads), dtype=score_src.dtype, device=score_src.device)

        shift = torch.full(
            (N, self.heads),
            -torch.inf,
            dtype=score_src.dtype,
            device=score_src.device,
        )
        for start, end in self._chunk_ranges(N):
            src_idx = torch.arange(start, end, device=score_src.device, dtype=torch.long)
            logits = torch.nn.functional.leaky_relu(
                score_src[src_idx].unsqueeze(0) + score_dst.unsqueeze(1),
                negative_slope=self.negative_slope,
            )
            logits[src_idx, src_idx - start] = -torch.inf
            shift = torch.maximum(shift, logits.max(dim=1).values)

        return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift)).detach()

    def _scaled_attention_weights(
        self,
        logits: torch.Tensor,
        logit_shift: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        return (logits - logit_shift[col]).exp()

    def _scaled_chunk_attention_weights(
        self,
        logits: torch.Tensor,
        logit_shift: torch.Tensor,
    ) -> torch.Tensor:
        return (logits - logit_shift.unsqueeze(1)).exp()

    def _clean_edge_alpha(
        self,
        h: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> torch.Tensor:
        logits = self._edge_logits(h, row, col, detach_params=detach_params)
        return self._edge_softmax(logits, col, h.size(0))

    def _aggregate(self, h: torch.Tensor, row: torch.Tensor, col: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        msg = alpha.unsqueeze(-1) * h[row]
        return _scatter_add_heads(msg, col, h.size(0))

    def _chunk_ranges(self, num_nodes: int):
        step = max(1, int(self.moment_chunk_size))
        for start in range(0, num_nodes, step):
            end = min(num_nodes, start + step)
            yield start, end

    def _clean_edge_keys_sorted(self, device: torch.device) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        keys = edge_row * int(self.graph_cache.num_nodes) + edge_col
        return torch.sort(keys).values

    @staticmethod
    def _edge_membership(keys: torch.Tensor, edge_keys_sorted: torch.Tensor) -> torch.Tensor:
        if keys.numel() == 0 or edge_keys_sorted.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        pos = torch.searchsorted(edge_keys_sorted, keys)
        valid = pos < edge_keys_sorted.numel()
        pos_clamped = pos.clamp(max=max(int(edge_keys_sorted.numel()) - 1, 0))
        return valid & (edge_keys_sorted[pos_clamped] == keys)

    def _sample_nonedge_sources_with_replacement(
        self,
        num_nodes: int,
        samples: int,
        device: torch.device,
        *,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw per-target clean non-neighbor sources for sampled GAT moments.

        Returns:
            src_out: [N, S] sampled source nodes j for each target i.
            valid_out: [N, S] mask, because targets with |C(i)| < S have fewer
                usable clean non-neighbors than the shared sampled width.
            scale: [N] Horvitz-Thompson-style multiplier |C(i)| / sample_count_i
                used to lift sampled complement sums back to full-complement sums.

        The sampler rejects self-pairs and clean graph edges, so every accepted
        pair (j, i) is a clean non-edge candidate. A fixed CPU generator keeps
        the sampled estimator deterministic for a given moment_seed.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = int(num_nodes)
        comp: torch.Tensor = self.graph_cache.comp_size.to(device=device, dtype=torch.long)
        max_comp: int = int(comp.max().item()) if comp.numel() > 0 else 0
        sample_cols: int = min(max(1, int(samples)), max_comp)
        if N <= 1 or max_comp <= 0:
            src_empty = torch.empty((N, 0), dtype=torch.long, device=device)
            valid_empty = torch.empty((N, 0), dtype=torch.bool, device=device)
            scale_empty = torch.zeros((N,), dtype=torch.float32, device=device)
            return src_empty, valid_empty, scale_empty

        target_counts: torch.Tensor = torch.clamp(comp, max=sample_cols)
        src_out: torch.Tensor = torch.zeros((N, sample_cols), dtype=torch.long, device=device)
        valid_out: torch.Tensor = torch.zeros((N, sample_cols), dtype=torch.bool, device=device)
        counts: torch.Tensor = torch.zeros((N,), dtype=torch.long, device=device)
        edge_keys_sorted: torch.Tensor = self._clean_edge_keys_sorted(device)
        row_ids: torch.Tensor = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.moment_seed) + int(seed_offset))

        # Draw candidates in increasingly large batches. This keeps dense
        # all-pairs materialization out of the sampled path while still filling
        # high-degree targets that reject many candidate neighbors.
        draw: int = max(sample_cols * 2, 64)
        max_attempts: int = 32
        for attempt in range(max_attempts):
            active = counts < target_counts
            if not bool(active.any()):
                break
            candidate_count = N - 1
            raw = torch.randint(
                low=0,
                high=candidate_count,
                size=(N, draw),
                generator=gen,
                device=torch.device("cpu"),
                dtype=torch.long,
            ).to(device)
            target = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1)
            cand = raw + (raw >= target).to(torch.long)
            keys = cand * N + target.expand_as(cand)
            accepted = (~self._edge_membership(keys.reshape(-1), edge_keys_sorted).view_as(cand)) & active.unsqueeze(1)
            rank = accepted.to(torch.long).cumsum(dim=1) - 1 + counts.unsqueeze(1)
            keep = accepted & (rank < target_counts.unsqueeze(1)) & (rank < sample_cols)
            if bool(keep.any()):
                # rank gives the destination column inside src_out for each
                # accepted source; counts tracks how many samples each target
                # still needs after this candidate batch.
                dst_pos = row_ids.expand_as(cand)[keep]
                rank_pos = rank[keep]
                src_out[dst_pos, rank_pos] = cand[keep]
                valid_out[dst_pos, rank_pos] = True
                counts = counts + keep.sum(dim=1).to(torch.long)
            draw = min(max(draw * 2, 64), max(sample_cols * 16, 1024))

        scale: torch.Tensor = torch.zeros((N,), dtype=torch.float32, device=device)
        nonzero = counts > 0
        scale[nonzero] = comp.to(torch.float32)[nonzero] / counts.to(torch.float32)[nonzero]
        return src_out, valid_out, scale

    def _attention_totals_sampled(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate per-target random denominator moments for sampled GAT.

        total_mu[i] and total_sigma[i] approximate the mean and variance of
        the perturbed attention denominator for target i:

            sum_k A'_ik w_ik

        Clean-neighbor terms are summed exactly because they are sparse. Dense
        clean-complement terms are estimated from sampled non-neighbors and
        scaled by |C(i)| / sample_count_i. Later, pair-specific code subtracts
        the numerator candidate j from these totals to get S_ij.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = h.size(0)
        if N <= 1:
            score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
            logit_shift = self._attention_logit_shift(score_src, score_dst)
            zeros = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return zeros, zeros, score_src, score_dst, logit_shift
        dtype: torch.dtype = h.dtype
        device: torch.device = h.device
        p_t: torch.Tensor = torch.as_tensor(p, dtype=dtype, device=device)
        q_t: torch.Tensor = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t: torch.Tensor = 1.0 - p_t

        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        logit_shift = self._attention_logit_shift(score_src, score_dst)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        # In sampled-index mode, clean-edge denominator contributions are still
        # exact because they are sparse and cheap to aggregate.
        edge_logits = torch.nn.functional.leaky_relu(
            score_src[edge_row] + score_dst[edge_col],
            negative_slope=self.negative_slope,
        )
        edge_w = self._scaled_attention_weights(edge_logits, logit_shift, edge_col)
        total_mu = _scatter_add_rows(keep_t.expand_as(edge_w) * edge_w, edge_col, N)
        total_sigma = _scatter_add_rows((keep_t * p_t).expand_as(edge_w) * edge_w.square(), edge_col, N)

        # Sample only clean non-neighbors. valid masks padded sample slots, and
        # scale lifts each target's sample sum back to its full complement size.
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.moment_samples,
            device,
            seed_offset=17,
        )
        if src_idx.numel() == 0:
            return total_mu, total_sigma, score_src, score_dst, logit_shift
        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx] + score_dst.unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits, logit_shift)
        pi = (torch.zeros_like(src_idx, dtype=dtype) + q_t) * valid.to(dtype)
        pi_h = pi.unsqueeze(-1)
        scale_h = scale.to(dtype).view(N, 1)
        # Non-neighbor terms are estimated from sampled clean complement
        # sources and scaled back to the full complement size.
        total_mu = total_mu + scale_h * (pi_h * wij).sum(dim=1)
        total_sigma = total_sigma + scale_h * (pi_h * (1.0 - pi_h) * wij.square()).sum(dim=1)
        return total_mu, total_sigma, score_src, score_dst, logit_shift

    def _attention_totals(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._attention_totals_sampled(h, p=p, q=q, detach_params=detach_params)

    def _pair_moments_from_totals(
        self,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        pi: torch.Tensor | float,
        p: torch.Tensor | float | None = None,
        q: torch.Tensor | float | None = None,
        total_mu: torch.Tensor,
        total_sigma: torch.Tensor,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        logit_shift: torch.Tensor | None = None,
        seed_offset: int = 53,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if row.numel() == 0:
            empty = torch.empty((0, self.heads), dtype=score_src.dtype, device=score_src.device)
            return empty, empty, empty
        pi_t = torch.as_tensor(pi, dtype=score_src.dtype, device=score_src.device)
        if logit_shift is None:
            logit_shift = self._attention_logit_shift(score_src, score_dst)
        logits = torch.nn.functional.leaky_relu(
            score_src[row] + score_dst[col],
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_attention_weights(logits, logit_shift, col)
        pi_h = pi_t.expand_as(wij) if pi_t.ndim == 0 else pi_t.unsqueeze(-1).expand_as(wij)
        # total_mu/total_sigma include j. Subtract the candidate contribution
        # so mu/sigma describe S_ij = sum_{k != i,j}.
        mu = total_mu[col] - pi_h * wij
        mu = mu.clamp_min(0.0)
        sigma = total_sigma[col] - pi_h * (1.0 - pi_h) * wij.square().clamp_min(0.0)
        sigma = sigma.clamp_min(0.0)
        denom = (wij + mu).clamp(min=self.ep_eps)
        e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))
        e2_alpha = pi_h * wij.square() * (denom.pow(-2.0) + 3.0 * sigma * denom.pow(-4.0))
        var_alpha = (e2_alpha - e_alpha.square()).clamp_min(0.0)
        return e_alpha, e2_alpha, var_alpha

    def _nonedge_expectation_sum_sampled(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the expected added non-edge message for each target node.

        centered=False returns the full expected non-edge message sum used by
        RADE-OFS inference. centered=True returns the expected non-edge mean
        mu_i used by RADE-OF to center sampled added messages during training.
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = h.size(0)
        if N <= 1:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=h.dtype, device=h.device)
            den = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return (num if not centered else num), den
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        dtype: torch.dtype = h.dtype
        device: torch.device = h.device
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.nonedge_samples,
            device,
            seed_offset=31,
        )
        if src_idx.numel() == 0:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=dtype, device=device)
            den = torch.zeros((N, self.heads), dtype=dtype, device=device)
            return (num if centered else num), den

        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx] + score_dst.unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits, logit_shift)
        pi_h = wij.new_full(wij.shape, float(q)) * valid.unsqueeze(-1).to(dtype)

        mu = total_mu.unsqueeze(1) - pi_h * wij
        mu = mu.clamp_min(0.0)
        sigma = total_sigma.unsqueeze(1) - pi_h * (1.0 - pi_h) * wij.square()
        sigma = sigma.clamp_min(0.0)
        denom = (wij + mu).clamp(min=self.ep_eps)
        e_alpha = pi_h * wij * (denom.reciprocal() + sigma * denom.pow(-3.0))

        # num/den are Monte Carlo estimates of:
        #   sum_{j in C(i)} E[alpha'_ij] h_j
        #   sum_{j in C(i)} E[alpha'_ij]
        # after multiplying by each target's complement-size scale.
        num = torch.einsum("ish,ishd->ihd", e_alpha, h[src_idx])
        den = e_alpha.sum(dim=1)
        scale_h = scale.to(dtype).view(N, 1)
        num = scale_h.unsqueeze(-1) * num
        den = scale_h * den
        mean = num / den.clamp(min=self.ep_eps).unsqueeze(-1)
        return (mean if centered else num), den

    def _nonedge_expectation_sum(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._nonedge_expectation_sum_sampled(h, p=p, q=q, centered=centered)

    def _nonedge_variance_sum_sampled(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        total_mu: torch.Tensor,
        total_sigma: torch.Tensor,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
        logit_shift: torch.Tensor,
        q_t: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Sampled add-side variance proxy for the GAT GradNorm objective.

        This mirrors _nonedge_expectation_sum_sampled, but accumulates
        Var(alpha'_ij) times either centered message squares (RADE-OF) or raw
        message squares (RADE-OFS).
        """
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N: int = m.size(0)
        if N <= 1:
            return torch.zeros_like(m_sq)

        device: torch.device = m.device
        dtype: torch.dtype = m.dtype
        src_idx, valid, scale = self._sample_nonedge_sources_with_replacement(
            N,
            self.nonedge_samples,
            device,
            seed_offset=43,
        )
        if src_idx.numel() == 0:
            return torch.zeros_like(m_sq)

        logits = torch.nn.functional.leaky_relu(
            score_src[src_idx].squeeze(-1) + score_dst.squeeze(-1).unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        wij = self._scaled_chunk_attention_weights(logits.unsqueeze(-1), logit_shift).squeeze(-1)
        pi = q_t.expand_as(wij) * valid.to(dtype)

        mu_d = total_mu.squeeze(-1).unsqueeze(1) - pi * wij
        mu_d = mu_d.clamp_min(0.0)
        sigma_d = total_sigma.squeeze(-1).unsqueeze(1) - pi * (1.0 - pi) * wij.square()
        sigma_d = sigma_d.clamp_min(0.0)
        denom_d = (wij + mu_d).clamp(min=self.ep_eps)
        e_add = pi * wij * (denom_d.reciprocal() + sigma_d * denom_d.pow(-3.0))
        e2_add = pi * wij.square() * (denom_d.pow(-2.0) + 3.0 * sigma_d * denom_d.pow(-4.0))
        var_add = (e2_add - e_add.square()).clamp_min(0.0)

        if self.rade_variant == "rade-of":
            # For RADE-OF, the sampled add message is centered by the estimated
            # added-neighborhood mean for the same target node.
            scale_t = scale.to(dtype).view(N, 1)
            den = scale_t.squeeze(1) * e_add.sum(dim=1)
            num = scale_t * (e_add.unsqueeze(-1) * m[src_idx]).sum(dim=1)
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)
            centered_sq = (m[src_idx] - mu_msg.unsqueeze(1)).square()
            return scale_t * (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)

        # RADE-OFS keeps added non-edge messages uncentered, so the variance
        # contribution uses m_j^2 directly.
        return scale.to(dtype).view(N, 1) * (var_add.unsqueeze(-1) * m_sq[src_idx]).sum(dim=1)

    def gat_variance_proxy(
        self,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            return torch.zeros_like(m_sq)
        if self.heads != 1:
            raise ValueError("GAT GradNorm variance currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."

        h = self._last_h_for_moments.to(device=m.device, dtype=m.dtype)
        # GradNorm uses the last forward-pass hidden states but detaches GAT
        # attention parameters, so this remains a variance proxy for the
        # classifier-side messages.
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row, col = self.graph_cache.edge_index_clean_dir[0].to(m.device), self.graph_cache.edge_index_clean_dir[1].to(m.device)

        # Clean-edge part: compare random expected attention to clean attention
        # and scale message variance by Var(alpha_ij) / E[alpha_ij]^2.
        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True).squeeze(-1)
        e_alpha, _, var_alpha = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=(1.0 - torch.as_tensor(p, dtype=m.dtype, device=m.device)),
            p=p,
            q=q,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
            seed_offset=101,
        )
        e_alpha = e_alpha.squeeze(-1)
        var_alpha = var_alpha.squeeze(-1)
        factor = clean_alpha.square() * (var_alpha / (e_alpha.square() + float(eps)))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, m.size(0))

        add_var = torch.zeros_like(neigh_var)
        q_t = torch.as_tensor(q, dtype=m.dtype, device=m.device)
        if float(q_t.detach()) <= 0.0 and not q_t.requires_grad:
            return neigh_var

        add_var = self._nonedge_variance_sum_sampled(
            m,
            m_sq,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
            q_t=q_t,
            eps=float(eps),
        )
        return neigh_var + add_var

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
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        h = self._project(x)
        self._last_h_for_moments = h.detach()
        N = h.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
            # Clean graph path: use ordinary GAT attention over clean edges.
            # At inference RADE-OFS may add the expected non-edge contribution.
            row, col = edge_index_clean[0], edge_index_clean[1]
            mask = row != col
            row, col = row[mask], col[mask]
            alpha = self._clean_edge_alpha(h, row, col)
            out = self._aggregate(h, row, col, alpha)

            if use_ofs and (not self.training) and q > 0.0:
                nonedge_sum, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=False)
                out = out + nonedge_sum

            return self._finalize_heads(out)

        if edge_index_keep is None:
            row_k, col_k = edge_index_clean[0], edge_index_clean[1]
            mask_k = row_k != col_k
            row_k, col_k = row_k[mask_k], col_k[mask_k]
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        # Build the realized perturbed graph for this training step:
        # retained clean edges plus sampled clean non-edges. GAT normalizes
        # attention over exactly these realized incoming edges per target.
        row_all = torch.cat([row_k, row_a], dim=0)
        col_all = torch.cat([col_k, col_a], dim=0)
        alpha_all = self._edge_softmax(self._edge_logits(h, row_all, col_all), col_all, N)
        alpha_k = alpha_all[: row_k.numel()]
        alpha_a = alpha_all[row_k.numel():]

        if not self.ep_correction:
            # No expectation-preservation correction: aggregate the realized
            # perturbed graph directly.
            out = self._aggregate(h, row_all, col_all, alpha_all)
            return self._finalize_heads(out)

        # EP correction rescales retained clean-edge messages so their expected
        # attention matches clean attention under the sampled p,q moment model.
        # _attention_totals keeps sparse clean-edge terms exact and estimates
        # only dense non-edge denominator terms.
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        clean_alpha_k = self._clean_edge_alpha(h, row_k, col_k)
        e_keep, _, _ = self._pair_moments_from_totals(
            row=row_k,
            col=col_k,
            pi=1.0 - float(p),
            p=p,
            q=q,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
            seed_offset=107,
        )
        corr_k = clean_alpha_k / e_keep.clamp(min=self.ep_eps)
        if self.ep_corr_clip > 0.0:
            corr_k = corr_k.clamp(max=float(self.ep_corr_clip))
        out_k = self._aggregate(h, row_k, col_k, alpha_k * corr_k)

        if row_a.numel() > 0:
            if use_ofs:
                # RADE-OFS intentionally keeps added non-edge messages.
                out_a = self._aggregate(h, row_a, col_a, alpha_a)
            else:
                # RADE-OF centers added non-edge messages by subtracting the
                # expected added-neighborhood message for each target. In the
                # sampled GAT mode, that expectation is itself estimated from
                # sampled clean non-neighbors and scaled to the full complement.
                mu_h, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=True)
                msg_a = alpha_a.unsqueeze(-1) * (h[row_a] - mu_h[col_a])
                out_a = _scatter_add_heads(msg_a, col_a, N)
        else:
            out_a = torch.zeros_like(out_k)

        return self._finalize_heads(out_k + out_a)


class RADEGINConv(nn.Module):
    """
    GIN sum aggregation with analytic RADE expectation preservation.
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

        self.graph_cache: Optional[GraphCache] = None

    def set_graph(self, graph_cache: GraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) once after loading data."
        N = x.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, N)

            if use_ofs and (not self.training) and q > 0.0:
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + max(0.0, float(q)) * comp_sum

            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        if not self.ep_correction:
            keep_sum = _scatter_add_rows(x[row_k], col_k, N)
            add_sum = (
                _scatter_add_rows(x[row_a], col_a, N)
                if row_a.numel() > 0
                else torch.zeros_like(keep_sum)
            )
            out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
            return self.mlp(out)

        keep_prob = max(self.ep_eps, 1.0 - float(p))
        alpha = 1.0 / keep_prob
        keep_sum = _scatter_add_rows(x[row_k], col_k, N) * alpha

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, N)
            if use_ofs:
                add_sum = add_sum_raw
            else:
                mu = self.graph_cache.complement_mean_unweighted(x)
                add_deg = _scatter_add_scalar(
                    torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                    col_a,
                    N,
                ).unsqueeze(1)
                add_sum = add_sum_raw - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
        return self.mlp(out)

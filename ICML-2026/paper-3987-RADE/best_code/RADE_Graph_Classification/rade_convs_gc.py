from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset

try:
    from RADE_Node_Classification.full_batch.rade_convs import RADEGATConv as _RADEGATConvBase
except Exception:
    import importlib.util
    import sys
    from pathlib import Path

    _RADE_GAT_PATH = Path(__file__).resolve().parents[1] / "RADE_Node_Classification" / "full_batch" / "rade_convs.py"
    _RADE_GAT_SPEC = importlib.util.spec_from_file_location("rade_full_batch_gat_base", _RADE_GAT_PATH)
    if _RADE_GAT_SPEC is None or _RADE_GAT_SPEC.loader is None:
        raise ImportError(f"Could not load full-batch RADEGATConv from {_RADE_GAT_PATH}")
    _RADE_GAT_MODULE = importlib.util.module_from_spec(_RADE_GAT_SPEC)
    sys.modules[_RADE_GAT_SPEC.name] = _RADE_GAT_MODULE
    _RADE_GAT_SPEC.loader.exec_module(_RADE_GAT_MODULE)
    _RADEGATConvBase = _RADE_GAT_MODULE.RADEGATConv


def _scatter_add_rows(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_scalar(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_add_heads(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros(
            (dim_size, values.size(1), values.size(2)),
            device=values.device,
            dtype=values.dtype,
        )
    flat = values.reshape(values.size(0), values.size(1) * values.size(2))
    out = _scatter_add_rows(flat, index, dim_size)
    return out.view(dim_size, values.size(1), values.size(2))


def _scatter_add_graph_rows(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros((num_graphs, values.size(1)), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
    return out


def _scatter_add_graph_scalar(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros((num_graphs,), device=values.device, dtype=values.dtype)
    out.index_add_(0, batch, values)
    return out


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}")
    return rade_variant


@dataclass
class BatchedGraphCache:
    edge_index_clean_dir: torch.Tensor
    batch: torch.Tensor
    num_nodes: int
    num_graphs: int
    deg_clean: torch.Tensor
    comp_size: torch.Tensor
    n_per_node: torch.Tensor
    global_n_id: torch.Tensor
    global_num_nodes_total: int

    @staticmethod
    def from_edge_index(
        edge_index_clean: torch.Tensor,
        batch: torch.Tensor,
        *,
        num_graphs: Optional[int] = None,
        node_ids: Optional[torch.Tensor] = None,
        global_num_nodes_total: Optional[int] = None,
    ) -> "BatchedGraphCache":
        batch = batch.to(torch.long)
        num_nodes = int(batch.numel())
        num_graphs = int(batch.max().item() + 1) if num_graphs is None and batch.numel() > 0 else int(num_graphs or 0)

        row, col = edge_index_clean[0], edge_index_clean[1]
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index_clean_dir = torch.stack([row, col], dim=0)

        ones_e = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
        deg = torch.zeros((num_nodes,), dtype=torch.float32, device=col.device)
        deg.index_add_(0, col, ones_e)

        ones_n = torch.ones((num_nodes,), dtype=torch.float32, device=batch.device)
        n_g = torch.zeros((num_graphs,), dtype=torch.float32, device=batch.device)
        n_g.index_add_(0, batch, ones_n)
        n_per_node = n_g[batch]
        comp = (n_per_node - 1.0 - deg).clamp(min=0.0)

        if node_ids is None:
            global_n_id = torch.arange(num_nodes, device=batch.device, dtype=torch.long)
        else:
            global_n_id = node_ids.to(device=batch.device, dtype=torch.long)
        if global_num_nodes_total is None:
            total_nodes = int(global_n_id.max().item()) + 1 if global_n_id.numel() > 0 else int(num_nodes)
        else:
            total_nodes = int(global_num_nodes_total)

        return BatchedGraphCache(
            edge_index_clean_dir=edge_index_clean_dir,
            batch=batch,
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            deg_clean=deg,
            comp_size=comp,
            n_per_node=n_per_node,
            global_n_id=global_n_id,
            global_num_nodes_total=total_nodes,
        )

    def complement_sum_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        neigh_sum = _scatter_add_rows(x[row], col, self.num_nodes)
        total_g = _scatter_add_graph_rows(x, self.batch.to(x.device), self.num_graphs)
        total_i = total_g[self.batch.to(x.device)]
        return total_i - x - neigh_sum

    def complement_mean_unweighted(self, x: torch.Tensor) -> torch.Tensor:
        comp_sum = self.complement_sum_unweighted(x)
        denom = self.comp_size.to(x.device).clamp(min=1.0).unsqueeze(1)
        return comp_sum / denom

    def complement_sum_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = _scatter_add_graph_rows(wx, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        neigh_wx = _scatter_add_rows(wx[row], col, self.num_nodes)
        return total_wx - wx - neigh_wx

    def complement_mean_weighted(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        row, col = self.edge_index_clean_dir.to(x.device)
        wx = w.unsqueeze(1) * x
        total_wx = _scatter_add_graph_rows(wx, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        total_w = _scatter_add_graph_scalar(w, self.batch.to(x.device), self.num_graphs)[self.batch.to(x.device)]
        neigh_wx = _scatter_add_rows(wx[row], col, self.num_nodes)
        neigh_w = _scatter_add_scalar(w[row], col, self.num_nodes)
        num = total_wx - wx - neigh_wx
        den = (total_w - w - neigh_w).clamp(min=1e-12).unsqueeze(1)
        return num / den


class RADEGATConvGC(_RADEGATConvBase):
    """
    Graph-classification GAT RADE operator.

    Attention moments are computed over each graph's local node set in the
    packed batch. Cross-graph pairs have pi=0 and are excluded from non-edge
    centering and RADE-OFS inference corrections. The sampled mode keeps
    clean-neighbor terms exact and samples graph-local non-neighbors with
    replacement.
    """

    def __init__(self, *args, ep_corr_clip: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_corr_clip = max(0.0, float(ep_corr_clip))

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
        self.graph_cache = graph_cache

    def _same_graph_chunk(self, src_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self.graph_cache is not None
        batch = self.graph_cache.batch.to(device)
        return batch.unsqueeze(1).eq(batch[src_idx].unsqueeze(0))

    def _attention_logit_shift(
        self,
        score_src: torch.Tensor,
        score_dst: torch.Tensor,
    ) -> torch.Tensor:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
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
            local_pair = self._same_graph_chunk(src_idx, score_src.device)
            logits = logits.masked_fill(~local_pair.unsqueeze(-1), -torch.inf)
            logits[src_idx, src_idx - start] = -torch.inf
            shift = torch.maximum(shift, logits.max(dim=1).values)

        return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift)).detach()

    def _sample_nonedge_sources_with_replacement(
        self,
        num_nodes: int,
        samples: int,
        device: torch.device,
        *,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = int(num_nodes)
        comp = self.graph_cache.comp_size.detach().to(dtype=torch.long, device=torch.device("cpu"))
        max_comp = int(comp.max().item()) if comp.numel() > 0 else 0
        sample_cols = min(max(1, int(samples)), max_comp)
        if N <= 1 or max_comp <= 0:
            src_empty = torch.empty((N, 0), dtype=torch.long, device=device)
            valid_empty = torch.empty((N, 0), dtype=torch.bool, device=device)
            scale_empty = torch.zeros((N,), dtype=torch.float32, device=device)
            return src_empty, valid_empty, scale_empty

        batch_cpu = self.graph_cache.batch.detach().to(dtype=torch.long, device=torch.device("cpu"))
        graph_nodes = [
            torch.nonzero(batch_cpu == graph_idx, as_tuple=False).view(-1)
            for graph_idx in range(int(self.graph_cache.num_graphs))
        ]
        node_pos = {}
        for nodes in graph_nodes:
            for pos, node in enumerate(nodes.tolist()):
                node_pos[int(node)] = int(pos)

        clean_sources = [set() for _ in range(N)]
        edge_row = self.graph_cache.edge_index_clean_dir[0].detach().to(dtype=torch.long, device=torch.device("cpu"))
        edge_col = self.graph_cache.edge_index_clean_dir[1].detach().to(dtype=torch.long, device=torch.device("cpu"))
        for src, dst in zip(edge_row.tolist(), edge_col.tolist()):
            clean_sources[int(dst)].add(int(src))

        target_counts = torch.clamp(comp, max=sample_cols)
        src_out = torch.zeros((N, sample_cols), dtype=torch.long)
        valid_out = torch.zeros((N, sample_cols), dtype=torch.bool)
        counts = torch.zeros((N,), dtype=torch.long)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.moment_seed) + int(seed_offset))

        for dst in range(N):
            target = int(target_counts[dst].item())
            if target <= 0:
                continue
            nodes = graph_nodes[int(batch_cpu[dst].item())]
            candidate_count = int(nodes.numel()) - 1
            if candidate_count <= 0:
                continue
            dst_pos = int(node_pos[dst])
            draw = max(sample_cols * 2, 64)
            for _ in range(32):
                if int(counts[dst].item()) >= target:
                    break
                raw = torch.randint(
                    low=0,
                    high=candidate_count,
                    size=(draw,),
                    generator=gen,
                    device=torch.device("cpu"),
                    dtype=torch.long,
                )
                cand = nodes[raw + (raw >= dst_pos).to(torch.long)]
                for src in cand.tolist():
                    if int(src) in clean_sources[dst]:
                        continue
                    pos = int(counts[dst].item())
                    if pos >= target or pos >= sample_cols:
                        break
                    src_out[dst, pos] = int(src)
                    valid_out[dst, pos] = True
                    counts[dst] += 1
                draw = min(max(draw * 2, 64), max(sample_cols * 16, 1024))

        scale = torch.zeros((N,), dtype=torch.float32)
        nonzero = counts > 0
        scale[nonzero] = comp.to(torch.float32)[nonzero] / counts.to(torch.float32)[nonzero]
        return src_out.to(device), valid_out.to(device), scale.to(device)

    def _attention_totals_sampled(
        self,
        h: torch.Tensor,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        *,
        detach_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = h.size(0)
        if N <= 1:
            score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
            logit_shift = self._attention_logit_shift(score_src, score_dst)
            zeros = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return zeros, zeros, score_src, score_dst, logit_shift

        dtype = h.dtype
        device = h.device
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        q_t = torch.as_tensor(q, dtype=dtype, device=device)
        keep_t = 1.0 - p_t

        score_src, score_dst = self._node_attention_scores(h, detach_params=detach_params)
        logit_shift = self._attention_logit_shift(score_src, score_dst)

        edge_row = self.graph_cache.edge_index_clean_dir[0].to(device)
        edge_col = self.graph_cache.edge_index_clean_dir[1].to(device)
        edge_logits = torch.nn.functional.leaky_relu(
            score_src[edge_row] + score_dst[edge_col],
            negative_slope=self.negative_slope,
        )
        edge_w = self._scaled_attention_weights(edge_logits, logit_shift, edge_col)
        total_mu = _scatter_add_rows(keep_t.expand_as(edge_w) * edge_w, edge_col, N)
        total_sigma = _scatter_add_rows((keep_t * p_t).expand_as(edge_w) * edge_w.square(), edge_col, N)

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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        h = self._project(x)
        self._last_h_for_moments = h.detach()
        N = h.size(0)
        use_ofs = self.rade_variant == "rade-ofs"

        if edge_index_keep is None and edge_index_add is None:
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

        row_all = torch.cat([row_k, row_a], dim=0)
        col_all = torch.cat([col_k, col_a], dim=0)
        alpha_all = self._edge_softmax(self._edge_logits(h, row_all, col_all), col_all, N)
        alpha_k = alpha_all[: row_k.numel()]
        alpha_a = alpha_all[row_k.numel():]

        if not self.ep_correction:
            out = self._aggregate(h, row_all, col_all, alpha_all)
            return self._finalize_heads(out)

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
                out_a = self._aggregate(h, row_a, col_a, alpha_a)
            else:
                mu_h, _ = self._nonedge_expectation_sum(h, p=p, q=q, centered=True)
                msg_a = alpha_a.unsqueeze(-1) * (h[row_a] - mu_h[col_a])
                out_a = _scatter_add_heads(msg_a, col_a, N)
        else:
            out_a = torch.zeros_like(out_k)

        return self._finalize_heads(out_k + out_a)

    def gat_denominator_cv2_proxy(
        self,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            if torch.is_tensor(p):
                return p.new_zeros((0,))
            return torch.zeros((0,), dtype=torch.float32)
        if self.heads != 1:
            raise ValueError("GAT denominator stability proxy currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        if torch.is_tensor(p):
            device = p.device
            dtype = p.dtype
        elif torch.is_tensor(q):
            device = q.device
            dtype = q.dtype
        else:
            h_ref = self._last_h_for_moments
            device = h_ref.device
            dtype = h_ref.dtype

        h = self._last_h_for_moments.to(device=device, dtype=dtype)
        total_mu, total_sigma, _, _, _ = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        return total_sigma / total_mu.square().clamp_min(float(eps))

    def gat_correction_gain_proxy(
        self,
        *,
        p: torch.Tensor | float,
        q: torch.Tensor | float,
        eps: float,
    ) -> torch.Tensor:
        if self._last_h_for_moments is None:
            if torch.is_tensor(p):
                return p.new_zeros((0,))
            return torch.zeros((0,), dtype=torch.float32)
        if self.heads != 1:
            raise ValueError("GAT correction-gain proxy currently supports heads=1.")
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        if torch.is_tensor(p):
            device = p.device
            dtype = p.dtype
        elif torch.is_tensor(q):
            device = q.device
            dtype = q.dtype
        else:
            h_ref = self._last_h_for_moments
            device = h_ref.device
            dtype = h_ref.dtype

        h = self._last_h_for_moments.to(device=device, dtype=dtype)
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row = self.graph_cache.edge_index_clean_dir[0].to(device)
        col = self.graph_cache.edge_index_clean_dir[1].to(device)
        if row.numel() == 0:
            return h.new_zeros((0,))

        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True)
        p_t = torch.as_tensor(p, dtype=dtype, device=device)
        e_keep, _, _ = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=1.0 - p_t,
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
        )
        return clean_alpha / e_keep.clamp_min(float(eps))

    def _nonedge_expectation_sum_sampled(
        self,
        h: torch.Tensor,
        p: float,
        q: float,
        *,
        centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = h.size(0)
        if N <= 1:
            num = torch.zeros((N, self.heads, self.out_per_head), dtype=h.dtype, device=h.device)
            den = torch.zeros((N, self.heads), dtype=h.dtype, device=h.device)
            return (num if centered else num), den

        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(h, p=p, q=q)
        dtype = h.dtype
        device = h.device
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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."
        N = m.size(0)
        if N <= 1:
            return torch.zeros_like(m_sq)

        device = m.device
        dtype = m.dtype
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

        scale_t = scale.to(dtype).view(N, 1)
        if self.rade_variant == "rade-of":
            den = scale_t.squeeze(1) * e_add.sum(dim=1)
            num = scale_t * (e_add.unsqueeze(-1) * m[src_idx]).sum(dim=1)
            mu_msg = num / den.clamp(min=float(eps)).unsqueeze(1)
            centered_sq = (m[src_idx] - mu_msg.unsqueeze(1)).square()
            return scale_t * (var_add.unsqueeze(-1) * centered_sq).sum(dim=1)

        return scale_t * (var_add.unsqueeze(-1) * m_sq[src_idx]).sum(dim=1)

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
        assert self.graph_cache is not None, "Call model.set_graph(...) before graph-classification GAT forward."

        h = self._last_h_for_moments.to(device=m.device, dtype=m.dtype)
        total_mu, total_sigma, score_src, score_dst, logit_shift = self._attention_totals(
            h,
            p=p,
            q=q,
            detach_params=True,
        )
        row = self.graph_cache.edge_index_clean_dir[0].to(m.device)
        col = self.graph_cache.edge_index_clean_dir[1].to(m.device)

        clean_alpha = self._clean_edge_alpha(h, row, col, detach_params=True).squeeze(-1)
        e_alpha, _, var_alpha = self._pair_moments_from_totals(
            row=row,
            col=col,
            pi=(1.0 - torch.as_tensor(p, dtype=m.dtype, device=m.device)),
            total_mu=total_mu,
            total_sigma=total_sigma,
            score_src=score_src,
            score_dst=score_dst,
            logit_shift=logit_shift,
        )
        e_alpha = e_alpha.squeeze(-1)
        var_alpha = var_alpha.squeeze(-1)
        factor = clean_alpha.square() * (var_alpha / (e_alpha.square() + float(eps)))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, m.size(0))

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


class RADEGCNConvGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
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
        self.graph_cache: Optional[BatchedGraphCache] = None

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) per batch before forward."
        x = self.lin(x)
        num_nodes = x.size(0)

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            ar = torch.arange(num_nodes, device=x.device)
            row_all = torch.cat([row, ar])
            col_all = torch.cat([col, ar])
            deg = _scatter_add_scalar(torch.ones(row_all.numel(), device=x.device, dtype=x.dtype), col_all, num_nodes)
            inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
            gamma = inv_sqrt[row_all] * inv_sqrt[col_all]
            out = _scatter_add_rows(gamma.unsqueeze(1) * x[row_all], col_all, num_nodes)

            if self.rade_variant == "rade-ofs" and (not self.training) and self.ep_correction and (q > 0.0):
                deg_clean = self.graph_cache.deg_clean.to(x.device)
                comp_size = self.graph_cache.comp_size.to(x.device)
                t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                comp_sum = self.graph_cache.complement_sum_weighted(x, w=t_b)
                out = out + (q * t_b).unsqueeze(1) * comp_sum

            if self.bias is not None:
                out = out + self.bias
            return out

        if edge_index_keep is None:
            row_k, col_k = self.graph_cache.edge_index_clean_dir.to(x.device)
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = col_k.new_empty((0,))

        ar = torch.arange(num_nodes, device=x.device)
        row_all = torch.cat([row_k, row_a, ar])
        col_all = torch.cat([col_k, col_a, ar])
        deg_aug = _scatter_add_scalar(torch.ones(row_all.numel(), device=x.device, dtype=x.dtype), col_all, num_nodes).clamp(min=1.0)
        inv_sqrt_aug = deg_aug.pow(-0.5)

        gamma_k = inv_sqrt_aug[row_k] * inv_sqrt_aug[col_k]
        gamma_a = inv_sqrt_aug[row_a] * inv_sqrt_aug[col_a] if row_a.numel() else None
        gamma_self = inv_sqrt_aug * inv_sqrt_aug

        deg_clean = self.graph_cache.deg_clean.to(x.device)
        comp_size = self.graph_cache.comp_size.to(x.device)
        deg_hat = (deg_clean + 1.0).clamp(min=1.0)
        inv_sqrt_hat = deg_hat.pow(-0.5)
        alpha_k = inv_sqrt_hat[row_k] * inv_sqrt_hat[col_k]
        alpha_self = inv_sqrt_hat * inv_sqrt_hat

        if self.ep_correction:
            t_a = _delta_E_inv_sqrt_caseA(deg_clean, comp_size, p=p, q=q)
            e_gamma_k = ((1.0 - p) * t_a[row_k] * t_a[col_k]).to(x.dtype)
            e_self = _delta_E_inv_self(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
            corr_k = alpha_k / e_gamma_k.clamp(min=self.ep_eps)
            out_k = _scatter_add_rows((gamma_k * corr_k).unsqueeze(1) * x[row_k], col_k, num_nodes)
            corr_self = alpha_self / e_self.clamp(min=self.ep_eps)
            out_self = (gamma_self * corr_self).unsqueeze(1) * x if self.correct_self_loop else gamma_self.unsqueeze(1) * x
        else:
            out_k = _scatter_add_rows(gamma_k.unsqueeze(1) * x[row_k], col_k, num_nodes)
            out_self = gamma_self.unsqueeze(1) * x

        if row_a.numel():
            sum_a = _scatter_add_rows(gamma_a.unsqueeze(1) * x[row_a], col_a, num_nodes)
            if self.rade_variant == "rade-ofs":
                out_a = sum_a
            else:
                t_b = _delta_E_inv_sqrt_caseB(deg_clean, comp_size, p=p, q=q).to(x.device, dtype=x.dtype)
                mu = self.graph_cache.complement_mean_weighted(x, w=t_b)
                sum_g = _scatter_add_scalar(gamma_a, col_a, num_nodes).unsqueeze(1)
                out_a = sum_a - mu * sum_g
        else:
            out_a = torch.zeros_like(out_k)

        out = out_k + out_a + out_self
        if self.bias is not None:
            out = out + self.bias
        return out

class RADEGINConvGC(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        *,
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
        self.graph_cache: Optional[BatchedGraphCache] = None

    def set_graph(self, graph_cache: BatchedGraphCache) -> None:
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
        assert self.graph_cache is not None, "Call model.set_graph(...) per batch before forward."
        num_nodes = x.size(0)

        if edge_index_keep is None and edge_index_add is None:
            row, col = edge_index_clean[0], edge_index_clean[1]
            neigh_sum = _scatter_add_rows(x[row], col, num_nodes)
            if self.rade_variant == "rade-ofs" and (not self.training) and self.ep_correction and (q > 0.0):
                comp_sum = self.graph_cache.complement_sum_unweighted(x)
                neigh_sum = neigh_sum + float(q) * comp_sum
            out = (1.0 + self.eps.to(x.device)) * x + neigh_sum
            return self.mlp(out)

        if edge_index_keep is None:
            row_k, col_k = self.graph_cache.edge_index_clean_dir.to(x.device)
        else:
            row_k, col_k = edge_index_keep[0], edge_index_keep[1]
        if edge_index_add is not None and edge_index_add.numel() > 0:
            row_a, col_a = edge_index_add[0], edge_index_add[1]
        else:
            row_a = row_k.new_empty((0,))
            col_a = row_k.new_empty((0,))

        if self.ep_correction:
            keep_prob = max(self.ep_eps, 1.0 - float(p))
            keep_sum = _scatter_add_rows(x[row_k], col_k, num_nodes) / keep_prob
        else:
            keep_prob = max(self.ep_eps, 1.0 - float(p))
            keep_sum = _scatter_add_rows(x[row_k], col_k, num_nodes) / keep_prob

        if row_a.numel() > 0:
            add_sum_raw = _scatter_add_rows(x[row_a], col_a, num_nodes)
            if self.rade_variant == "rade-ofs":
                add_sum = add_sum_raw
            else:
                mu = self.graph_cache.complement_mean_unweighted(x)
                add_deg = _scatter_add_scalar(
                    torch.ones(row_a.numel(), device=x.device, dtype=x.dtype),
                    col_a,
                    num_nodes,
                ).unsqueeze(1)
                add_sum = add_sum_raw - mu * add_deg
        else:
            add_sum = torch.zeros_like(keep_sum)

        out = (1.0 + self.eps.to(x.device)) * x + keep_sum + add_sum
        return self.mlp(out)

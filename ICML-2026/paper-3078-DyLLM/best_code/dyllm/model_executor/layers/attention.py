import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
import flash_attn_2_cuda as flash_attn_gpu

from dyllm.utils.context import get_context, set_context
from dyllm.utils.metadata import get_metadata
from dyllm.engine.cache_manager import CacheManager

from dyllm.attention_ops import attention_sparse_varlen, fused_offset_launch


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        threshold: float = 0.99,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.threshold = threshold
        self.context_cache = CacheManager(self.num_heads * self.head_dim)
        self.v_cache = CacheManager(self.num_kv_heads * self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_context()
        metadata = get_metadata()
        num_repeat = self.num_heads // self.num_kv_heads

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if ctx.is_full:

            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )

            c_cache = o.flatten(-2, -1).contiguous()  # sum(L), H * D
            self.context_cache.reset_full(
                c_cache, seq_ids=metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs
            )
            self.v_cache.reset_full(v.flatten(-2, -1), metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)

            self.context_cache.finish(metadata.finished_seqs)
            self.v_cache.finish(metadata.finished_seqs)
            return o

        else:
            is_q_pruned = ctx.idx_salient_row_k is not None

            v_cache = self.v_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_kv_heads, self.head_dim)
            torch.cuda.nvtx.range_pop()
            if is_q_pruned:
                v_delta = v - v_cache[ctx.idx_salient_row_k]  # K, H_kv, D
                v_cache[ctx.idx_salient_row_k] = v
            else:
                v_delta = v - v_cache[ctx.idx_salient_row]  # K, H_kv, D
                v_cache[ctx.idx_salient_row] = v

            o_salient = flash_attn_varlen_func(
                q[ctx.idx_salient_row],
                k,
                v_cache,
                cu_seqlens_q=ctx.cu_salientlens,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )

            v_delta_sparse = torch.zeros_like(k)  # [K, H_kv, D]

            if is_q_pruned:
                v_delta_sparse[ctx.idx_salient_row_k] = v_delta
                self.v_cache.scatter_update(metadata.running_seqs_tensor, ctx.idx_salient_row_k, v.flatten(-2, -1))
            else:
                v_delta_sparse[ctx.idx_salient_row] = v_delta
                self.v_cache.scatter_update(metadata.running_seqs_tensor, ctx.idx_salient_row, v.flatten(-2, -1))

            cos_sim_stats = torch.zeros((ctx.total_seqlen, 3), dtype=torch.float32, device=q.device)
            cos_sim_mask = torch.empty((ctx.total_seqlen,), dtype=torch.bool, device=q.device)

            idx_salient_row_k = (
                ctx.idx_salient_row_k
                if ctx.idx_salient_row_k is not None
                else torch.empty(0, dtype=torch.int32, device=q.device)
            )

            c_cache = (
                self.context_cache.get_seqs_block(metadata.running_seqs_tensor).view(-1, self.num_heads, self.head_dim)
                if is_q_pruned
                else self.context_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_heads, self.head_dim)
            )

            # new_c = flash_attn_varlen_func(
            #     q, k, v_cache, cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k, max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k, dropout_p=0.0, softmax_scale=self.scale, causal=False
            # )
            

            # cos_sim = F.cosine_similarity(c_cache.flatten(-2, -1), new_c.flatten(-2, -1), dim=-1)  # [L]
            # idxs = torch.nonzero(cos_sim < self.threshold, as_tuple=False).squeeze(-1).contiguous()
            # import pdb; pdb.set_trace()

            new_c = attention_sparse_varlen(
                q,
                k,
                v_delta_sparse,
                c_cache,
                o_salient,
                ctx.cu_seqlens_q.to(torch.int32),
                ctx.cu_seqlens_k.to(torch.int32),
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                ctx.total_seqlen,
                ctx.cu_salientlens.to(torch.int32),
                ctx.idx_salient_row.to(torch.int32),
                cos_sim_stats,
                cos_sim_mask,
                self.threshold,
                is_q_pruned,
                idx_salient_row_k.to(torch.int32),
            )

            idxs = torch.nonzero(cos_sim_mask, as_tuple=False).squeeze(-1).contiguous()
            num_valid = idxs.numel()

            cu_salientlens = torch.searchsorted(idxs, ctx.cu_seqlens_q).to(torch.int32)

            if ctx.idx_salient_row_k is not None:
                self.context_cache.reset_block(new_c.flatten(-2, -1), metadata.running_seqs_tensor)
                MAX_TOKENS = ctx.total_seqlen
                if getattr(self, "idx_salient_buffer", None) is None or self.idx_salient_buffer.size(0) < MAX_TOKENS:
                    self.idx_salient_buffer = torch.empty(MAX_TOKENS, dtype=torch.long, device=idxs.device)
                num_idxs = torch.tensor([num_valid], dtype=torch.int32, device=idxs.device)
                fused_offset_launch(
                    idxs.to(torch.int32),
                    num_idxs,
                    ctx.cu_promptlens,
                    cu_salientlens,
                    self.idx_salient_buffer,
                    MAX_TOKENS,
                )
                ctx.idx_salient_row_k = self.idx_salient_buffer[:num_valid]
            else:
                self.context_cache.reset_full(
                    new_c.flatten(-2, -1), metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs
                )

            ctx.idx_salient_row = idxs.to(torch.long)
            ctx.cu_salientlens = cu_salientlens

            self.v_cache.finish(metadata.finished_seqs)
            self.context_cache.finish(metadata.finished_seqs)

            return new_c.view(-1, self.num_heads, self.head_dim)

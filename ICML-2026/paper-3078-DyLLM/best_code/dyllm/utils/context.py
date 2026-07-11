from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class Context:
    is_full: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_q_cpu: Optional[List] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    cu_promptlens: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: Optional[torch.Tensor] = None
    context_lens_cpu: Optional[List] = None
    cu_updatedlens: Optional[torch.Tensor] = None
    cu_salientlens: Optional[torch.Tensor] = None
    idx_salient_row: Optional[torch.Tensor] = None
    idx_salient_row_k: Optional[torch.Tensor] = None
    idx_updated_row: Optional[torch.Tensor] = None
    cache_starts: Optional[torch.Tensor] = None
    binary_mask: Optional[torch.Tensor] = None
    total_seqlen_k: int = 0
    total_seqlen: int = 0
    positions_k: Optional[torch.Tensor] = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_full,
    cu_seqlens_q=None,
    cu_seqlens_q_cpu=None,
    cu_seqlens_k=None,
    cu_promptlens=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    context_lens=None,
    context_lens_cpu=None,
    cu_updatedlens=None,
    cu_salientlens=None,
    idx_salient_row=None,
    idx_salient_row_k=None,
    idx_updated_row=None,
    total_seqlen_k=0,
    total_seqlen=0,
    positions_k=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_full=is_full,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_cpu=cu_seqlens_q_cpu,
        cu_seqlens_k=cu_seqlens_k,
        cu_promptlens=cu_promptlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        context_lens=context_lens,
        context_lens_cpu=context_lens_cpu,
        cu_updatedlens=cu_updatedlens,
        cu_salientlens=cu_salientlens,
        idx_salient_row=idx_salient_row,
        idx_salient_row_k=idx_salient_row_k,
        idx_updated_row=idx_updated_row,
        total_seqlen=total_seqlen,
        total_seqlen_k=total_seqlen_k,
        positions_k=positions_k,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

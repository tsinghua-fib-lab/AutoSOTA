from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from dyllm.utils.context import get_context
from dyllm.cache import get_seqs_cuda, scatter_update_cuda, reset_full_cuda, get_block_cuda, reset_block_cuda


class CacheManager:

    def __init__(
        self,
        hidden_dim: int,
        max_num_seqs: int = 16384,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self._seq_to_start: dict[int, int] = {}
        self._seq_to_len: dict[int, int] = {}
        self.cache = torch.empty((0, hidden_dim), dtype=dtype, device=device)
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.seq_start_gpu = torch.full((max_num_seqs,), -1, dtype=torch.long, device=device)
        self.seq_len_gpu = torch.zeros((max_num_seqs,), dtype=torch.long, device=device)

    @property
    def num_total_seqs(self) -> int:
        return len(self._seq_to_start)

    def select_seq_range(self, seq_id: int) -> torch.Tensor:
        start = self._seq_to_start[seq_id]
        length = self._seq_to_len[seq_id]
        return start, start + length

    def get_seqs(self, seq_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        return get_seqs_cuda(
            self.cache, self.seq_start_gpu, ctx.cu_seqlens_k, seq_ids, max(ctx.total_seqlen, ctx.total_seqlen_k)
        )

    def get_seqs_block(self, seq_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        return get_block_cuda(
            self.cache, self.seq_start_gpu, ctx.cu_seqlens_k, ctx.cu_seqlens_q, seq_ids, ctx.total_seqlen
        )

    def get_start_indices(self, seq_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self._seq_to_start[s] for s in seq_ids], dtype=torch.long)

    def reset_seq(self, c: torch.Tensor, seq_id):
        """
        c: [L, D]
        seq_id: int
        """
        if seq_id in self._seq_to_start:
            start, end = self.select_seq_range(seq_id)
            if end - start == c.size(0):
                self.cache[start:end] = c
            else:
                self.remove_seq(seq_id)
                self.add_seq(c, seq_id)
        else:
            self.add_seq(c, seq_id)

    def add_seq(self, c: torch.Tensor, seq_id: int):
        """
        c: [L, D]
        seq_id: int
        """
        if self.cache.numel() == 0:
            self.cache = c.contiguous()
            self._seq_to_start[seq_id] = 0
            self.seq_start_gpu[seq_id] = 0
            self._seq_to_len[seq_id] = c.size(0)
            self.seq_len_gpu[seq_id] = c.size(0)
        else:
            start = self.cache.size(0)
            self.cache = torch.cat([self.cache, c], dim=0)
            self._seq_to_start[seq_id] = start
            self.seq_start_gpu[seq_id] = start
            self._seq_to_len[seq_id] = c.size(0)
            self.seq_len_gpu[seq_id] = c.size(0)

    def remove_seq(self, seq_id: int):
        if seq_id not in self._seq_to_start:
            return

        start, end = self.select_seq_range(seq_id)
        length = end - start

        if start > 0:
            self.cache = torch.cat([self.cache[:start], self.cache[end:]], dim=0)
        else:
            self.cache = self.cache[end:]

        ids_to_update = []
        for other_id, other_start in self._seq_to_start.items():
            if other_id == seq_id:
                continue
            if other_start > start:
                self._seq_to_start[other_id] -= length
                ids_to_update.append(other_id)

        if ids_to_update:
            ids_tensor = torch.tensor(ids_to_update, device=self.device, dtype=torch.long)
            self.seq_start_gpu[ids_tensor] -= length

        del self._seq_to_start[seq_id]
        del self._seq_to_len[seq_id]
        self.seq_start_gpu[seq_id] = -1
        self.seq_len_gpu[seq_id] = 0

    def _vectorized_allocate_new_seqs(self, new_ids: list[int], new_lengths: list[int]):
        total_new_len = sum(new_lengths)
        if total_new_len == 0:
            return

        new_ids_gpu = torch.tensor(new_ids, device=self.device, dtype=torch.long)
        new_lengths_gpu = torch.tensor(new_lengths, device=self.device, dtype=torch.long)

        current_cache_size = self.cache.size(0)
        new_space = torch.empty((total_new_len, self.hidden_dim), dtype=self.dtype, device=self.device)
        self.cache = torch.cat([self.cache, new_space], dim=0)

        relative_starts = torch.cumsum(new_lengths_gpu, dim=0) - new_lengths_gpu
        absolute_starts = relative_starts + current_cache_size

        self.seq_start_gpu[new_ids_gpu] = absolute_starts
        self.seq_len_gpu[new_ids_gpu] = new_lengths_gpu

        for i, seq_id in enumerate(new_ids):
            self._seq_to_start[seq_id] = absolute_starts[i].item()
            self._seq_to_len[seq_id] = new_lengths[i]

    def reset_full(self, c: torch.Tensor, seq_ids: torch.Tensor, seq_ids_list: Optional[List[int]] = None):
        ctx = get_context()
        if not c.is_contiguous():
            c = c.contiguous()

        if seq_ids_list is None:
            seq_ids_list = seq_ids.cpu().tolist()

        new_ids_to_alloc = []
        new_lens_to_alloc = []
        batch_lens_list = ctx.context_lens_cpu

        for i, seq_id in enumerate(seq_ids_list):
            if seq_id not in self._seq_to_start:
                new_ids_to_alloc.append(seq_id)
                new_lens_to_alloc.append(batch_lens_list[i])

        if new_ids_to_alloc:
            self._vectorized_allocate_new_seqs(new_ids_to_alloc, new_lens_to_alloc)

        reset_full_cuda(self.cache, c, self.seq_start_gpu, ctx.cu_seqlens_q, seq_ids, ctx.total_seqlen)

    def reset_block(self, c: torch.Tensor, seq_ids: torch.Tensor):
        ctx = get_context()
        if not c.is_contiguous():
            c = c.contiguous()
        reset_block_cuda(
            self.cache, c, self.seq_start_gpu, ctx.cu_seqlens_k, ctx.cu_seqlens_q, seq_ids, ctx.total_seqlen
        )

    def get_start_indices(self, seq_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self._seq_to_start[s] for s in seq_ids], dtype=torch.long)

    def scatter_update(self, seq_ids: torch.Tensor, row_idx: torch.Tensor, new: torch.Tensor):
        ctx = get_context()
        scatter_update_cuda(self.cache, new, row_idx, ctx.cu_seqlens_k, self.seq_start_gpu, seq_ids)

    def finish(self, finished_seq_ids: List[int]):
        if len(finished_seq_ids) == 0:
            return False
        for seq_id in sorted(finished_seq_ids, key=lambda x: self._seq_to_start[x], reverse=True):
            self.remove_seq(seq_id)
        return True

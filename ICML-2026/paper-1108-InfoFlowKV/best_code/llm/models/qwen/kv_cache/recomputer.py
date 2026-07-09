"""KV cache recomputation with selective updates."""

import math
import torch
import torch.nn.functional as F
import numpy as np
import time
from bisect import bisect_right
from itertools import chain
from typing import List, Tuple, Optional, Any
from transformers.cache_utils import DynamicCache

import flashinfer

from .base import KVCacheData


class KVCacheRecomputer:
    """
    Selectively recompute KV cache at important positions.
    
    Strategy:
    - Layer 0: Recompute full KV (need correct hidden states for subsequent layers)
    - Layer 1+: Only recompute at selected indices
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_type: str = "qwen",
        num_recompute_chunks: int = 2,
        recompute_attention_mode: str = "flashinfer",
    ):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            model_type: Model type ("qwen", "glm", "llama")
            num_recompute_chunks: Number of chunks for chunked attention optimization (default 2)
            recompute_attention_mode: Attention backend for sparse recompute:
                - "flashinfer": Use FlashInfer kernel (default, fastest)
                - "sdpa": Use PyTorch SDPA (auto-selects flash/math)
                - "math": Use PyTorch SDPA with forced math backend (for comparison)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.device = next(model.parameters()).device
        self.num_recompute_chunks = num_recompute_chunks
        self.recompute_attention_mode = recompute_attention_mode.lower()

        # Get model config
        config = model.config
        self.num_layers = getattr(config, "num_layers",
                                  getattr(config, "num_hidden_layers", 0))
        self.num_heads = getattr(config, "num_attention_heads", 32)
        self.num_kv_heads = getattr(config, "multi_query_group_num",
                                    getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = getattr(config, "hidden_size", 0) // max(1, self.num_heads)
        self.hidden_size = getattr(config, "hidden_size", 0)

    @torch.no_grad()
    def recompute(
        self,
        kv_data: KVCacheData,
        recompute_indices: np.ndarray,
        **kwargs,
    ) -> KVCacheData:
        """
        Recompute KV cache at specified indices.

        Args:
            kv_data: Extracted KV cache data (must contain input_ids)
            recompute_indices: Positions to recompute (numpy array or tensor)

        Returns:
            Updated KVCacheData with recomputed entries
        """
        if len(recompute_indices) == 0:
            return kv_data

        device = self.device
        dtype = self.model.dtype
        
        # Convert to numpy array first if it's a list
        if isinstance(recompute_indices, list):
            recompute_indices = np.array(recompute_indices)
        
        # Convert indices to tensor
        if isinstance(recompute_indices, np.ndarray):
            recompute_indices = torch.from_numpy(recompute_indices).to(device).long()
        else:
            recompute_indices = recompute_indices.to(device).long()
        
        descending = kwargs.get('descending', False)
        recompute_indices = torch.unique(recompute_indices, sorted=True)
        if descending:
            recompute_indices = recompute_indices.flip(0)
        K = recompute_indices.numel()

        # Pre-compute chunks for efficient sparse attention (done ONCE, used for all 48 layers)
        # Get T from cache
        if isinstance(kv_data.past_key_values, DynamicCache):
            T_cache = kv_data.past_key_values.key_cache[0].size(2)
        else:
            T_cache = kv_data.past_key_values[0][0].size(2)

        num_chunks = self.num_recompute_chunks
        if num_chunks > 1 and K > num_chunks:
            chunks = self._prepare_chunks(recompute_indices, num_chunks)
        else:
            chunks = None  # Fallback to non-chunked

        # Get input_ids and compute embeddings
        input_ids = kv_data.input_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        
        batch_size, seq_len = input_ids.shape
        
        # Compute input embeddings from input_ids
        embed_layer = self.model.get_input_embeddings()
        hidden_states = embed_layer(input_ids).to(dtype)  # [B, T, H]
        
        # Compute RoPE embeddings (position_embeddings)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cos_full, sin_full = self.model.model.rotary_emb(hidden_states, position_ids)

        # Get the cache (DynamicCache)
        cache = kv_data.past_key_values

        # Get model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'encoder'):
            layers = self.model.transformer.encoder.layers
        else:
            raise ValueError("Cannot find model layers")

        for layer_idx in range(self.num_layers):
            layer = layers[layer_idx]
            
            # Get cache for this layer
            if isinstance(cache, DynamicCache):
                k_cache = cache.key_cache[layer_idx]
                v_cache = cache.value_cache[layer_idx]
            else:
                k_cache, v_cache = cache[layer_idx]

            if layer_idx == 0:
                # Layer 0: Recompute full KV
                hidden_states, k_cache, v_cache = self._recompute_layer_full(
                    layer, hidden_states, k_cache, v_cache, cos_full, sin_full,
                    recompute_indices, batch_size, seq_len, chunks
                )
            else:
                # Layer 1+: Only recompute at selected indices
                hidden_states, k_cache, v_cache = self._recompute_layer_sparse(
                    layer, hidden_states, k_cache, v_cache, cos_full, sin_full,
                    recompute_indices, batch_size, K, chunks
                )

            # Update cache in place
            if isinstance(cache, DynamicCache):
                cache.key_cache[layer_idx] = k_cache
                cache.value_cache[layer_idx] = v_cache
            else:
                cache[layer_idx] = (k_cache, v_cache)

        return KVCacheData(
            past_key_values=cache,
            input_ids=input_ids,
            attention_mask=kv_data.attention_mask,
            chunk_lens=kv_data.chunk_lens,
        )

    def _recompute_layer_full(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        recompute_indices: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        chunks: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute layer with full sequence.

        Args:
            recompute_indices: If None, return full hidden_states (CacheBlend Layer 0).
                              If provided, return hidden_states subset at these indices.

        Returns:
            (hidden_states, updated_k_cache, updated_v_cache)
        """
        dtype = hidden_states.dtype
        attn = layer.self_attn

        # Layer norm
        normed = layer.input_layernorm(hidden_states)

        # Compute Q, K, V for all positions
        q = attn.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = attn.k_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply Q/K norms if present
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = attn.k_norm(k)

        # Apply RoPE
        q = self._apply_rope(q, cos_full, sin_full)
        k = self._apply_rope(k, cos_full, sin_full)

        # Update cache with new K, V
        k_cache = k.to(dtype)
        v_cache = v.to(dtype)

        if recompute_indices is not None:
            # Subset for next layer
            q_for_attn = q[:, :, recompute_indices, :]
            query_positions = recompute_indices
            hidden_states_for_residual = hidden_states[:, recompute_indices, :]
        else:
            # Full attention (CacheBlend Layer 0)
            q_for_attn = q
            query_positions = None
            hidden_states_for_residual = hidden_states

        # Compute attention
        attn_output = self._compute_attention(
            q_for_attn, k_cache, v_cache, query_positions, chunks
        )

        # Project output
        output_len = attn_output.size(2)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, output_len, self.hidden_size
        )
        attn_output = attn.o_proj(attn_output).to(dtype)

        # Residual connection
        hidden_states_out = hidden_states_for_residual + attn_output

        # MLP
        residual = hidden_states_out
        mlp_out = layer.post_attention_layernorm(hidden_states_out)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states_out = residual + mlp_out

        return hidden_states_out, k_cache, v_cache

    def _recompute_layer_sparse(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        recompute_indices: torch.Tensor,
        batch_size: int,
        K: int,
        chunks: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute layer at sparse positions only.

        Args:
            hidden_states: [B, K, H] hidden states at recompute positions

        Returns:
            (hidden_states_out, updated_k_cache, updated_v_cache)
        """
        dtype = hidden_states.dtype
        attn = layer.self_attn

        # Layer norm
        normed = layer.input_layernorm(hidden_states)

        # Compute Q, K, V only at recompute positions
        q = attn.q_proj(normed).view(batch_size, K, self.num_heads, self.head_dim)
        k_new = attn.k_proj(normed).view(batch_size, K, self.num_kv_heads, self.head_dim)
        v_new = attn.v_proj(normed).view(batch_size, K, self.num_kv_heads, self.head_dim)

        # Transpose to [B, H, K, D]
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        # Apply Q/K norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k_new = attn.k_norm(k_new)

        # Apply RoPE at correct positions
        q = self._apply_rope(q, cos_full, sin_full, recompute_indices)
        k_new = self._apply_rope(k_new, cos_full, sin_full, recompute_indices)

        # Update cache at recompute positions
        k_cache = k_cache.index_copy(2, recompute_indices, k_new.to(dtype))
        v_cache = v_cache.index_copy(2, recompute_indices, v_new.to(dtype))

        # Compute attention
        attn_output = self._compute_attention(q, k_cache, v_cache, recompute_indices, chunks)

        # Project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, K, self.hidden_size)
        attn_output = attn.o_proj(attn_output).to(dtype)

        # Residual connection
        hidden_states = hidden_states + attn_output

        # MLP
        residual = hidden_states
        mlp_out = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states = residual + mlp_out

        return hidden_states, k_cache, v_cache

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        """
        Apply RoPE to tensor x (Qwen style).
        
        Args:
            x: [B, H, T, D] tensor
            cos: [B, T_total, D] or [1, T_total, D] cosine values
            sin: [B, T_total, D] or [1, T_total, D] sine values
            position_ids: [K] or [B, K] positions to select
            unsqueeze_dim: dimension to unsqueeze (default 1 for head dimension)
            
        Returns:
            [B, H, T, D] tensor with RoPE applied
        """
        if position_ids is not None:
            if position_ids.dim() == 1:  # [K]
                cos = cos.index_select(1, position_ids)
                sin = sin.index_select(1, position_ids)
            elif position_ids.dim() == 2:  # [B, K]
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)  # [B, K, D]
                cos = torch.gather(cos, 1, idx)
                sin = torch.gather(sin, 1, idx)
        
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    def _remove_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        """
        Remove RoPE from tensor x (Qwen style).
        
        Args:
            x: [B, H, T, D] tensor with RoPE applied
            cos: [B, T_total, D] or [1, T_total, D] cosine values
            sin: [B, T_total, D] or [1, T_total, D] sine values
            position_ids: [K] or [B, K] positions to select
            unsqueeze_dim: dimension to unsqueeze (default 1 for head dimension)
            
        Returns:
            [B, H, T, D] tensor with RoPE removed
        """
        if position_ids is not None:
            if position_ids.dim() == 1:  # [K]
                cos = cos.index_select(1, position_ids)
                sin = sin.index_select(1, position_ids)
            elif position_ids.dim() == 2:  # [B, K]
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)  # [B, K, D]
                cos = torch.gather(cos, 1, idx)
                sin = torch.gather(sin, 1, idx)
        
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        
        x = (x * cos) - (self._rotate_half(x) * sin)
        return x

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
        chunks: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """
        Compute attention with causal mask.

        Args:
            q: [B, H, K, D] or [B, H, T, D] query
            k: [B, H, T, D] full key cache
            v: [B, H, T, D] full value cache
            query_positions: [K] positions of queries (for causal mask), or None for full causal
            chunks: Pre-computed chunk metadata for efficient sparse attention

        Returns:
            [B, H, K, D] or [B, H, T, D] attention output
        """
        # If query_positions is None, use full causal attention (already fast with SDPA)
        if query_positions is None:
            # Handle GQA for full causal
            if self.num_kv_heads < self.num_heads:
                num_groups = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(num_groups, dim=1)
                v = v.repeat_interleave(num_groups, dim=1)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
            return attn_output

        # Sparse attention: route based on recompute_attention_mode
        mode = self.recompute_attention_mode

        if mode == "flashinfer":
            # Use FlashInfer kernel (fastest)
            if chunks is not None:
                return self._compute_attention_chunked(q, k, v, chunks)
            return self._compute_attention_flashinfer(q, k, v, query_positions)
        elif mode in ("sdpa", "math"):
            # Use PyTorch SDPA (math forces math backend, sdpa auto-selects)
            if chunks is not None:
                return self._compute_attention_chunked_sdpa(q, k, v, chunks)
            return self._compute_attention_sdpa(q, k, v, query_positions)
        else:
            raise ValueError(f"Unknown recompute_attention_mode: {mode}. Use 'flashinfer', 'sdpa', or 'math'")

    def _prepare_chunks(
        self,
        query_positions: torch.Tensor,
        num_chunks: int = 2,
    ) -> List[dict]:
        """
        Pre-compute chunk metadata for chunked causal attention.

        Splits query positions by position value, so earlier positions are in
        earlier chunks. Each chunk only needs KV up to its max query position.

        Args:
            query_positions: [K] positions to recompute (already sorted)
            num_chunks: Number of chunks (default 2)

        Returns:
            List of chunk metadata dicts
        """
        K = query_positions.numel()
        device = query_positions.device

        # query_positions is already sorted, so we can split directly
        chunk_size = (K + num_chunks - 1) // num_chunks
        chunks = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, K)
            if start >= end:
                break

            # Indices into the original query array
            chunk_indices = torch.arange(start, end, device=device)
            chunk_positions = query_positions[start:end]
            max_pos = chunk_positions.max().item() + 1  # +1 because we slice [:max_pos]

            # Causal mask for this chunk: [chunk_K, max_pos]
            key_positions = torch.arange(max_pos, device=device)
            chunk_mask = chunk_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

            chunks.append({
                'indices': chunk_indices,      # indices into q tensor
                'max_pos': max_pos,            # KV cache slice end
                'mask': chunk_mask,            # [chunk_K, max_pos] causal mask
            })

        return chunks

    def _compute_attention_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunks: List[dict],
    ) -> torch.Tensor:
        """
        Compute sparse causal attention using chunked processing.

        Each chunk only loads KV up to its max query position, avoiding
        loading and computing attention for positions that would be masked.

        Args:
            q: [B, H_q, K, D] queries at sparse positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            chunks: Pre-computed chunk metadata from _prepare_chunks()

        Returns:
            [B, H_q, K, D] attention output
        """
        B, H_q, K, D = q.shape
        output = torch.empty_like(q)

        for chunk in chunks:
            idx = chunk['indices']
            max_pos = chunk['max_pos']
            mask = chunk['mask']

            # Extract queries for this chunk
            q_chunk = q[:, :, idx, :].squeeze(0).permute(1, 0, 2).contiguous()  # [chunk_K, H_q, D]

            # OPTIMIZATION: Only load KV up to max_pos (not full T)
            k_chunk = k[:, :, :max_pos, :].squeeze(0).permute(1, 0, 2).contiguous()  # [max_pos, H_kv, D]
            v_chunk = v[:, :, :max_pos, :].squeeze(0).permute(1, 0, 2).contiguous()  # [max_pos, H_kv, D]

            # FlashInfer attention with smaller mask
            attn_chunk = flashinfer.single_prefill_with_kv_cache(
                q_chunk, k_chunk, v_chunk,
                custom_mask=mask,
                causal=False,
                kv_layout="NHD",
            )  # [chunk_K, H_q, D]

            # Place result back
            attn_chunk = attn_chunk.permute(1, 0, 2).unsqueeze(0)  # [1, H_q, chunk_K, D]
            output[:, :, idx, :] = attn_chunk

        return output

    def _compute_attention_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention using FlashInfer (non-chunked fallback).

        Args:
            q: [B, H_q, K, D] query at sparse positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries

        Returns:
            [B, H_q, K, D] attention output
        """
        T = k.size(2)
        device = q.device

        q_fi = q.squeeze(0).permute(1, 0, 2).contiguous()  # [K, H_q, D]
        k_fi = k.squeeze(0).permute(1, 0, 2).contiguous()  # [T, H_kv, D]
        v_fi = v.squeeze(0).permute(1, 0, 2).contiguous()  # [T, H_kv, D]

        # Create causal mask for sparse positions: [K, T]
        key_positions = torch.arange(T, device=device)
        causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

        attn_output = flashinfer.single_prefill_with_kv_cache(
            q_fi, k_fi, v_fi,
            custom_mask=causal_mask,
            causal=False,
            kv_layout="NHD",
        )

        return attn_output.permute(1, 0, 2).unsqueeze(0)

    def _compute_attention_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention using SDPA with configurable backend (math/flash/auto).

        Args:
            q: [B, H_q, K, D] query at sparse positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries

        Returns:
            [B, H_q, K, D] attention output
        """
        B, H_q, K, D = q.shape
        T = k.size(2)
        H_kv = k.size(1)
        device = q.device

        # Handle GQA: expand K, V to match Q heads
        if H_kv < H_q:
            num_groups = H_q // H_kv
            k = k.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]
            v = v.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]

        # Create causal mask for SDPA: [K, T] -> [1, 1, K, T]
        # SDPA expects: True = attend, False = mask out
        key_positions = torch.arange(T, device=device)
        causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)  # [K, T]
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, K, T] - let SDPA broadcast

        # Use SDPA with backend control
        if self.recompute_attention_mode == "math":
            # Force math backend
            with torch.nn.attention.sdpa_kernel(
                backends=[torch.nn.attention.SDPBackend.MATH]
            ):
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            # Auto-select (flash if available, else math)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        return attn_output

    def _compute_attention_chunked_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunks: List[dict],
    ) -> torch.Tensor:
        """
        Compute sparse causal attention using chunked processing with SDPA.

        Args:
            q: [B, H_q, K, D] queries at sparse positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            chunks: Pre-computed chunk metadata from _prepare_chunks()

        Returns:
            [B, H_q, K, D] attention output
        """
        B, H_q, K, D = q.shape
        H_kv = k.size(1)
        output = torch.empty_like(q)

        # Handle GQA: expand K, V to match Q heads
        if H_kv < H_q:
            num_groups = H_q // H_kv
            k = k.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]
            v = v.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]

        for chunk in chunks:
            idx = chunk['indices']
            max_pos = chunk['max_pos']
            mask = chunk['mask']  # [chunk_K, max_pos]
            chunk_K = idx.size(0)

            # Extract queries for this chunk
            q_chunk = q[:, :, idx, :]  # [B, H_q, chunk_K, D]

            # OPTIMIZATION: Only load KV up to max_pos
            k_chunk = k[:, :, :max_pos, :]  # [B, H_q, max_pos, D]
            v_chunk = v[:, :, :max_pos, :]  # [B, H_q, max_pos, D]

            # Mask for SDPA: [chunk_K, max_pos] -> [1, 1, chunk_K, max_pos] - let SDPA broadcast
            attn_mask = mask.unsqueeze(0).unsqueeze(0)

            # Use SDPA with backend control
            if self.recompute_attention_mode == "math":
                with torch.nn.attention.sdpa_kernel(
                    backends=[torch.nn.attention.SDPBackend.MATH]
                ):
                    attn_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=attn_mask, is_causal=False)
            else:
                attn_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=attn_mask, is_causal=False)

            output[:, :, idx, :] = attn_chunk

        return output

    @torch.no_grad()
    def recompute_cacheblend(
        self,
        kv_data: KVCacheData,
        recompute_ratio: float = 0.15,
    ) -> KVCacheData:
        """
        CacheBlend recomputation strategy:
        - Layer 0: Full KV recompute for all positions, no subsetting
        - Layer 1: Full KV recompute, select top positions by V diff, narrow hidden states
        - Layer 2+: Selective recompute at selected positions from layer 1
        
        Args:
            kv_data: Extracted KV cache data
            recompute_ratio: Ratio of positions to select in layer 1 (default 0.15)
            
        Returns:
            KVCacheData with updated cache
        """
        device = self.device
        dtype = self.model.dtype
        
        # Extract from kv_data
        past_key_values = kv_data.past_key_values
        input_ids = kv_data.input_ids
        attention_mask = kv_data.attention_mask
        
        # Convert to DynamicCache if needed
        if not isinstance(past_key_values, DynamicCache):
            cache = DynamicCache()
            cache.key_cache = [k for k, v in past_key_values]
            cache.value_cache = [v for k, v in past_key_values]
        else:
            cache = past_key_values
            
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        
        batch_size, T = input_ids.shape
        seq_len_cache = cache.key_cache[0].size(2)
        assert T == seq_len_cache, f"input_ids length {T} must match cached KV length {seq_len_cache}"
        
        # Get input embeddings
        embed_layer = self.model.get_input_embeddings()
        hidden_states = embed_layer(input_ids).to(dtype)  # [B, T, H]
        
        # Get RoPE embeddings
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        
        # Get model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Cannot find model layers")
        
        # Track selected positions for layers 2+
        selected_indices = None
        
        for layer_idx in range(self.num_layers):
            layer = layers[layer_idx]
            k_cache = cache.key_cache[layer_idx]
            v_cache = cache.value_cache[layer_idx]
            
            if layer_idx == 0:
                # Layer 0: Full KV recompute, no subsetting (recompute_indices=None)
                hidden_states, k_cache, v_cache = self._recompute_layer_full(
                    layer, hidden_states, k_cache, v_cache, cos, sin,
                    recompute_indices=None, batch_size=batch_size, seq_len=T
                )
                
            elif layer_idx == 1:
                # Layer 1: Full KV recompute + select top positions by V diff
                attn = layer.self_attn
                normed = layer.input_layernorm(hidden_states)
                
                # Compute K, V for all positions
                k_new = attn.k_proj(normed).view(batch_size, T, self.num_kv_heads, self.head_dim)
                v_new = attn.v_proj(normed).view(batch_size, T, self.num_kv_heads, self.head_dim)
                
                k_new = k_new.transpose(1, 2).contiguous()  # [B, Hk, T, D]
                v_new = v_new.transpose(1, 2).contiguous()  # [B, Hk, T, D]
                
                # Apply K norm if present
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k_new = attn.k_norm(k_new)
                
                k_new = self._apply_rope(k_new, cos, sin)
                
                # Compare v_old vs v_new to select top positions
                dims_to_average = [i for i in range(v_new.dim()) if i != 2]
                diff_per_token = torch.mean((v_new - v_cache) ** 2, dim=dims_to_average)  # [T]
                num_selected = max(1, int(T * recompute_ratio))
                top_values, top_indices = torch.topk(diff_per_token, num_selected)
                selected_indices, _ = torch.sort(top_indices)
                
                # Update full cache for layer 1
                k_cache.copy_(k_new.to(dtype))
                v_cache.copy_(v_new.to(dtype))
                
                # Narrow to selected positions
                normed = normed[:, selected_indices, :]
                hidden_states = hidden_states[:, selected_indices, :]  # [B, K, H]
                K = selected_indices.numel()
                
                # Compute Q only for selected positions
                q = attn.q_proj(normed).view(batch_size, K, self.num_heads, self.head_dim)
                q = q.transpose(1, 2).contiguous()  # [B, Hq, K, D]
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q = attn.q_norm(q)
                q = self._apply_rope(q, cos, sin, selected_indices)
                
                # Attention for selected positions
                attn_output = self._compute_attention(q, k_cache, v_cache, selected_indices)
                
                # Project output
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, K, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)
                
                # Residual + MLP
                hidden_states = hidden_states + attn_output
                residual = hidden_states
                mlp_out = layer.post_attention_layernorm(hidden_states)
                mlp_out = layer.mlp(mlp_out).to(dtype)
                hidden_states = residual + mlp_out
                
            else:
                # Layer 2+: Selective recompute at selected positions only
                K = selected_indices.numel()
                hidden_states, k_cache, v_cache = self._recompute_layer_sparse(
                    layer, hidden_states, k_cache, v_cache, cos, sin,
                    selected_indices, batch_size, K
                )
            
            # Update cache
            cache.key_cache[layer_idx] = k_cache
            cache.value_cache[layer_idx] = v_cache
        
        # Build attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, T, dtype=torch.long, device=device)
        
        # Return KVCacheData
        return KVCacheData(
            past_key_values=cache,
            input_ids=input_ids,
            attention_mask=kv_data.attention_mask,
            chunk_lens=kv_data.chunk_lens,
        )

    @torch.no_grad()
    def recompute_at_positions(
        self,
        kv_data: KVCacheData,
        positions: np.ndarray,
    ) -> KVCacheData:
        """
        Legacy method: wrapper around recompute() for backward compatibility.
        
        Args:
            kv_data: Original KV cache data
            positions: Positions to recompute
            
        Returns:
            Updated KVCacheData
        """
        return self.recompute(kv_data, positions)

    @torch.no_grad()
    def reorder_and_rebase_kv(
        self,
        kv_data: KVCacheData,
        important_idx: List[int],
        put_higher_ratio_to_tail: bool = True,
    ) -> KVCacheData:
        """
        Reorder KV cache chunks based on importance ratio and rebase RoPE positions.
        
        This method efficiently reorders cache chunks using views (no data copy during split).
        Only the final reordered cache requires actual memory allocation.
        
        Args:
            kv_data: Original KV cache data with multiple chunks
            important_idx: List of important token positions (global indices)
            put_higher_ratio_to_tail: If True, put chunks with higher importance ratio at the tail
            
        Returns:
            Reordered KVCacheData with rebased RoPE positions
        """
        start_time = time.perf_counter()
        device = self.device
        
        # Extract information from kv_data
        seq_lens = kv_data.chunk_lens if isinstance(kv_data.chunk_lens, list) else [kv_data.chunk_lens]
        input_ids = kv_data.input_ids.squeeze(0).tolist()  # [total_len]
        
        cache = kv_data.past_key_values
        num_chunks = len(seq_lens)
        num_layers = self.num_layers
        
        # Split input_ids into chunks 
        ids_each = []
        offset = 0
        for L in seq_lens:
            ids_each.append(input_ids[offset:offset + int(L)])
            offset += int(L)
        
        #  Calculate chunk boundaries for efficient slicing
        chunk_starts = [0]
        for L in seq_lens[:-1]:
            chunk_starts.append(chunk_starts[-1] + int(L))
        
        # Calculate importance ratios for each chunk
        pref_end, acc = [], 0
        for L in seq_lens:
            acc += int(L)
            pref_end.append(acc)
        
        counts = [0] * num_chunks
        for idx in important_idx:
            c = bisect_right(pref_end, idx)
            if 0 <= c < num_chunks:
                counts[c] += 1
        

        ratios = [counts[i] / max(1, int(seq_lens[i])) for i in range(num_chunks)]
        
        # Qwen prefix: one dummy token (BOS/EOS)
        prefix_len = 1
        original_prefix_ids = ids_each[0][:prefix_len]
        
        # Determine reordering
        if num_chunks <= 1:
            order = list(range(num_chunks))
        else:
            if put_higher_ratio_to_tail:
                order = sorted(range(num_chunks), key=lambda i: (ratios[i], i))
            else:
                order = sorted(range(num_chunks), key=lambda i: (-ratios[i], i))
        
        # Reorder metadata 
        new_seq_lens = [int(seq_lens[i]) for i in order]
        new_ids_each = [ids_each[i] for i in order]
        
        # Handle original chunk 0 prefix
        original_chunk_0_idx = order.index(0) if 0 in order else -1
        if original_chunk_0_idx >= 0:
            new_seq_lens[original_chunk_0_idx] = new_seq_lens[original_chunk_0_idx] - prefix_len
            new_ids_each[original_chunk_0_idx] = new_ids_each[original_chunk_0_idx][prefix_len:]
        
        total_len = int(sum(new_seq_lens)) + prefix_len

        # Calculate offsets
        offsets = []
        cur = prefix_len
        for L_i in new_seq_lens:
            offsets.append(cur)
            cur += int(L_i)
        
        # Prepare RoPE embeddings for Qwen
        model_dtype = self.model.dtype
        position_ids_full = torch.arange(total_len, device=device).unsqueeze(0)
        cos_full, sin_full = self.model.model.rotary_emb(
            x=torch.empty(1, 1, 1, 1, device=device, dtype=model_dtype),
            position_ids=position_ids_full,
        )
        
        # Allocate final cache 
        H_kv = self.num_kv_heads
        D = self.head_dim
        
        # Get dtype from original cache
        if isinstance(cache, DynamicCache):
            cache_dtype = cache.key_cache[0].dtype
        else:
            cache_dtype = cache[0][0].dtype
        
        V_final = [torch.empty((1, H_kv, total_len, D), device=device, dtype=cache_dtype) for _ in range(num_layers)]
        K_final = [torch.empty((1, H_kv, total_len, D), device=device, dtype=cache_dtype) for _ in range(num_layers)]
        
        max_chunk_len = max(new_seq_lens) if new_seq_lens else 0
        base_pos = torch.arange(max_chunk_len + prefix_len + 1, device=device)
        
        # Process each layer
        for layer_idx in range(num_layers):
            K_buf = K_final[layer_idx]
            
            for chunk_idx, orig_chunk_id in enumerate(order):
                T_i = int(new_seq_lens[chunk_idx])
                off = int(offsets[chunk_idx])
                if T_i == 0:
                    continue
                
                # Get chunk view 
                chunk_start = chunk_starts[orig_chunk_id]
                chunk_end = chunk_start + seq_lens[orig_chunk_id]
                
                if isinstance(cache, DynamicCache):
                    K_chunk = cache.key_cache[layer_idx][:, :, chunk_start:chunk_end, :]
                    V_chunk = cache.value_cache[layer_idx][:, :, chunk_start:chunk_end, :]
                else:
                    K_full, V_full = cache[layer_idx]
                    K_chunk = K_full[:, :, chunk_start:chunk_end, :]
                    V_chunk = V_full[:, :, chunk_start:chunk_end, :]
                
                # old_pos should be based on original chunk length, not reordered length
                # All chunks: positions start from prefix_len because extraction adds prefix before forward
                orig_chunk_len = int(seq_lens[orig_chunk_id])
                if orig_chunk_id == 0:
                    # Chunk 0: K_chunk has prefix, we remove it, so old_pos is [prefix_len, orig_chunk_len)
                    old_pos = base_pos[prefix_len:orig_chunk_len]
                else:
                    # Other chunks: K_chunk already has prefix removed, but was computed with prefix
                    # so positions are [prefix_len, orig_chunk_len+prefix_len)
                    old_pos = base_pos[prefix_len:orig_chunk_len + prefix_len]

                if orig_chunk_id == 0:
                    # Copy prefix 
                    K_final[layer_idx][:, :, :prefix_len, :].copy_(K_chunk[:, :, :prefix_len, :])
                    V_final[layer_idx][:, :, :prefix_len, :].copy_(V_chunk[:, :, :prefix_len, :])
                    
                    # Remove RoPE from K (Qwen style)
                    K_removed = self._remove_rope(
                        K_chunk[:, :, prefix_len:, :], cos_full, sin_full, position_ids=old_pos
                    )
                    K_buf[:, :, off:off + T_i, :].copy_(K_removed)
                    V_final[layer_idx][:, :, off:off + T_i, :].copy_(V_chunk[:, :, prefix_len:, :])
                else:
                    # Remove RoPE (Qwen style)
                    K_removed = self._remove_rope(
                        K_chunk, cos_full, sin_full, position_ids=old_pos
                    )
                    K_buf[:, :, off:off + T_i, :].copy_(K_removed)
                    V_final[layer_idx][:, :, off:off + T_i, :].copy_(V_chunk)
            
            # Apply new RoPE (Qwen style)
            new_pos = torch.arange(prefix_len, total_len, device=device)
            K_rebased = self._apply_rope(
                K_buf[:, :, prefix_len:, :], cos_full, sin_full, position_ids=new_pos
            )

            K_final[layer_idx][:, :, prefix_len:, :].copy_(K_rebased)
        
        # Build final cache
        dyn = DynamicCache()
        dyn.key_cache = [K.contiguous() for K in K_final]
        dyn.value_cache = [V.contiguous() for V in V_final]
        
        # Rebuild input_ids
        ids_cat = list(original_prefix_ids) + list(chain.from_iterable(new_ids_each))
        new_input_ids = torch.tensor([ids_cat], dtype=torch.long, device=device)
        
        # Adjust seq_lens to include prefix in first chunk
        new_seq_lens[0] += prefix_len
        
        # Build attention mask
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        
        end_time = time.perf_counter()
        # print(f"Reorder and rebase took {(end_time - start_time) * 1000:.2f}ms")
        
        return KVCacheData(
            past_key_values=dyn,
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            chunk_lens=new_seq_lens,
        )


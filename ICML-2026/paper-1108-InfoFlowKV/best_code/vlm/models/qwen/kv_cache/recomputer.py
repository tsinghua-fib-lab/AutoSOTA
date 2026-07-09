"""KV cache recomputation for Qwen3-VL with multiple attention backends."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import DynamicCache

from .base import KVCacheData

# Optional FlashInfer import
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


class KVCacheRecomputer:
    """
    Recompute KV cache entries at selected positions.

    Handles:
    - MRoPE with 3D position_ids
    - GQA (grouped query attention)
    - Layer-by-layer recomputation with proper hidden state propagation
    - Multiple attention backends (SDPA, FlashInfer, math)
    - Chunked attention for memory efficiency

    Strategy:
    - Layer 0: Recompute full KV (need correct hidden states for subsequent layers)
    - Layer 1+: Only recompute at selected indices

    Usage:
        recomputer = KVCacheRecomputer(model, recompute_attention_mode="sdpa")
        updated_kv_data = recomputer.recompute(kv_data, recompute_indices)
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        model_type: str = "qwen",
        num_recompute_chunks: int = 2,
        recompute_attention_mode: str = "sdpa",
    ):
        """
        Args:
            model: The Qwen3-VL model
            tokenizer: Optional tokenizer (for cache-dev compatibility)
            model_type: Model type identifier
            num_recompute_chunks: Number of chunks for chunked attention
            recompute_attention_mode: Attention backend ("sdpa", "flashinfer", "math")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_recompute_chunks = num_recompute_chunks
        self.recompute_attention_mode = recompute_attention_mode
        self.language_model = model.model.language_model

        # Validate attention mode
        if recompute_attention_mode == "flashinfer" and not FLASHINFER_AVAILABLE:
            raise ImportError(
                "FlashInfer is not available. Install with: pip install flashinfer"
            )

        # Get model config
        config = self.language_model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.hidden_size = config.hidden_size

    @torch.no_grad()
    def recompute(
        self,
        kv_data: KVCacheData,
        recompute_indices: torch.Tensor,
        return_kv_data: bool = True,
    ) -> Union[KVCacheData, DynamicCache]:
        """
        Recompute KV cache at specified indices.

        Args:
            kv_data: Extracted KV cache data with input_embeds and position info
            recompute_indices: [K] tensor of positions to recompute
            return_kv_data: If True, return KVCacheData; else return DynamicCache

        Returns:
            Updated KVCacheData or DynamicCache with recomputed entries
        """
        if recompute_indices.numel() == 0:
            if return_kv_data:
                return kv_data
            return kv_data.past_key_values

        device = kv_data.device
        dtype = kv_data.dtype
        batch_size = kv_data.input_embeds.shape[0]
        seq_len = kv_data.seq_len

        # Ensure indices are on correct device and sorted
        recompute_indices = recompute_indices.to(device).long()
        recompute_indices = torch.unique(recompute_indices, sorted=True)
        K = recompute_indices.numel()

        # Get position embeddings
        cos_full, sin_full = kv_data.position_embeddings

        # Start with full input embeddings
        hidden_states = kv_data.input_embeds.to(dtype=dtype)  # [B, T, H]

        # Get the cache
        cache = kv_data.past_key_values

        # Prepare chunks if using chunked attention
        chunks = None
        if self.num_recompute_chunks > 1:
            chunks = self._prepare_chunks(recompute_indices, seq_len)

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]
            # DynamicCache uses self.layers, each layer has .keys and .values
            cache_layer = cache.layers[layer_idx]
            k_cache = cache_layer.keys  # [B, num_kv_heads, T, head_dim]
            v_cache = cache_layer.values

            if layer_idx == 0:
                # Layer 0: Recompute full KV (need all hidden states for subsequent layers)
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
            cache_layer.keys = k_cache
            cache_layer.values = v_cache

        if return_kv_data:
            # Return updated KVCacheData
            return KVCacheData(
                past_key_values=cache,
                position_ids=kv_data.position_ids,
                position_embeddings=kv_data.position_embeddings,
                input_embeds=kv_data.input_embeds,
                input_ids=kv_data.input_ids,
                seq_len=kv_data.seq_len,
                image_ranges=kv_data.image_ranges,
            )
        return cache

    def _prepare_chunks(
        self,
        recompute_indices: torch.Tensor,
        seq_len: int,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        """
        Pre-compute chunk metadata for chunked attention.

        Args:
            recompute_indices: [K] positions to recompute
            seq_len: Total sequence length

        Returns:
            List of (chunk_start, chunk_end, chunk_indices) tuples
        """
        K = recompute_indices.numel()
        chunk_size = (K + self.num_recompute_chunks - 1) // self.num_recompute_chunks
        chunks = []

        for i in range(self.num_recompute_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, K)
            if start_idx >= K:
                break

            chunk_indices = recompute_indices[start_idx:end_idx]
            # For causal attention, chunk can attend to all keys up to max position in chunk
            chunk_end = int(chunk_indices.max().item()) + 1

            chunks.append((start_idx, end_idx, chunk_indices))

        return chunks

    @torch.no_grad()
    def recompute_noop(
        self,
        kv_data: KVCacheData,
        return_kv_data: bool = True,
    ) -> Union[KVCacheData, DynamicCache]:
        """No recomputation - return cache as-is."""
        if return_kv_data:
            return kv_data
        return kv_data.past_key_values

    @torch.no_grad()
    def recompute_lego(
        self,
        kv_data: KVCacheData,
        ratio: float = 0.15,
        return_kv_data: bool = True,
    ) -> Union[KVCacheData, DynamicCache]:
        """
        LEGO: Select first ratio% tokens from the full sequence.
        Simple baseline that always picks the earliest positions.
        """
        seq_len = kv_data.seq_len
        k = max(1, int(seq_len * ratio))
        recompute_indices = torch.arange(k, device=kv_data.device)
        return self.recompute(kv_data, recompute_indices, return_kv_data)

    @torch.no_grad()
    def recompute_cacheblend(
        self,
        kv_data: KVCacheData,
        recompute_ratio: float = 0.15,
        return_kv_data: bool = True,
    ) -> Union[KVCacheData, DynamicCache]:
        """
        CacheBlend layer-wise strategy:
        - Layer 0: Full KV recompute, full sequence attention
        - Layer 1: Full KV recompute, select top positions by V diff
        - Layer 2+: Selective recompute at positions from Layer 1
        """
        device = kv_data.device
        dtype = kv_data.dtype
        batch_size = kv_data.input_embeds.shape[0]
        seq_len = kv_data.seq_len

        cos_full, sin_full = kv_data.position_embeddings
        hidden_states = kv_data.input_embeds.to(dtype=dtype)
        cache = kv_data.past_key_values

        recompute_indices = None

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]
            cache_layer = cache.layers[layer_idx]
            k_cache = cache_layer.keys
            v_cache = cache_layer.values

            if layer_idx == 0:
                # Layer 0: Full recompute, full attention
                hidden_states, k_cache, v_cache = self._recompute_layer_cacheblend_full(
                    layer, hidden_states, k_cache, v_cache, cos_full, sin_full,
                    batch_size, seq_len
                )
            elif layer_idx == 1:
                # Layer 1: Full recompute + select top positions by V diff
                hidden_states, k_cache, v_cache, recompute_indices = self._recompute_layer_cacheblend_select(
                    layer, hidden_states, k_cache, v_cache, cos_full, sin_full,
                    batch_size, seq_len, recompute_ratio
                )
            else:
                # Layer 2+: Selective recompute at positions from Layer 1
                if recompute_indices is not None and recompute_indices.numel() > 0:
                    hidden_states, k_cache, v_cache = self._recompute_layer_sparse(
                        layer, hidden_states, k_cache, v_cache, cos_full, sin_full,
                        recompute_indices, batch_size, recompute_indices.numel(), None
                    )

            cache_layer.keys = k_cache
            cache_layer.values = v_cache

        if return_kv_data:
            return KVCacheData(
                past_key_values=cache,
                position_ids=kv_data.position_ids,
                position_embeddings=kv_data.position_embeddings,
                input_embeds=kv_data.input_embeds,
                input_ids=kv_data.input_ids,
                seq_len=kv_data.seq_len,
                image_ranges=kv_data.image_ranges,
            )
        return cache

    def _recompute_layer_cacheblend_full(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CacheBlend Layer 0: Full recompute with full sequence attention."""
        dtype = hidden_states.dtype
        attn = layer.self_attn

        normed = layer.input_layernorm(hidden_states)

        q = attn.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = attn.k_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = attn.k_norm(k)

        q = self._apply_rope(q, cos_full, sin_full)
        k = self._apply_rope(k, cos_full, sin_full)

        k_cache = k.to(dtype)
        v_cache = v.to(dtype)

        # Compute attention using configured backend
        attn_output = self._compute_attention(q, k_cache, v_cache, query_positions=None)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = attn.o_proj(attn_output).to(dtype)

        hidden_states = hidden_states + attn_output

        residual = hidden_states
        mlp_out = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states = residual + mlp_out

        return hidden_states, k_cache, v_cache

    def _recompute_layer_cacheblend_select(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        batch_size: int,
        seq_len: int,
        recompute_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CacheBlend Layer 1: Full recompute + select top positions by V diff."""
        dtype = hidden_states.dtype
        attn = layer.self_attn

        normed = layer.input_layernorm(hidden_states)

        q = attn.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = attn.k_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = attn.k_norm(k)

        q = self._apply_rope(q, cos_full, sin_full)
        k = self._apply_rope(k, cos_full, sin_full)

        # Compute V diff to select important positions
        v_diff = (v - v_cache).abs().mean(dim=(1, 3))  # [B, T]
        num_recompute = max(1, int(seq_len * recompute_ratio))
        _, top_indices = torch.topk(v_diff[0], num_recompute)
        recompute_indices = top_indices.sort().values

        k_cache = k.to(dtype)
        v_cache = v.to(dtype)

        # Attention only at selected positions
        q_subset = q[:, :, recompute_indices, :]
        attn_output = self._compute_attention(q_subset, k_cache, v_cache, recompute_indices)

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, recompute_indices.numel(), self.hidden_size
        )
        attn_output = attn.o_proj(attn_output).to(dtype)

        hidden_states_subset = hidden_states[:, recompute_indices, :]
        hidden_states_subset = hidden_states_subset + attn_output

        residual = hidden_states_subset
        mlp_out = layer.post_attention_layernorm(hidden_states_subset)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states_subset = residual + mlp_out

        return hidden_states_subset, k_cache, v_cache, recompute_indices

    def _recompute_layer_full(
        self,
        layer,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        recompute_indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
        chunks: Optional[List[Tuple[int, int, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute layer 0 with full sequence.

        Returns:
            (hidden_states_subset, updated_k_cache, updated_v_cache)
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

        # Apply Q/K norms
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

        # Subset hidden states to recompute indices for next layers
        hidden_states_subset = hidden_states[:, recompute_indices, :]
        q_subset = q[:, :, recompute_indices, :]

        # Compute attention for subset positions
        attn_output = self._compute_attention(
            q_subset, k_cache, v_cache, recompute_indices, chunks
        )

        # Project output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, recompute_indices.numel(), self.hidden_size
        )
        attn_output = attn.o_proj(attn_output).to(dtype)

        # Residual connection
        hidden_states_subset = hidden_states_subset + attn_output

        # MLP
        residual = hidden_states_subset
        mlp_out = layer.post_attention_layernorm(hidden_states_subset)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states_subset = residual + mlp_out

        return hidden_states_subset, k_cache, v_cache

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
        chunks: Optional[List[Tuple[int, int, torch.Tensor]]] = None,
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
        q = self._apply_rope_at_positions(q, cos_full, sin_full, recompute_indices)
        k_new = self._apply_rope_at_positions(k_new, cos_full, sin_full, recompute_indices)

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

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embeddings to all positions."""
        # x: [B, H, T, D]
        # cos, sin: [B, T, D] or similar
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)  # [B, 1, T, D]
            sin = sin.unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _apply_rope_at_positions(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embeddings at specific positions."""
        # x: [B, H, K, D]
        # cos, sin: [B, T, D]
        # positions: [K]
        if cos.dim() == 3:
            cos = cos[:, positions, :].unsqueeze(1)  # [B, 1, K, D]
            sin = sin[:, positions, :].unsqueeze(1)
        else:
            cos = cos[:, :, positions, :]
            sin = sin[:, :, positions, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: Optional[torch.Tensor],
        chunks: Optional[List[Tuple[int, int, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Compute attention with causal mask using configured backend.

        Routes to appropriate backend based on self.recompute_attention_mode.

        Args:
            q: [B, H_q, K, D] query at K positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries (for causal mask), None for full causal
            chunks: Optional pre-computed chunk metadata

        Returns:
            [B, H_q, K, D] attention output
        """
        if query_positions is None:
            # Full causal attention
            return self._compute_attention_full_causal(q, k, v)

        mode = self.recompute_attention_mode

        if mode == "flashinfer":
            if chunks and len(chunks) > 1:
                return self._compute_attention_chunked_flashinfer(q, k, v, query_positions, chunks)
            return self._compute_attention_flashinfer(q, k, v, query_positions)
        elif mode in ("sdpa", "math"):
            if chunks and len(chunks) > 1:
                return self._compute_attention_chunked_sdpa(q, k, v, query_positions, chunks)
            return self._compute_attention_sdpa(q, k, v, query_positions)
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

    def _compute_attention_full_causal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full causal attention.

        Args:
            q: [B, H_q, T, D] query tensor
            k: [B, H_kv, T, D] key tensor
            v: [B, H_kv, T, D] value tensor

        Returns:
            [B, H_q, T, D] attention output
        """
        T = q.size(2)
        batch_size = q.size(0)

        # GQA expansion
        num_groups = self.num_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(num_groups, dim=1)
        v_expanded = v.repeat_interleave(num_groups, dim=1)

        if self.recompute_attention_mode == "math":
            # Manual attention with causal mask
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.matmul(attn_weights, v_expanded)
        else:
            # SDPA with is_causal=True
            return F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                is_causal=True,
            )

    def _compute_attention_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention with causal mask using SDPA.

        Uses the same attention backend as extraction (ChunkPrefiller) to ensure
        numerical consistency. This is critical for chunk_k=0 where recomputation
        should not change the cache values.

        Args:
            q: [B, H_q, K, D] query at K positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries (for causal mask)

        Returns:
            [B, H_q, K, D] attention output
        """
        T = k.size(2)
        K = q.size(2)
        device = q.device
        batch_size = q.size(0)

        # GQA expansion: expand K, V to match Q heads
        num_groups = self.num_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]
        v_expanded = v.repeat_interleave(num_groups, dim=1)  # [B, H_q, T, D]

        # Build causal mask [K, T]: query can attend to keys where key_pos <= query_pos
        key_positions = torch.arange(T, device=device)
        causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)  # [K, T]
        # Expand to [B, 1, K, T] for SDPA
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Use SDPA for consistency with extraction
        output = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=attn_mask,
            is_causal=False,
        )

        return output

    def _compute_attention_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention using FlashInfer with causal masking at sparse positions.

        Processes each query position separately with FlashInfer's single_prefill_with_kv_cache,
        using truncated KV cache for each position to achieve causal masking.

        Args:
            q: [B, H_q, K, D] query at K positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries (for causal mask)

        Returns:
            [B, H_q, K, D] attention output
        """
        if not FLASHINFER_AVAILABLE:
            raise ImportError("FlashInfer is not available. Install with: pip install flashinfer")

        B, H_q, K, D = q.shape
        _, H_kv, T, _ = k.shape
        device = q.device
        dtype = q.dtype

        # Allocate output
        output = torch.zeros(B, H_q, K, D, device=device, dtype=dtype)

        for b in range(B):
            # Extract single batch and transpose to FlashInfer format
            k_b = k[b].transpose(0, 1).contiguous()  # [T, H_kv, D]
            v_b = v[b].transpose(0, 1).contiguous()  # [T, H_kv, D]

            # Process each query position with its causal KV prefix
            for i, pos in enumerate(query_positions):
                pos_int = int(pos.item())
                kv_len = pos_int + 1  # Include position itself (causal)

                # Single query: [1, H_q, D]
                q_i = q[b, :, i:i+1, :].transpose(0, 1).contiguous()  # [1, H_q, D]
                k_prefix = k_b[:kv_len].contiguous()  # [kv_len, H_kv, D]
                v_prefix = v_b[:kv_len].contiguous()  # [kv_len, H_kv, D]

                # Compute attention with FlashInfer
                out_i = flashinfer.single_prefill_with_kv_cache(
                    q_i,
                    k_prefix,
                    v_prefix,
                    causal=False,  # Full attention to truncated KV
                    sm_scale=1.0 / (D ** 0.5),
                )  # [1, H_q, D]

                # out_i: [1, H_q, D] -> squeeze to [H_q, D]
                output[b, :, i, :] = out_i.squeeze(0)

        return output

    def _compute_attention_chunked_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
        chunks: List[Tuple[int, int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute attention in chunks using SDPA.

        Args:
            q: [B, H_q, K, D] query at K positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries
            chunks: Pre-computed chunk metadata

        Returns:
            [B, H_q, K, D] attention output
        """
        B, H_q, K, D = q.shape
        device = q.device
        dtype = q.dtype

        # GQA expansion
        num_groups = self.num_heads // self.num_kv_heads
        k_expanded = k.repeat_interleave(num_groups, dim=1)
        v_expanded = v.repeat_interleave(num_groups, dim=1)

        # Allocate output
        output = torch.zeros(B, H_q, K, D, device=device, dtype=dtype)

        for start_idx, end_idx, chunk_indices in chunks:
            chunk_size = end_idx - start_idx
            chunk_positions = query_positions[start_idx:end_idx]

            # Get queries for this chunk
            q_chunk = q[:, :, start_idx:end_idx, :]  # [B, H_q, chunk_size, D]

            # Build causal mask for this chunk
            T = k.size(2)
            key_positions = torch.arange(T, device=device)
            causal_mask = chunk_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

            # Compute attention for this chunk
            chunk_output = F.scaled_dot_product_attention(
                q_chunk, k_expanded, v_expanded,
                attn_mask=attn_mask,
                is_causal=False,
            )

            output[:, :, start_idx:end_idx, :] = chunk_output

        return output

    def _compute_attention_chunked_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
        chunks: List[Tuple[int, int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute attention in chunks using FlashInfer.

        Processes each query position within chunks using truncated KV cache for causal masking.

        Args:
            q: [B, H_q, K, D] query at K positions
            k: [B, H_kv, T, D] full key cache
            v: [B, H_kv, T, D] full value cache
            query_positions: [K] positions of queries
            chunks: Pre-computed chunk metadata

        Returns:
            [B, H_q, K, D] attention output
        """
        if not FLASHINFER_AVAILABLE:
            raise ImportError("FlashInfer is not available. Install with: pip install flashinfer")

        B, H_q, K, D = q.shape
        _, H_kv, T, _ = k.shape
        device = q.device
        dtype = q.dtype

        # Allocate output
        output = torch.zeros(B, H_q, K, D, device=device, dtype=dtype)

        for b in range(B):
            # Extract single batch and transpose to FlashInfer format
            k_b = k[b].transpose(0, 1).contiguous()  # [T, H_kv, D]
            v_b = v[b].transpose(0, 1).contiguous()  # [T, H_kv, D]

            for start_idx, end_idx, chunk_indices in chunks:
                chunk_positions = query_positions[start_idx:end_idx]

                # Process each position within chunk
                for i, pos in enumerate(chunk_positions):
                    global_idx = start_idx + i
                    pos_int = int(pos.item())
                    kv_len = pos_int + 1

                    q_i = q[b, :, global_idx:global_idx+1, :].transpose(0, 1).contiguous()
                    k_prefix = k_b[:kv_len].contiguous()
                    v_prefix = v_b[:kv_len].contiguous()

                    out_i = flashinfer.single_prefill_with_kv_cache(
                        q_i,
                        k_prefix,
                        v_prefix,
                        causal=False,
                        sm_scale=1.0 / (D ** 0.5),
                    )
                    # out_i: [1, H_q, D] -> squeeze to [H_q, D]
                    output[b, :, global_idx, :] = out_i.squeeze(0)

        return output

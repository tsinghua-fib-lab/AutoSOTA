"""
Distributed recomputation using Ring Attention.

Ring attention enables efficient sequence parallelism by:
1. Keeping Q stationary on each GPU (only important positions)
2. Rotating KV blocks through the ring
3. Overlapping communication with computation
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List, Any
from transformers.cache_utils import DynamicCache

from .config import DistributedConfig
from .extractor import DistributedKVCacheData

# Try to import flash attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    import flashinfer
    from flashinfer import merge_states
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


class RingAttentionRecomputer:
    """
    Distributed recomputation using Ring Attention.

    Key insight:
    - Q is SPARSE (only important positions on this GPU)
    - KV ROTATES through ring (each GPU sees all KV eventually)
    - Communication overlaps with computation

    For single GPU, falls back to standard flash attention or SDPA.

    Example:
        recomputer = RingAttentionRecomputer(model, config)
        updated_kv = recomputer.recompute_distributed(
            local_kv, local_important_positions, global_important_positions
        )
    """

    def __init__(
        self,
        model,
        config: DistributedConfig,
        use_ring_attention: bool = True,
    ):
        """
        Args:
            model: The language model
            config: DistributedConfig with process info
            use_ring_attention: Whether to use ring attention for multi-GPU
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.use_ring_attention = use_ring_attention

        # Get model config
        model_config = model.config
        self.num_layers = getattr(
            model_config, "num_layers",
            getattr(model_config, "num_hidden_layers", 28)
        )
        self.num_heads = getattr(model_config, "num_attention_heads", 32)
        self.num_kv_heads = getattr(
            model_config, "multi_query_group_num",
            getattr(model_config, "num_key_value_heads", self.num_heads)
        )
        self.hidden_size = getattr(model_config, "hidden_size", 4096)
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_head_dim = getattr(model_config, "kv_channels", self.head_dim)

        # Pre-allocate FlashInfer workspace for batch prefill with paged KV
        if FLASHINFER_AVAILABLE:
            self._fi_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
        else:
            self._fi_workspace = None

        # Get model components
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
            self.embed_layer = model.get_input_embeddings()
            self.rotary_emb = model.model.rotary_emb
        else:
            raise ValueError("Cannot find model layers")

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
        """Apply RoPE to tensor x."""
        if position_ids is not None:
            # Ensure position_ids is 1D
            if position_ids.dim() > 1:
                position_ids = position_ids.squeeze()
            if position_ids.dim() == 0:
                position_ids = position_ids.unsqueeze(0)

            # Use index_select for 1D positions
            cos = cos.index_select(1, position_ids)
            sin = sin.index_select(1, position_ids)

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    def _compute_q_at_positions(
        self,
        layer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q vectors at specified positions.

        Args:
            layer: Transformer layer
            hidden_states: Hidden states [B, K, hidden_size] at positions
            positions: Global position indices [K]
            cos, sin: RoPE embeddings

        Returns:
            Q tensor [B, num_heads, K, head_dim] with RoPE applied
        """
        B, K, H = hidden_states.shape
        attn = layer.self_attn

        # Layer norm + Q projection
        normed = layer.input_layernorm(hidden_states)
        q = attn.q_proj(normed).view(B, K, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, num_heads, K, head_dim]

        # Apply Q norm if present
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)

        # Apply RoPE at correct positions
        q = self._apply_rope(q, cos, sin, positions)

        return q

    def _compute_kv_at_positions(
        self,
        layer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute K, V vectors at specified positions.

        Args:
            layer: Transformer layer
            hidden_states: Hidden states [B, K, hidden_size]
            positions: Global position indices [K]
            cos, sin: RoPE embeddings

        Returns:
            K, V tensors [B, num_kv_heads, K, head_dim]
        """
        B, K, H = hidden_states.shape
        attn = layer.self_attn

        # Layer norm + K, V projection
        normed = layer.input_layernorm(hidden_states)
        k = attn.k_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)
        v = attn.v_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)

        k = k.transpose(1, 2)  # [B, num_kv_heads, K, head_dim]
        v = v.transpose(1, 2)

        # Apply K norm if present
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = attn.k_norm(k)

        # Apply RoPE to K
        k = self._apply_rope(k, cos, sin, positions)

        return k, v

    @torch.no_grad()
    def recompute_distributed(
        self,
        local_kv: DistributedKVCacheData,
        local_important_positions: torch.Tensor,
        global_important_positions: torch.Tensor,
    ) -> DistributedKVCacheData:
        """
        Recompute attention at important positions using ring attention.

        Strategy (matching non-SP implementation):
        - Layer 0: Full KV recomputation at ALL positions (need correct hidden states)
        - Layer 1+: Sparse recomputation at important positions only

        Args:
            local_kv: Local KV cache partition
            local_important_positions: Important positions within this GPU's partition (local indices)
            global_important_positions: All important positions in global indices

        Returns:
            Updated DistributedKVCacheData
        """
        device = self.device
        dtype = self.model.dtype
        multi_gpu = self.config.enabled and self.config.world_size > 1 and self.use_ring_attention

        # Only return early for single-GPU. In multi-GPU mode, even ranks with
        # K_local=0 must participate in collectives (they provide KV for other
        # ranks' queries via cascade merge).
        if len(local_important_positions) == 0 and not multi_gpu:
            return local_kv

        # Convert positions to tensors if needed
        if isinstance(local_important_positions, (list, tuple)):
            local_important_positions = torch.tensor(
                local_important_positions, device=device, dtype=torch.long
            )
        else:
            local_important_positions = local_important_positions.to(device)

        # Ensure local_important_positions is 1D
        if local_important_positions.dim() > 1:
            local_important_positions = local_important_positions.squeeze()
        if local_important_positions.dim() == 0:
            local_important_positions = local_important_positions.unsqueeze(0)

        # Get global positions for RoPE
        global_offset = local_kv.global_offset
        global_important_global = local_important_positions + global_offset

        # Get input embeddings for ALL local positions (not just important ones)
        # This is critical for Layer 0 full recompute
        local_ids = local_kv.input_ids
        if local_ids.dim() == 1:
            local_ids = local_ids.unsqueeze(0)

        local_T = local_kv.local_seq_len if local_kv.local_seq_len > 0 else local_ids.shape[1]
        total_len = local_kv.global_total_len if local_kv.global_total_len > 0 else local_kv.total_len

        # Compute embeddings for ALL local positions
        hidden_states_full = self.embed_layer(local_ids).to(dtype)  # [B, local_T, H]

        # Ensure hidden_states is 3D [B, T, H]
        if hidden_states_full.dim() == 2:
            hidden_states_full = hidden_states_full.unsqueeze(0)

        B = hidden_states_full.shape[0]
        K = local_important_positions.numel()

        # Get RoPE embeddings for full sequence range
        max_pos = total_len + 1
        position_ids = torch.arange(max_pos, device=device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states_full, position_ids)

        # Local positions (0 to local_T-1) map to global positions (global_offset to global_offset+local_T-1)
        local_positions_tensor = torch.arange(local_T, device=device)
        global_positions_full = local_positions_tensor + global_offset

        # Get the cache
        cache = local_kv.past_key_values

        # Cascade local attention: each GPU computes attention for ALL queries
        # against its LOCAL KV shard, then merges via output+lse all-gather.
        # Trades per-layer Q gather + output gather for elimination of KV gather.
        # K=0 ranks still participate (they provide local KV for other ranks' queries).
        use_cascade = multi_gpu and FLASHINFER_AVAILABLE
        cascade_ctx = None
        if use_cascade:
            cascade_ctx = self._init_cascade_ctx(
                K, local_T, global_offset, total_len,
                global_important_positions, local_kv,
            )

        # Fallback: per-layer KV all-gather (when cascade is not available)
        gather_ctx = None
        gathered_kv_layers = []  # Save each layer's gathered result for generation
        if multi_gpu and not use_cascade:
            gather_ctx = self._init_gather_buffers(local_kv)

        # Use FlashInfer single_prefill for attention (no batch wrapper needed)
        use_flashinfer = multi_gpu and FLASHINFER_AVAILABLE and K > 0

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            attn = layer.self_attn

            # Get local K, V from cache
            if isinstance(cache, DynamicCache):
                k_local = cache.key_cache[layer_idx]
                v_local = cache.value_cache[layer_idx]
            else:
                k_local, v_local = cache[layer_idx]

            if layer_idx == 0:
                # Layer 0: Full KV recomputation at ALL local positions
                normed_full = layer.input_layernorm(hidden_states_full)

                # Compute K, V for ALL local positions
                k = attn.k_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                v = attn.v_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k = attn.k_norm(k)

                k = self._apply_rope(k, cos, sin, global_positions_full)

                k_local = k.to(dtype)
                v_local = v.to(dtype)

                # Gather this layer's KV from all ranks (fallback path only)
                if gather_ctx is not None:
                    k_full, v_full = self._gather_layer_kv(k_local, v_local, gather_ctx)
                    gathered_kv_layers.append((k_full, v_full))

                # Compute Q for ALL positions, then extract sparse
                q_full = attn.q_proj(normed_full).view(B, local_T, self.num_heads, self.head_dim)
                q_full = q_full.transpose(1, 2)

                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q_full = attn.q_norm(q_full)

                q_full = self._apply_rope(q_full, cos, sin, global_positions_full)
                q_sparse = q_full[:, :, local_important_positions, :]

                # Get hidden states at important positions for residual
                hidden_states_important = hidden_states_full[:, local_important_positions, :]

                # Compute attention: cascade (local + merge) or fallback (gather + full)
                if cascade_ctx is not None:
                    attn_output = self._cascade_attention(
                        q_sparse, k_local, v_local, cascade_ctx,
                    )
                elif gather_ctx is not None:
                    attn_output = self._attention_with_full_kv(
                        q_sparse, gathered_kv_layers, 0, global_important_global,
                        use_flashinfer=use_flashinfer,
                    )
                else:
                    attn_output = self._local_attention(
                        q_sparse, k_local, v_local, global_important_global,
                        key_offset=global_offset,
                    )

                # Project output
                attn_output = attn_output.transpose(1, 2).reshape(B, K, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)

                # Residual + MLP at important positions
                hidden_states_at_important = hidden_states_important + attn_output
                residual = hidden_states_at_important
                mlp_input = layer.post_attention_layernorm(hidden_states_at_important)
                mlp_output = layer.mlp(mlp_input).to(dtype)
                hidden_states_at_important = residual + mlp_output

                # After layer 0, keep only important positions [B, K, H]
                hidden_states_sparse = hidden_states_at_important

                # Free full hidden states (no longer needed)
                del hidden_states_full

            else:
                # Layer 1+: Sparse recomputation at important positions only
                # Fused: compute layernorm ONCE (shared by KV and Q projections)
                normed = layer.input_layernorm(hidden_states_sparse)

                # K, V projection + RoPE
                k_new = attn.k_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)
                v_new = attn.v_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)
                k_new = k_new.transpose(1, 2)
                v_new = v_new.transpose(1, 2)
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k_new = attn.k_norm(k_new)
                k_new = self._apply_rope(k_new, cos, sin, global_important_global)

                # Update local cache at important positions (in-place)
                k_local.index_copy_(2, local_important_positions, k_new.to(dtype))
                v_local.index_copy_(2, local_important_positions, v_new.to(dtype))

                # Start async all-gather (fallback path only, overlaps with Q computation)
                async_handles = None
                async_gather_lists = None
                if gather_ctx is not None:
                    async_handles, async_gather_lists = self._async_gather_layer_kv(
                        k_local, v_local, gather_ctx
                    )

                # Q projection + RoPE (overlapped with async gather on NVLink)
                q_sparse = attn.q_proj(normed).view(B, K, self.num_heads, self.head_dim)
                q_sparse = q_sparse.transpose(1, 2)
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q_sparse = attn.q_norm(q_sparse)
                q_sparse = self._apply_rope(q_sparse, cos, sin, global_important_global)

                # Compute attention: cascade (local + merge) or fallback (gather + full)
                if cascade_ctx is not None:
                    attn_output = self._cascade_attention(
                        q_sparse, k_local, v_local, cascade_ctx,
                    )
                elif gather_ctx is not None:
                    k_full, v_full = self._finish_async_gather(
                        async_handles, async_gather_lists, gather_ctx
                    )
                    gathered_kv_layers.append((k_full, v_full))
                    attn_output = self._attention_with_full_kv(
                        q_sparse, gathered_kv_layers, layer_idx, global_important_global,
                        use_flashinfer=use_flashinfer,
                    )
                else:
                    attn_output = self._local_attention(
                        q_sparse, k_local, v_local, global_important_global,
                        key_offset=global_offset,
                    )

                # Project output
                attn_output = attn_output.transpose(1, 2).reshape(B, K, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)

                # Residual + MLP
                hidden_states_sparse = hidden_states_sparse + attn_output
                residual = hidden_states_sparse
                mlp_input = layer.post_attention_layernorm(hidden_states_sparse)
                mlp_output = layer.mlp(mlp_input).to(dtype)
                hidden_states_sparse = residual + mlp_output

            # Update local cache in place
            if isinstance(cache, DynamicCache):
                cache.key_cache[layer_idx] = k_local
                cache.value_cache[layer_idx] = v_local
            else:
                cache[layer_idx] = (k_local, v_local)

        # Build generation cache: need full KV from all ranks for autoregressive decode.
        gathered_cache = None
        if cascade_ctx is not None:
            # Cascade path: one-time all-gather of all layers' final local KV
            gather_ctx_final = cascade_ctx["gather_ctx_final"]
            gathered_cache = DynamicCache()
            gathered_cache.key_cache = []
            gathered_cache.value_cache = []
            for layer_idx in range(self.num_layers):
                if isinstance(cache, DynamicCache):
                    k_l = cache.key_cache[layer_idx]
                    v_l = cache.value_cache[layer_idx]
                else:
                    k_l, v_l = cache[layer_idx]
                k_full, v_full = self._gather_layer_kv(k_l, v_l, gather_ctx_final)
                gathered_cache.key_cache.append(k_full)
                gathered_cache.value_cache.append(v_full)
        elif gathered_kv_layers:
            # Fallback path: reuse per-layer gathered results
            gathered_cache = DynamicCache()
            gathered_cache.key_cache = [kv[0] for kv in gathered_kv_layers]
            gathered_cache.value_cache = [kv[1] for kv in gathered_kv_layers]

        return DistributedKVCacheData(
            past_key_values=cache,
            input_ids=local_kv.input_ids,
            attention_mask=local_kv.attention_mask,
            chunk_lens=local_kv.chunk_lens,
            global_offset=local_kv.global_offset,
            global_total_len=local_kv.global_total_len,
            local_seq_len=local_kv.local_seq_len,
            gathered_full_kv=gathered_cache,
        )

    @torch.no_grad()
    def recompute_distributed_cacheblend(
        self,
        local_kv: DistributedKVCacheData,
        recompute_ratio: float = 0.15,
    ) -> DistributedKVCacheData:
        """
        CacheBlend distributed recomputation strategy.

        Matches single-GPU CacheBlend (models/qwen/kv_cache/recomputer.py:711-859):
        - Layer 0: Full KV recompute, full Q, local-only attention
        - Layer 1: Full KV recompute, V-diff selection, update full cache, sparse attention
        - Layer 2+: Sparse recompute at selected positions (same as guided recompute)

        Args:
            local_kv: Local KV cache partition from DistributedExtractor
            recompute_ratio: Fraction of positions to select at Layer 1

        Returns:
            Updated DistributedKVCacheData with gathered_full_kv for generation
        """
        device = self.device
        dtype = self.model.dtype
        multi_gpu = self.config.enabled and self.config.world_size > 1 and self.use_ring_attention

        # Get input embeddings for ALL local positions
        local_ids = local_kv.input_ids
        if local_ids.dim() == 1:
            local_ids = local_ids.unsqueeze(0)

        local_T = local_kv.local_seq_len if local_kv.local_seq_len > 0 else local_ids.shape[1]
        total_len = local_kv.global_total_len if local_kv.global_total_len > 0 else local_kv.total_len
        global_offset = local_kv.global_offset

        hidden_states_full = self.embed_layer(local_ids).to(dtype)  # [B, local_T, H]
        if hidden_states_full.dim() == 2:
            hidden_states_full = hidden_states_full.unsqueeze(0)
        B = hidden_states_full.shape[0]

        # RoPE for full sequence range
        max_pos = total_len + 1
        position_ids = torch.arange(max_pos, device=device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states_full, position_ids)

        local_positions_tensor = torch.arange(local_T, device=device)
        global_positions_full = local_positions_tensor + global_offset

        cache = local_kv.past_key_values

        # These are set at Layer 1 after V-diff selection
        local_important_positions = None
        global_important_positions = None
        global_important_global = None  # local important in global coords
        hidden_states_sparse = None
        K = 0

        # Attention contexts (initialized after Layer 1 position selection)
        cascade_ctx = None
        gather_ctx = None
        gathered_kv_layers = []

        use_flashinfer = False

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            attn = layer.self_attn

            if isinstance(cache, DynamicCache):
                k_local = cache.key_cache[layer_idx]
                v_local = cache.value_cache[layer_idx]
            else:
                k_local, v_local = cache[layer_idx]

            if layer_idx == 0:
                # LAYER 0: Full KV recompute, full Q, LOCAL-ONLY attention
                normed_full = layer.input_layernorm(hidden_states_full)

                k = attn.k_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                v = attn.v_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k = attn.k_norm(k)
                k = self._apply_rope(k, cos, sin, global_positions_full)

                k_local = k.to(dtype)
                v_local = v.to(dtype)

                q_full = attn.q_proj(normed_full).view(B, local_T, self.num_heads, self.head_dim)
                q_full = q_full.transpose(1, 2)
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q_full = attn.q_norm(q_full)
                q_full = self._apply_rope(q_full, cos, sin, global_positions_full)

                # Local-only attention (matches extraction: each GPU sees its own chunk)
                attn_output = self._local_attention(
                    q_full, k_local, v_local, global_positions_full,
                    key_offset=global_offset,
                )

                attn_output = attn_output.transpose(1, 2).reshape(B, local_T, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)

                hidden_states_full = hidden_states_full + attn_output
                residual = hidden_states_full
                mlp_input = layer.post_attention_layernorm(hidden_states_full)
                mlp_output = layer.mlp(mlp_input).to(dtype)
                hidden_states_full = residual + mlp_output

            elif layer_idx == 1:
                # LAYER 1: Full KV recompute + V-diff selection
                normed_full = layer.input_layernorm(hidden_states_full)

                k_new = attn.k_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                v_new = attn.v_proj(normed_full).view(B, local_T, self.num_kv_heads, self.kv_head_dim)
                k_new = k_new.transpose(1, 2)
                v_new = v_new.transpose(1, 2)
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k_new = attn.k_norm(k_new)
                k_new = self._apply_rope(k_new, cos, sin, global_positions_full)

                # V-diff selection (local to this GPU)
                dims_to_average = [i for i in range(v_new.dim()) if i != 2]
                diff_per_token = torch.mean((v_new - v_local) ** 2, dim=dims_to_average)  # [local_T]
                num_selected = max(1, int(local_T * recompute_ratio))
                _, top_indices = torch.topk(diff_per_token, num_selected)
                local_important_positions = torch.sort(top_indices).values

                # Update FULL cache for layer 1
                k_local = k_new.to(dtype)
                v_local = v_new.to(dtype)

                # All-gather positions across GPUs
                global_important_positions = allgather_positions(
                    local_important_positions, global_offset, self.config,
                )
                global_important_global = local_important_positions + global_offset
                K = local_important_positions.numel()

                # Initialize cascade/gather ctx for sparse attention (layers 1+)
                use_cascade = multi_gpu and FLASHINFER_AVAILABLE
                if use_cascade:
                    cascade_ctx = self._init_cascade_ctx(
                        K, local_T, global_offset, total_len,
                        global_important_positions, local_kv,
                    )
                elif multi_gpu:
                    gather_ctx = self._init_gather_buffers(local_kv)

                use_flashinfer = multi_gpu and FLASHINFER_AVAILABLE and K > 0

                # Gather KV for fallback path
                if gather_ctx is not None:
                    k_full, v_full = self._gather_layer_kv(k_local, v_local, gather_ctx)
                    gathered_kv_layers.append((k_full, v_full))

                # Narrow hidden states to selected positions
                hidden_states_sparse = hidden_states_full[:, local_important_positions, :]
                normed_sparse = normed_full[:, local_important_positions, :]

                # Q at selected positions
                q_sparse = attn.q_proj(normed_sparse).view(B, K, self.num_heads, self.head_dim)
                q_sparse = q_sparse.transpose(1, 2)
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q_sparse = attn.q_norm(q_sparse)
                q_sparse = self._apply_rope(q_sparse, cos, sin, global_important_global)

                # Attention
                if cascade_ctx is not None:
                    attn_output = self._cascade_attention(
                        q_sparse, k_local, v_local, cascade_ctx,
                    )
                elif gather_ctx is not None:
                    attn_output = self._attention_with_full_kv(
                        q_sparse, gathered_kv_layers, 0, global_important_global,
                        use_flashinfer=use_flashinfer,
                    )
                else:
                    attn_output = self._local_attention(
                        q_sparse, k_local, v_local, global_important_global,
                        key_offset=global_offset,
                    )

                attn_output = attn_output.transpose(1, 2).reshape(B, K, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)

                hidden_states_sparse = hidden_states_sparse + attn_output
                residual = hidden_states_sparse
                mlp_input = layer.post_attention_layernorm(hidden_states_sparse)
                mlp_output = layer.mlp(mlp_input).to(dtype)
                hidden_states_sparse = residual + mlp_output

                del hidden_states_full  # Free memory

            else:
                # LAYER 2+: Sparse recompute (identical to recompute_distributed Layer 1+)
                normed = layer.input_layernorm(hidden_states_sparse)

                k_new = attn.k_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)
                v_new = attn.v_proj(normed).view(B, K, self.num_kv_heads, self.kv_head_dim)
                k_new = k_new.transpose(1, 2)
                v_new = v_new.transpose(1, 2)
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k_new = attn.k_norm(k_new)
                k_new = self._apply_rope(k_new, cos, sin, global_important_global)

                k_local.index_copy_(2, local_important_positions, k_new.to(dtype))
                v_local.index_copy_(2, local_important_positions, v_new.to(dtype))

                # Start async all-gather (fallback path only)
                async_handles = None
                if gather_ctx is not None:
                    async_handles, async_gather_lists = self._async_gather_layer_kv(
                        k_local, v_local, gather_ctx
                    )

                q_sparse = attn.q_proj(normed).view(B, K, self.num_heads, self.head_dim)
                q_sparse = q_sparse.transpose(1, 2)
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q_sparse = attn.q_norm(q_sparse)
                q_sparse = self._apply_rope(q_sparse, cos, sin, global_important_global)

                if cascade_ctx is not None:
                    attn_output = self._cascade_attention(
                        q_sparse, k_local, v_local, cascade_ctx,
                    )
                elif gather_ctx is not None:
                    k_full, v_full = self._finish_async_gather(
                        async_handles, async_gather_lists, gather_ctx
                    )
                    gathered_kv_layers.append((k_full, v_full))
                    attn_output = self._attention_with_full_kv(
                        q_sparse, gathered_kv_layers, layer_idx - 1, global_important_global,
                        use_flashinfer=use_flashinfer,
                    )
                else:
                    attn_output = self._local_attention(
                        q_sparse, k_local, v_local, global_important_global,
                        key_offset=global_offset,
                    )

                attn_output = attn_output.transpose(1, 2).reshape(B, K, self.hidden_size)
                attn_output = attn.o_proj(attn_output).to(dtype)

                hidden_states_sparse = hidden_states_sparse + attn_output
                residual = hidden_states_sparse
                mlp_input = layer.post_attention_layernorm(hidden_states_sparse)
                mlp_output = layer.mlp(mlp_input).to(dtype)
                hidden_states_sparse = residual + mlp_output

            # Update cache
            if isinstance(cache, DynamicCache):
                cache.key_cache[layer_idx] = k_local
                cache.value_cache[layer_idx] = v_local
            else:
                cache[layer_idx] = (k_local, v_local)

        # Build generation cache (same as recompute_distributed)
        gathered_cache = None
        if cascade_ctx is not None:
            gather_ctx_final = cascade_ctx["gather_ctx_final"]
            gathered_cache = DynamicCache()
            gathered_cache.key_cache = []
            gathered_cache.value_cache = []
            for li in range(self.num_layers):
                if isinstance(cache, DynamicCache):
                    k_l = cache.key_cache[li]
                    v_l = cache.value_cache[li]
                else:
                    k_l, v_l = cache[li]
                k_full, v_full = self._gather_layer_kv(k_l, v_l, gather_ctx_final)
                gathered_cache.key_cache.append(k_full)
                gathered_cache.value_cache.append(v_full)
        elif gathered_kv_layers:
            gathered_cache = DynamicCache()
            gathered_cache.key_cache = [kv[0] for kv in gathered_kv_layers]
            gathered_cache.value_cache = [kv[1] for kv in gathered_kv_layers]

        return DistributedKVCacheData(
            past_key_values=cache,
            input_ids=local_kv.input_ids,
            attention_mask=local_kv.attention_mask,
            chunk_lens=local_kv.chunk_lens,
            global_offset=local_kv.global_offset,
            global_total_len=local_kv.global_total_len,
            local_seq_len=local_kv.local_seq_len,
            gathered_full_kv=gathered_cache,
        )

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: torch.Tensor,
        key_offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute attention for single GPU or local-only case.

        Args:
            q: [B, num_heads, K, head_dim] query at sparse positions
            k: [B, num_kv_heads, T, head_dim] full key cache
            v: [B, num_kv_heads, T, head_dim] full value cache
            query_positions: [K] global positions of queries
            key_offset: Global position offset for keys (0 for single GPU)

        Returns:
            [B, num_heads, K, head_dim] attention output
        """
        B, H_q, K, D = q.shape
        T = k.size(2)
        H_kv = k.size(1)
        device = q.device

        # Handle GQA: expand K, V to match Q heads (zero-copy view)
        if H_kv < H_q:
            num_groups = H_q // H_kv
            B_k = k.size(0)
            k = k[:, :, None, :, :].expand(B_k, H_kv, num_groups, T, D).reshape(B_k, -1, T, D)
            v = v[:, :, None, :, :].expand(B_k, H_kv, num_groups, T, D).reshape(B_k, -1, T, D)

        # Create causal mask: q_pos can attend to k_pos if q_pos >= k_pos
        # key_offset maps local key index to global position
        query_positions = query_positions.view(-1)
        key_positions = torch.arange(T, device=device) + key_offset
        causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)  # [K, T]
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, K, T]

        # Compute attention with SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )

        return attn_output

    def _pre_gather_full_kv(
        self,
        local_kv: DistributedKVCacheData,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        All-gather full extraction KV for all layers once.

        This replaces per-layer all_gathers in the recompute loop.
        Cost: 1 length gather + num_layers × 2 K,V gathers.

        Args:
            local_kv: Local KV cache partition

        Returns:
            List of (k_full, v_full) tuples per layer, each [B, H_kv, total_T, D]
        """
        device = self.device
        world_size = self.config.world_size
        process_group = self.config.process_group
        cache = local_kv.past_key_values

        # Get local sequence length from first layer
        if isinstance(cache, DynamicCache):
            k0 = cache.key_cache[0]
        else:
            k0 = cache[0][0]
        if k0.dim() == 3:
            k0 = k0.unsqueeze(0)
        B, H_kv, local_T, D = k0.shape
        dtype = k0.dtype

        # Gather lengths once (1 collective)
        local_len_tensor = torch.tensor([local_T], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len_tensor, group=process_group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)

        # Pre-allocate padded buffers and gather lists (reused across layers)
        k_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        v_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        k_gathered = [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]
        v_gathered = [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]

        full_kv = []
        for layer_idx in range(self.num_layers):
            if isinstance(cache, DynamicCache):
                k_local = cache.key_cache[layer_idx]
                v_local = cache.value_cache[layer_idx]
            else:
                k_local, v_local = cache[layer_idx]

            if k_local.dim() == 3:
                k_local = k_local.unsqueeze(0)
                v_local = v_local.unsqueeze(0)

            # Pad and gather K
            k_padded.zero_()
            k_padded[:, :, :local_T, :] = k_local
            dist.all_gather(k_gathered, k_padded.contiguous(), group=process_group)

            # Pad and gather V
            v_padded.zero_()
            v_padded[:, :, :local_T, :] = v_local
            dist.all_gather(v_gathered, v_padded.contiguous(), group=process_group)

            # Concatenate and trim
            k_full = torch.cat([k_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
            v_full = torch.cat([v_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

            full_kv.append((k_full, v_full))

        return full_kv

    # Max mask elements per FlashInfer chunk (matches single-GPU limit)
    _MAX_MASK_ELEMS = 4_000_000

    def _attention_with_full_kv(
        self,
        q: torch.Tensor,
        full_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        layer_idx: int,
        query_positions: torch.Tensor,
        use_flashinfer: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention using pre-gathered full KV. No communication.

        Args:
            q: [B, num_heads, K, head_dim] sparse query
            full_kv: Pre-gathered list of (k_full, v_full) per layer
            layer_idx: Which layer's KV to use
            query_positions: [K] global positions of queries
            use_flashinfer: Use FlashInfer single_prefill_with_kv_cache

        Returns:
            [B, num_heads, K, head_dim] attention output
        """
        k_full, v_full = full_kv[layer_idx]
        total_len = k_full.size(2)
        device = q.device
        H_q = q.size(1)
        H_kv = k_full.size(1)
        query_positions = query_positions.view(-1)
        K = query_positions.numel()

        if use_flashinfer and K > 0:
            # Batch prefill with paged KV: each query is a separate
            # "request" with its own KV length (= query_position + 1).
            # This avoids custom_mask entirely, sidestepping a CUDA crash
            # in FlashInfer 0.2.0's single-prefill mask kernel.
            D_head = q.size(-1)
            page_size = 256

            # Convert to NHD layout: [B,H,T,D] -> [T,H,D]
            q_fi = q.squeeze(0).permute(1, 0, 2).contiguous()      # [K, H_q, D]
            k_fi = k_full.squeeze(0).permute(1, 0, 2).contiguous()  # [T, H_kv, D]
            v_fi = v_full.squeeze(0).permute(1, 0, 2).contiguous()  # [T, H_kv, D]

            # Cache wrapper + plan across layers (same query positions)
            plan_key = (K, total_len)
            if getattr(self, '_fi_plan_key', None) != plan_key:
                num_pages_total = (total_len + page_size - 1) // page_size
                self._fi_pad_t = num_pages_total * page_size - total_len
                self._fi_num_pages = num_pages_total

                qo_indptr = torch.arange(K + 1, dtype=torch.int32, device=device)
                kv_lens = (query_positions + 1).to(torch.int32)
                num_pages_per_req = (kv_lens + page_size - 1) // page_size
                last_page_lens = ((kv_lens - 1) % page_size + 1).to(torch.int32)

                paged_kv_indptr = torch.zeros(K + 1, dtype=torch.int32, device=device)
                paged_kv_indptr[1:] = torch.cumsum(num_pages_per_req, dim=0)

                max_pages = num_pages_per_req.max().item()
                page_range = torch.arange(max_pages, dtype=torch.int32, device=device)
                page_range_exp = page_range.unsqueeze(0).expand(K, -1)
                valid = page_range_exp < num_pages_per_req.unsqueeze(1)
                paged_kv_indices = page_range_exp[valid].contiguous()

                self._fi_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self._fi_workspace, "NHD"
                )
                self._fi_wrapper.plan(
                    qo_indptr, paged_kv_indptr, paged_kv_indices,
                    last_page_lens, H_q, H_kv, D_head, page_size,
                    causal=False,
                    q_data_type=q_fi.dtype,
                )
                self._fi_plan_key = plan_key

            # Convert KV to paged format: [T, H_kv, D] -> [num_pages, page_size, H_kv, D]
            if self._fi_pad_t > 0:
                k_paged = F.pad(k_fi, (0, 0, 0, 0, 0, self._fi_pad_t)).view(
                    self._fi_num_pages, page_size, H_kv, D_head)
                v_paged = F.pad(v_fi, (0, 0, 0, 0, 0, self._fi_pad_t)).view(
                    self._fi_num_pages, page_size, H_kv, D_head)
            else:
                k_paged = k_fi.view(self._fi_num_pages, page_size, H_kv, D_head)
                v_paged = v_fi.view(self._fi_num_pages, page_size, H_kv, D_head)

            attn_output = self._fi_wrapper.run(q_fi, (k_paged, v_paged))

            # [K, H_q, D] -> [1, H_q, K, D]
            return attn_output.permute(1, 0, 2).unsqueeze(0)

        # Fallback: SDPA with manual GQA expansion
        if K == 0:
            return q

        # GQA expansion (zero-copy view)
        if H_kv < H_q:
            num_groups = H_q // H_kv
            B_k, _, T_k, D_k = k_full.shape
            k_full = k_full[:, :, None, :, :].expand(B_k, H_kv, num_groups, T_k, D_k).reshape(B_k, -1, T_k, D_k)
            v_full = v_full[:, :, None, :, :].expand(B_k, H_kv, num_groups, T_k, D_k).reshape(B_k, -1, T_k, D_k)

        # Causal mask
        key_positions = torch.arange(total_len, device=device)
        causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        return F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=attn_mask, is_causal=False
        )

    def _cascade_attention(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cascade_ctx: dict,
    ) -> torch.Tensor:
        """
        Cascade local attention: all-gather Q, each GPU computes attention for
        ALL queries against its LOCAL KV shard, then all-gather output+lse and
        merge via merge_states.

        Args:
            q: [B, H_q, K_local, D] THIS GPU's queries (sparse positions)
            k_local: [B, H_kv, local_T, D] this GPU's local key cache
            v_local: [B, H_kv, local_T, D] this GPU's local value cache
            cascade_ctx: Pre-computed context from _init_cascade_ctx

        Returns:
            [B, H_q, K_local, D] attention output for this GPU's queries
        """
        K_total = cascade_ctx["K_total"]
        K_local = cascade_ctx["K_local"]
        max_K = cascade_ctx["max_K"]
        per_rank_K = cascade_ctx["per_rank_K"]
        local_K_offset = cascade_ctx["local_K_offset"]
        process_group = cascade_ctx["process_group"]
        world_size = cascade_ctx["world_size"]
        H_q = cascade_ctx["H_q"]
        H_kv = cascade_ctx["H_kv"]
        D_head = cascade_ctx["D_head"]

        device = q.device
        dtype = q.dtype

        # --- Step 1: All-gather Q across ranks ---
        # Convert to NHD: [B, H_q, K_local, D] -> [K_local, H_q, D]
        q_nhd = q.squeeze(0).permute(1, 0, 2).contiguous()

        # Pad to max_K so all ranks have equal-sized tensors
        q_padded = torch.zeros(max_K, H_q, D_head, device=device, dtype=dtype)
        q_padded[:K_local] = q_nhd

        q_gather = cascade_ctx["q_gather"]
        dist.all_gather(q_gather, q_padded.contiguous(), group=process_group)

        # Trim and concatenate: [K_total, H_q, D]
        q_all = torch.cat([q_gather[r][:per_rank_K[r]] for r in range(world_size)], dim=0)

        # --- Step 2: Compute attention for ALL queries against local KV ---
        local_output = torch.zeros(K_total, H_q, D_head, device=device, dtype=dtype)
        local_lse = torch.full((K_total, H_q), float('-inf'), device=device, dtype=torch.float32)

        # Convert local KV to NHD layout: [B,H_kv,T,D] -> [T,H_kv,D]
        k_nhd = k_local.squeeze(0).permute(1, 0, 2).contiguous()
        v_nhd = v_local.squeeze(0).permute(1, 0, 2).contiguous()

        # Full group: queries past this GPU's range → see ALL local KV
        full_idx = cascade_ctx["full_idx"]
        partial_idx = cascade_ctx["partial_idx"]
        if full_idx.numel() > 0:
            q_full = q_all[full_idx].contiguous()
            out_full, lse_full = flashinfer.single_prefill_with_kv_cache(
                q_full, k_nhd, v_nhd,
                causal=False, return_lse=True, kv_layout="NHD",
            )
            local_output[full_idx] = out_full
            local_lse[full_idx] = lse_full

        # Partial group: queries within this GPU's range → variable KV lengths
        partial_wrapper = cascade_ctx["partial_wrapper"]
        if partial_idx.numel() > 0 and partial_wrapper is not None:
            q_partial = q_all[partial_idx].contiguous()

            page_size = cascade_ctx["page_size"]
            num_pages_total = cascade_ctx["num_pages_total"]
            pad_t = cascade_ctx["pad_t"]

            if pad_t > 0:
                k_paged = F.pad(k_nhd, (0, 0, 0, 0, 0, pad_t)).view(
                    num_pages_total, page_size, H_kv, D_head)
                v_paged = F.pad(v_nhd, (0, 0, 0, 0, 0, pad_t)).view(
                    num_pages_total, page_size, H_kv, D_head)
            else:
                k_paged = k_nhd.view(num_pages_total, page_size, H_kv, D_head)
                v_paged = v_nhd.view(num_pages_total, page_size, H_kv, D_head)

            out_partial, lse_partial = partial_wrapper.run(
                q_partial, (k_paged, v_paged), return_lse=True,
            )
            local_output[partial_idx] = out_partial
            local_lse[partial_idx] = lse_partial

        # None group: output=0, lse=-inf (contributes nothing to merge)

        # --- Step 3: All-gather output+lse across GPUs, merge ---
        out_gather = cascade_ctx["out_gather"]
        lse_gather = cascade_ctx["lse_gather"]
        dist.all_gather(out_gather, local_output.contiguous(), group=process_group)
        dist.all_gather(lse_gather, local_lse.contiguous(), group=process_group)

        # Stack: [K_total, world_size, H_q, D] and [K_total, world_size, H_q]
        v_stacked = torch.stack(out_gather, dim=1)
        s_stacked = torch.stack(lse_gather, dim=1)

        merged_out, _ = merge_states(v_stacked, s_stacked)  # [K_total, H_q, D]

        # --- Step 4: Extract this rank's queries ---
        local_out = merged_out[local_K_offset:local_K_offset + K_local]  # [K_local, H_q, D]

        # Reshape to [B, H_q, K_local, D]
        return local_out.permute(1, 0, 2).unsqueeze(0)

    def _init_gather_buffers(
        self,
        local_kv: DistributedKVCacheData,
    ) -> dict:
        """
        Pre-allocate buffers for per-layer KV all_gather. Called once before the loop.

        Includes double-buffered async buffers (set A and B) so that while
        layer N's async gather is in flight on set A, layer N+1 can write
        into set B without waiting, eliminating per-layer allocations.
        """
        device = self.device
        world_size = self.config.world_size
        process_group = self.config.process_group
        cache = local_kv.past_key_values

        if isinstance(cache, DynamicCache):
            k0 = cache.key_cache[0]
        else:
            k0 = cache[0][0]
        if k0.dim() == 3:
            k0 = k0.unsqueeze(0)
        B, H_kv, local_T, D = k0.shape
        dtype = k0.dtype

        local_len_tensor = torch.tensor([local_T], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len_tensor, group=process_group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)

        # Synchronous gather buffers (for layer 0)
        k_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        v_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        k_gather_list = [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]
        v_gather_list = [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]

        # Double-buffered async gather buffers (for layers 1+)
        def _make_async_set():
            return {
                "k_padded": torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype),
                "v_padded": torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype),
                "k_gather": [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)],
                "v_gather": [torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype) for _ in range(world_size)],
            }

        return {
            "all_lens": all_lens, "max_len": max_len, "local_T": local_T,
            "k_padded": k_padded, "v_padded": v_padded,
            "k_gather_list": k_gather_list, "v_gather_list": v_gather_list,
            "process_group": process_group, "world_size": world_size,
            "async_a": _make_async_set(), "async_b": _make_async_set(),
            "flip": False,
        }

    def _init_cascade_ctx(
        self,
        K_local: int,
        local_T: int,
        global_offset: int,
        total_len: int,
        all_query_positions: torch.Tensor,
        local_kv: "DistributedKVCacheData",
    ) -> dict:
        """
        Pre-compute cascade attention context. Called once before the layer loop.

        Each GPU computes attention for ALL queries (from all GPUs) against its
        LOCAL KV shard. This requires all-gathering Q per layer, but avoids the
        much larger per-layer KV all-gather.

        Classifies ALL queries into three groups relative to this GPU's local KV:
        - full_idx: queries past this GPU's range → attend to ALL local KV
        - partial_idx: queries within this GPU's range → variable KV lengths
        - none_idx: queries before this GPU's range → no local KV to attend

        Pre-plans a BatchPrefillWithPagedKVCacheWrapper for the partial group
        (reused across all layers).
        """
        device = self.device
        world_size = self.config.world_size
        rank = self.config.rank
        process_group = self.config.process_group

        all_query_positions = all_query_positions.view(-1).to(device)
        K_total = all_query_positions.numel()  # Same on all ranks (num_pos from scorer)

        # Ensure all ranks have exactly the same query positions.
        # torch.topk tie-breaking can differ across GPUs with bf16 scores,
        # causing per_rank_K/max_K to diverge → NCCL size mismatch.
        dist.broadcast(all_query_positions, src=0, group=process_group)

        # --- Determine per-rank K counts ---
        # All-gather actual K_local from each rank (authoritative, matches Q tensors).
        K_local_tensor = torch.tensor([K_local], device=device, dtype=torch.long)
        all_K_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_K_list, K_local_tensor, group=process_group)
        per_rank_K = [int(k.item()) for k in all_K_list]

        max_K = max(per_rank_K)
        per_rank_K_offset = []
        cumulative = 0
        for k in per_rank_K:
            per_rank_K_offset.append(cumulative)
            cumulative += k
        local_K_offset = per_rank_K_offset[rank]

        # Gather local sequence lengths for rank boundaries
        local_len_tensor = torch.tensor([local_T], device=device, dtype=torch.long)
        all_lens_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens_list, local_len_tensor, group=process_group)
        all_lens = [int(l.item()) for l in all_lens_list]

        # --- Classify ALL K_total queries against this GPU's local KV ---
        local_end = global_offset + local_T

        full_mask = all_query_positions >= local_end
        partial_mask = (all_query_positions >= global_offset) & (all_query_positions < local_end)
        none_mask = all_query_positions < global_offset

        full_idx = torch.where(full_mask)[0]
        partial_idx = torch.where(partial_mask)[0]
        none_idx = torch.where(none_mask)[0]

        # --- Paged KV setup for partial group ---
        page_size = 256
        num_pages_total = (local_T + page_size - 1) // page_size
        pad_t = num_pages_total * page_size - local_T

        H_kv = self.num_kv_heads
        H_q = self.num_heads
        D_head = self.head_dim

        partial_wrapper = None
        if partial_idx.numel() > 0:
            partial_positions = all_query_positions[partial_idx]
            kv_lens = (partial_positions - global_offset + 1).to(torch.int32)

            K_partial = partial_idx.numel()
            qo_indptr = torch.arange(K_partial + 1, dtype=torch.int32, device=device)
            num_pages_per_req = (kv_lens + page_size - 1) // page_size
            last_page_lens = ((kv_lens - 1) % page_size + 1).to(torch.int32)

            paged_kv_indptr = torch.zeros(K_partial + 1, dtype=torch.int32, device=device)
            paged_kv_indptr[1:] = torch.cumsum(num_pages_per_req, dim=0)

            max_pages = num_pages_per_req.max().item()
            page_range = torch.arange(max_pages, dtype=torch.int32, device=device)
            page_range_exp = page_range.unsqueeze(0).expand(K_partial, -1)
            valid = page_range_exp < num_pages_per_req.unsqueeze(1)
            paged_kv_indices = page_range_exp[valid].contiguous()

            cascade_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            partial_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                cascade_workspace, "NHD"
            )
            partial_wrapper.plan(
                qo_indptr, paged_kv_indptr, paged_kv_indices,
                last_page_lens, H_q, H_kv, D_head, page_size,
                causal=False,
                q_data_type=self.model.dtype,
            )

        # --- Pre-allocate gather buffers ---
        dtype = self.model.dtype

        # Q gather: pad to max_K per rank (K_local varies across ranks)
        q_gather = [torch.zeros(max_K, H_q, D_head, device=device, dtype=dtype)
                    for _ in range(world_size)]

        # Output + LSE gather: K_total is the SAME on all ranks
        out_gather = [torch.empty(K_total, H_q, D_head, device=device, dtype=dtype)
                      for _ in range(world_size)]
        lse_gather = [torch.empty(K_total, H_q, device=device, dtype=torch.float32)
                      for _ in range(world_size)]

        # Final KV gather for generation cache
        gather_ctx_final = self._init_gather_buffers(local_kv)

        return {
            "full_idx": full_idx, "partial_idx": partial_idx, "none_idx": none_idx,
            "partial_wrapper": partial_wrapper,
            "page_size": page_size, "num_pages_total": num_pages_total, "pad_t": pad_t,
            "q_gather": q_gather, "out_gather": out_gather, "lse_gather": lse_gather,
            "process_group": process_group, "world_size": world_size,
            "K_total": K_total, "K_local": K_local, "max_K": max_K,
            "per_rank_K": per_rank_K, "local_K_offset": local_K_offset,
            "H_q": H_q, "H_kv": H_kv, "D_head": D_head, "local_T": local_T,
            "gather_ctx_final": gather_ctx_final,
        }

    def _gather_layer_kv(
        self,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        ctx: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-gather K,V for a single layer from all ranks."""
        all_lens = ctx["all_lens"]
        local_T = ctx["local_T"]
        k_padded = ctx["k_padded"]
        v_padded = ctx["v_padded"]
        k_gather_list = ctx["k_gather_list"]
        v_gather_list = ctx["v_gather_list"]
        process_group = ctx["process_group"]
        world_size = ctx["world_size"]

        if k_local.dim() == 3:
            k_local = k_local.unsqueeze(0)
            v_local = v_local.unsqueeze(0)

        k_padded.zero_()
        k_padded[:, :, :local_T, :] = k_local
        dist.all_gather(k_gather_list, k_padded.contiguous(), group=process_group)

        v_padded.zero_()
        v_padded[:, :, :local_T, :] = v_local
        dist.all_gather(v_gather_list, v_padded.contiguous(), group=process_group)

        k_full = torch.cat([k_gather_list[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
        v_full = torch.cat([v_gather_list[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

        return k_full, v_full

    def _async_gather_layer_kv(
        self,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        ctx: dict,
    ) -> Tuple[List[Any], List[Any]]:
        """Start non-blocking all-gather for K,V using pre-allocated double buffers.

        Alternates between async_a and async_b buffer sets via a flip flag.
        While layer N's async is in flight on one set, layer N+1 writes into
        the other, eliminating per-layer tensor allocations.
        """
        local_T = ctx["local_T"]
        process_group = ctx["process_group"]

        if k_local.dim() == 3:
            k_local = k_local.unsqueeze(0)
            v_local = v_local.unsqueeze(0)

        # Select buffer set and flip for next call
        buf = ctx["async_a"] if not ctx["flip"] else ctx["async_b"]
        ctx["flip"] = not ctx["flip"]

        # Fill pre-allocated padded buffers
        k_pad = buf["k_padded"]
        v_pad = buf["v_padded"]
        k_pad.zero_()
        v_pad.zero_()
        k_pad[:, :, :local_T, :] = k_local
        v_pad[:, :, :local_T, :] = v_local

        k_gather = buf["k_gather"]
        v_gather = buf["v_gather"]

        k_handle = dist.all_gather(k_gather, k_pad.contiguous(), group=process_group, async_op=True)
        v_handle = dist.all_gather(v_gather, v_pad.contiguous(), group=process_group, async_op=True)

        return [k_handle, v_handle], [k_gather, v_gather]

    def _finish_async_gather(
        self,
        handles: List[Any],
        gather_lists: List[List[torch.Tensor]],
        ctx: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wait for async all-gather and return concatenated full K,V."""
        for h in handles:
            h.wait()

        all_lens = ctx["all_lens"]
        world_size = ctx["world_size"]
        k_gather_async, v_gather_async = gather_lists

        k_full = torch.cat([k_gather_async[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
        v_full = torch.cat([v_gather_async[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

        return k_full, v_full

    def _init_ring_ctx(
        self,
        local_kv: DistributedKVCacheData,
    ) -> dict:
        """
        Pre-allocate ring attention buffers. Called once before the recompute loop.

        Gathers sequence lengths (1 collective), computes offsets, and allocates
        double buffers for async send/recv during ring KV rotation.
        """
        device = self.device
        world_size = self.config.world_size
        rank = self.config.rank
        process_group = self.config.process_group
        cache = local_kv.past_key_values

        if isinstance(cache, DynamicCache):
            k0 = cache.key_cache[0]
        else:
            k0 = cache[0][0]
        if k0.dim() == 3:
            k0 = k0.unsqueeze(0)
        B, H_kv, local_T, D = k0.shape
        dtype = k0.dtype

        # Gather lengths once (1 all_gather for all layers)
        local_len_t = torch.tensor([local_T], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len_t, group=process_group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)

        # Compute global offsets for causal masking
        offsets = [0]
        for l in all_lens[:-1]:
            offsets.append(offsets[-1] + l)

        # Double buffers for ring send/recv (reused across all layers)
        buf_a_k = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        buf_a_v = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        buf_b_k = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        buf_b_v = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)

        return {
            "all_lens": all_lens, "max_len": max_len, "offsets": offsets,
            "local_T": local_T, "world_size": world_size, "rank": rank,
            "process_group": process_group,
            "buf_a_k": buf_a_k, "buf_a_v": buf_a_v,
            "buf_b_k": buf_b_k, "buf_b_v": buf_b_v,
        }

    def _ring_attention(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        query_positions: torch.Tensor,
        ctx: dict,
    ) -> torch.Tensor:
        """
        Ring attention: Q stays local, KV rotates through the ring.

        Communication is overlapped with computation via async P2P send/recv.
        Uses online softmax (sigmoid/logsigmoid trick from ring-flash-attention)
        to combine partial attentions from each KV chunk.

        Mathematically identical to all_gather KV + full attention, but with
        overlapped communication and lower peak memory.

        Args:
            q: [B, H_q, K_sparse, D] sparse query at important positions
            k_local: [B, H_kv, local_T, D] this layer's local key cache
            v_local: [B, H_kv, local_T, D] this layer's local value cache
            query_positions: [K_sparse] global positions of queries
            ctx: Ring context from _init_ring_ctx

        Returns:
            [B, H_q, K_sparse, D] attention output (same as full-KV attention)
        """
        all_lens = ctx["all_lens"]
        offsets = ctx["offsets"]
        world_size = ctx["world_size"]
        rank = ctx["rank"]
        process_group = ctx["process_group"]

        device = q.device
        dtype = q.dtype
        B, H_q, K_sparse, D = q.shape
        H_kv = k_local.size(1)
        num_groups = H_q // H_kv if H_kv < H_q else 1

        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        # Double buffers: cur = compute source, nxt = recv target
        cur_k = ctx["buf_a_k"]
        cur_v = ctx["buf_a_v"]
        nxt_k = ctx["buf_b_k"]
        nxt_v = ctx["buf_b_v"]

        # Load this layer's local KV into cur buffer
        local_T = k_local.size(2)
        cur_k.zero_()
        cur_v.zero_()
        cur_k[:, :, :local_T, :] = k_local
        cur_v[:, :, :local_T, :] = v_local

        # Online softmax accumulators (float32 for precision)
        acc_out = None
        acc_lse = None

        query_pos = query_positions.view(-1)
        q_f32 = q.float()

        for step in range(world_size):
            # Which rank's KV are we processing?
            kv_rank = (rank - step) % world_size
            kv_len = all_lens[kv_rank]
            kv_offset = offsets[kv_rank]

            # Start async ring communication for next step (overlap with compute)
            reqs = []
            if step < world_size - 1:
                reqs = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, cur_k.contiguous(), next_rank, group=process_group),
                    dist.P2POp(dist.isend, cur_v.contiguous(), next_rank, group=process_group),
                    dist.P2POp(dist.irecv, nxt_k, prev_rank, group=process_group),
                    dist.P2POp(dist.irecv, nxt_v, prev_rank, group=process_group),
                ])

            # Causal check: query at pos P can attend to key at pos K only if P >= K
            # For later ranks' KV (kv_offset > max query pos), skip entirely
            min_key_pos = kv_offset
            max_query_pos = query_pos.max().item() if K_sparse > 0 else -1
            if max_query_pos >= min_key_pos:
                # Extract actual KV (unpad) and GQA expand (zero-copy view + contiguous for matmul)
                k_chunk = cur_k[:, :, :kv_len, :]
                v_chunk = cur_v[:, :, :kv_len, :]
                if num_groups > 1:
                    B_k = k_chunk.size(0)
                    D_k = k_chunk.size(3)
                    k_chunk = k_chunk[:, :, None, :, :].expand(B_k, H_kv, num_groups, kv_len, D_k).reshape(B_k, -1, kv_len, D_k).contiguous()
                    v_chunk = v_chunk[:, :, None, :, :].expand(B_k, H_kv, num_groups, kv_len, D_k).reshape(B_k, -1, kv_len, D_k).contiguous()

                # Causal mask
                key_pos = torch.arange(kv_len, device=device) + kv_offset
                causal_mask = query_pos.unsqueeze(1) >= key_pos.unsqueeze(0)  # [K_sparse, kv_len]

                # Compute QK^T with manual scaling (need LSE for online softmax)
                scale = 1.0 / (D ** 0.5)
                scores = torch.matmul(q_f32, k_chunk.float().transpose(-2, -1)) * scale
                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )

                # Partial attention output and LSE
                block_lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B, H_q, K_sparse, 1]
                block_probs = torch.exp(scores - block_lse)
                block_out = torch.matmul(block_probs, v_chunk.float())  # [B, H_q, K_sparse, D]

                # Online softmax combination (sigmoid/logsigmoid trick)
                if acc_out is None:
                    acc_out = block_out
                    acc_lse = block_lse
                else:
                    # Numerically stable update from ring-flash-attention:
                    # out = out - sigmoid(block_lse - lse) * (out - block_out)
                    # lse = lse - logsigmoid(lse - block_lse)
                    acc_out = acc_out - F.sigmoid(block_lse - acc_lse) * (acc_out - block_out)
                    acc_lse = acc_lse - F.logsigmoid(acc_lse - block_lse)

            # Wait for ring communication to complete
            for req in reqs:
                req.wait()

            # Swap double buffers: received KV becomes current for next step
            if step < world_size - 1:
                cur_k, nxt_k = nxt_k, cur_k
                cur_v, nxt_v = nxt_v, cur_v

        return acc_out.to(dtype) if acc_out is not None else torch.zeros_like(q)

    def _manual_ring_attention(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        query_positions: torch.Tensor,
        global_offset: int,
        local_T: int,
    ) -> torch.Tensor:
        """
        Simplified distributed attention using all_gather.

        Instead of complex ring communication, we all_gather K,V and compute locally.
        This is simpler and more reliable, though uses more memory.

        Args:
            q: [B, num_heads, K, head_dim] sparse query
            k_local: [B, num_kv_heads, local_T, head_dim] local key cache
            v_local: [B, num_kv_heads, local_T, head_dim] local value cache
            query_positions: [K] global positions of queries
            global_offset: Global offset for this rank's sequence
            local_T: Local sequence length

        Returns:
            [B, num_heads, K, head_dim] attention output
        """
        device = q.device
        dtype = q.dtype
        B, H_q, K_sparse, D = q.shape
        H_kv = k_local.size(1)
        world_size = self.config.world_size
        process_group = self.config.process_group

        # All-gather K and V from all ranks (using native KV heads to save bandwidth)
        # GQA expansion happens AFTER all-gather to avoid multiplying communication by num_groups
        local_len_tensor = torch.tensor([local_T], device=device, dtype=torch.long)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len_tensor, group=process_group)
        all_lens = [int(l.item()) for l in all_lens]
        max_len = max(all_lens)
        total_len = sum(all_lens)

        # Pad local K, V to max_len for all_gather
        k_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        v_padded = torch.zeros(B, H_kv, max_len, D, device=device, dtype=dtype)
        k_padded[:, :, :local_T, :] = k_local
        v_padded[:, :, :local_T, :] = v_local

        # All-gather
        k_gathered = [torch.zeros_like(k_padded) for _ in range(world_size)]
        v_gathered = [torch.zeros_like(v_padded) for _ in range(world_size)]
        dist.all_gather(k_gathered, k_padded, group=process_group)
        dist.all_gather(v_gathered, v_padded, group=process_group)

        # Concatenate and trim to actual lengths
        k_full = torch.cat([k_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
        v_full = torch.cat([v_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

        # Ensure query_positions is 1D
        query_positions = query_positions.view(-1)

        if FLASHINFER_AVAILABLE:
            # FlashInfer: native GQA, custom_mask, ~150 TFLOPS
            K = query_positions.numel()
            if K == 0:
                return q  # empty [B, H_q, 0, D]

            # Truncate KV to max needed position (positions beyond are masked out).
            max_pos = query_positions.max().item() + 1

            # Convert [B, H, T, D] → [T, H, D]
            q_fi = q.squeeze(0).permute(1, 0, 2).contiguous()
            k_fi = k_full[:, :, :max_pos, :].squeeze(0).permute(1, 0, 2).contiguous()
            v_fi = v_full[:, :, :max_pos, :].squeeze(0).permute(1, 0, 2).contiguous()

            key_positions = torch.arange(max_pos, device=device)
            causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

            attn_output = flashinfer.single_prefill_with_kv_cache(
                q_fi, k_fi, v_fi,
                custom_mask=causal_mask,
                causal=False,
                kv_layout="NHD",
            )
            return attn_output.permute(1, 0, 2).unsqueeze(0)
        else:
            # Fallback: SDPA with manual GQA expansion
            if H_kv < H_q:
                num_groups = H_q // H_kv
                B_k, _, T_k, D_k = k_full.shape
                k_full = k_full[:, :, None, :, :].expand(B_k, H_kv, num_groups, T_k, D_k).reshape(B_k, -1, T_k, D_k)
                v_full = v_full[:, :, None, :, :].expand(B_k, H_kv, num_groups, T_k, D_k).reshape(B_k, -1, T_k, D_k)

            key_positions = torch.arange(total_len, device=device)
            causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            attn_output = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=attn_mask, is_causal=False
            )
            return attn_output


def all_gather_kv(
    local_kv: DistributedKVCacheData,
    config: DistributedConfig,
) -> DynamicCache:
    """
    All-gather KV cache from all GPUs.

    This reconstructs the full KV cache on each GPU for generation.

    Args:
        local_kv: Local KV cache partition
        config: Distributed configuration

    Returns:
        DynamicCache with full KV cache (same on all ranks)
    """
    if not config.enabled or config.world_size == 1:
        return local_kv.past_key_values

    # If recompute_distributed already gathered the full KV, reuse it
    if local_kv.gathered_full_kv is not None:
        return local_kv.gathered_full_kv

    device = local_kv.device
    world_size = config.world_size

    # Change 3a: Gather sequence lengths ONCE outside the layer loop
    # All layers have the same local_T, so this is redundant per-layer
    k0 = local_kv.past_key_values.key_cache[0]
    if k0.dim() == 3:
        k0 = k0.unsqueeze(0)
    B, H, local_T, D = k0.shape

    local_len_tensor = torch.tensor([local_T], device=device, dtype=torch.long)
    all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_lens, local_len_tensor, group=config.process_group)
    all_lens = [int(l.item()) for l in all_lens]
    max_len = max(all_lens)

    num_layers = local_kv.num_layers
    dtype = local_kv.dtype

    full_cache = DynamicCache()
    full_cache.key_cache = []
    full_cache.value_cache = []

    # Change 3b: Pre-allocate padded buffers and gather lists (reused across layers)
    k_padded = torch.zeros(B, H, max_len, D, device=device, dtype=dtype)
    v_padded = torch.zeros(B, H, max_len, D, device=device, dtype=dtype)
    k_gathered = [torch.zeros(B, H, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]
    v_gathered = [torch.zeros(B, H, max_len, D, device=device, dtype=dtype) for _ in range(world_size)]

    for layer_idx in range(num_layers):
        k_local = local_kv.past_key_values.key_cache[layer_idx]
        v_local = local_kv.past_key_values.value_cache[layer_idx]

        if k_local.dim() == 3:
            k_local = k_local.unsqueeze(0)
            v_local = v_local.unsqueeze(0)

        # Change 3b: Use all_gather + concatenate + trim instead of all_reduce
        # all_gather communicates only each rank's padded data,
        # while all_reduce communicates the FULL tensor from every rank
        k_padded.zero_()
        k_padded[:, :, :local_T, :] = k_local
        dist.all_gather(k_gathered, k_padded.contiguous(), group=config.process_group)

        v_padded.zero_()
        v_padded[:, :, :local_T, :] = v_local
        dist.all_gather(v_gathered, v_padded.contiguous(), group=config.process_group)

        # Concatenate and trim to actual lengths
        k_full = torch.cat([k_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)
        v_full = torch.cat([v_gathered[r][:, :, :all_lens[r], :] for r in range(world_size)], dim=2)

        full_cache.key_cache.append(k_full)
        full_cache.value_cache.append(v_full)

    return full_cache


def allgather_positions(
    local_positions: torch.Tensor,
    global_offset: int,
    config: DistributedConfig,
) -> torch.Tensor:
    """
    All-gather locally-selected important positions into a globally-consistent tensor.

    Each GPU selects positions independently (e.g., via V-diff or LEGO rule).
    This function converts local indices to global, all-gathers, sorts, and
    broadcasts from rank 0 for deterministic ordering.

    Args:
        local_positions: Local indices on this GPU [K_local]
        global_offset: This GPU's global position offset
        config: DistributedConfig

    Returns:
        global_important: Sorted global positions, identical on all ranks
    """
    device = local_positions.device
    world_size = config.world_size

    if not config.enabled or world_size == 1:
        return local_positions + global_offset

    # Convert to global
    local_global = local_positions + global_offset

    # All-gather lengths
    local_len = torch.tensor([local_global.numel()], device=device, dtype=torch.long)
    all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_lens, local_len, group=config.process_group)
    all_lens_int = [int(l.item()) for l in all_lens]
    max_len = max(all_lens_int)

    # Pad and all-gather
    padded = torch.zeros(max_len, device=device, dtype=torch.long)
    padded[:local_global.numel()] = local_global
    all_padded = [torch.zeros(max_len, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_padded, padded, group=config.process_group)

    # Concatenate, trim, sort
    global_important = torch.cat([all_padded[r][:all_lens_int[r]] for r in range(world_size)])
    global_important = global_important.sort().values

    # Broadcast from rank 0 for consistency (same tie-breaking as scorer)
    dist.broadcast(global_important, src=0, group=config.process_group)

    return global_important

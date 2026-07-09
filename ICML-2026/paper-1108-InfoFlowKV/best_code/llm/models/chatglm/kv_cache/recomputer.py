"""KV cache recomputation with selective updates for ChatGLM."""

import torch
import torch.nn.functional as F
import numpy as np
import time
from bisect import bisect_right
from itertools import chain
from typing import List, Tuple, Optional, Any
from transformers.cache_utils import DynamicCache

from .base import KVCacheData


class KVCacheRecomputer:
    """
    Selectively recompute KV cache at important positions for ChatGLM.
    
    Strategy:
    - Layer 0: Recompute full KV (need correct hidden states for subsequent layers)
    - Layer 1+: Only recompute at selected indices
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_type: str = "glm",
    ):
        """
        Args:
            model: The ChatGLM model
            tokenizer: The tokenizer
            model_type: Model type (default: "glm")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.device = next(model.parameters()).device

        # Get model config
        config = model.config
        self.num_layers = getattr(config, "num_layers", 40)
        self.num_heads = getattr(config, "num_attention_heads", 32)
        self.num_kv_heads = getattr(config, "multi_query_group_num", 2)
        self.head_dim = getattr(config, "hidden_size", 4096) // max(1, self.num_heads)
        self.hidden_size = getattr(config, "hidden_size", 4096)

    @torch.no_grad()
    def recompute(
        self,
        kv_data: KVCacheData,
        recompute_indices: np.ndarray,
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
        
        recompute_indices = torch.unique(recompute_indices, sorted=True)
        K = recompute_indices.numel()

        # Get input_ids and compute embeddings
        input_ids = kv_data.input_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        
        batch_size, seq_len = input_ids.shape
        
        # Compute input embeddings from input_ids
        embed_layer = self.model.get_input_embeddings()
        hidden_states = embed_layer(input_ids).to(dtype)  # [B, T, H]
        
        # Compute RoPE embeddings (GLM format: [T, rot_dim//2, 2])
        rope_len = seq_len
        cos_sin_full = self.model.transformer.rotary_pos_emb(rope_len).to(device=device, dtype=dtype)

        # Get the cache (DynamicCache)
        cache = kv_data.past_key_values

        # Get model layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'encoder'):
            layers = self.model.transformer.encoder.layers
        else:
            raise ValueError("Cannot find ChatGLM model layers")

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
                    layer, hidden_states, k_cache, v_cache, cos_sin_full,
                    recompute_indices, batch_size, seq_len
                )
            else:
                # Layer 1+: Only recompute at selected indices
                hidden_states, k_cache, v_cache = self._recompute_layer_sparse(
                    layer, hidden_states, k_cache, v_cache, cos_sin_full,
                    recompute_indices, batch_size, K
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
        cos_sin_full: torch.Tensor,
        recompute_indices: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
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
        attn = layer.self_attention

        # Layer norm
        normed = layer.input_layernorm(hidden_states)

        # Compute mixed QKV
        mixed_x_layer = attn.query_key_value(normed)

        # Split into Q, K, V
        if attn.multi_query_attention:
            # GQA: [B, T, (H + 2*KV_H) * D]
            qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
            q, k, v = mixed_x_layer.split([
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim
            ], dim=-1)
            
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        else:
            q, k, v = mixed_x_layer.split([
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim
            ], dim=-1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, cos_sin_full)
        k = self._apply_rope(k, cos_sin_full)

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
            q_for_attn, k_cache, v_cache, query_positions
        )

        # Project output
        output_len = attn_output.size(2)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, output_len, self.hidden_size
        )
        attn_output = attn.dense(attn_output).to(dtype)

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
        cos_sin_full: torch.Tensor,
        recompute_indices: torch.Tensor,
        batch_size: int,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute layer at sparse positions only.

        Args:
            hidden_states: [B, K, H] hidden states at recompute positions

        Returns:
            (hidden_states_out, updated_k_cache, updated_v_cache)
        """
        dtype = hidden_states.dtype
        attn = layer.self_attention

        # Layer norm
        normed = layer.input_layernorm(hidden_states)

        # Compute mixed QKV
        mixed_x_layer = attn.query_key_value(normed)

        # Split into Q, K, V
        if attn.multi_query_attention:
            q, k_new, v_new = mixed_x_layer.split([
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim
            ], dim=-1)
            
            q = q.view(batch_size, K, self.num_heads, self.head_dim)
            k_new = k_new.view(batch_size, K, self.num_kv_heads, self.head_dim)
            v_new = v_new.view(batch_size, K, self.num_kv_heads, self.head_dim)
        else:
            q, k_new, v_new = mixed_x_layer.split([
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim
            ], dim=-1)
            q = q.view(batch_size, K, self.num_heads, self.head_dim)
            k_new = k_new.view(batch_size, K, self.num_heads, self.head_dim)
            v_new = v_new.view(batch_size, K, self.num_heads, self.head_dim)

        # Transpose to [B, H, K, D]
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        # Apply RoPE at correct positions
        q = self._apply_rope(q, cos_sin_full, recompute_indices)
        k_new = self._apply_rope(k_new, cos_sin_full, recompute_indices)

        # Update cache at recompute positions
        k_cache = k_cache.index_copy(2, recompute_indices, k_new.to(dtype))
        v_cache = v_cache.index_copy(2, recompute_indices, v_new.to(dtype))

        # Compute attention
        attn_output = self._compute_attention(q, k_cache, v_cache, recompute_indices)

        # Project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, K, self.hidden_size)
        attn_output = attn.dense(attn_output).to(dtype)

        # Residual connection
        hidden_states = hidden_states + attn_output

        # MLP
        residual = hidden_states
        mlp_out = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states = residual + mlp_out

        return hidden_states, k_cache, v_cache



    def _select_rope(
        self,
        rope_cache: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select RoPE embeddings at given positions.

        Args:
            rope_cache: [S, dim, 2] from transformer.rotary_pos_emb
            position_ids: [T] or [B, K] positions to select

        Returns:
            Selected rope embeddings
        """
        if position_ids is None:
            return rope_cache
        if position_ids.dim() == 1:
            return rope_cache.index_select(0, position_ids)
        elif position_ids.dim() == 2:
            B, K = position_ids.shape
            flat = position_ids.reshape(-1)
            rope_flat = rope_cache.index_select(0, flat)
            return rope_flat.view(B, K, rope_cache.size(1), 2)
        else:
            raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")

    def _apply_rope(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply RoPE to tensor x (GLM style).

        Args:
            x: [B, H, T, D] tensor
            rope_cache: [S, dim, 2] from transformer.rotary_pos_emb
            position_ids: [T] or [B, K] positions to select

        Returns:
            [B, H, T, D] tensor with RoPE applied
        """
        if position_ids is None:
            rope = rope_cache[: x.size(2)]  # [T, dim, 2]
        else:
            rope = self._select_rope(rope_cache, position_ids)  # Use helper
        
        if rope.dim() == 3:  # [T, dim, 2]
            rope = rope.unsqueeze(0)  # [1, T, dim, 2]
        rope = rope.unsqueeze(1)  # [1, 1, T, dim, 2]
        
        rot_dim = rope.shape[-2] * 2
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:] if x.size(-1) > rot_dim else None
        
        x_rot = x_rot.view(x_rot.size(0), x_rot.size(1), x_rot.size(2), rot_dim // 2, 2)
        cos = rope[..., 0]  # [1, 1, T, dim]
        sin = rope[..., 1]  # [1, 1, T, dim]
        
        real = x_rot[..., 0]
        imag = x_rot[..., 1]
        rot_real = real * cos - imag * sin
        rot_imag = imag * cos + real * sin
        x_out2 = torch.stack([rot_real, rot_imag], dim=-1).flatten(3)  # [B, H, T, rot_dim]
        
        if x_pass is not None and x_pass.numel() > 0:
            return torch.cat([x_out2, x_pass], dim=-1)
        return x_out2

    def _remove_rope(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Remove RoPE from tensor x (GLM style).

        Args:
            x: [B, H, T, D] tensor with RoPE applied
            rope_cache: [S, dim, 2] from transformer.rotary_pos_emb
            position_ids: [T] or [B, K] positions to select

        Returns:
            [B, H, T, D] tensor with RoPE removed
        """
        if position_ids is None:
            rope = rope_cache[: x.size(2)]
        else:
            rope = self._select_rope(rope_cache, position_ids)  # Use helper
        
        if rope.dim() == 3:
            rope = rope.unsqueeze(0)
        rope = rope.unsqueeze(1)  # [1, 1, T, dim, 2]
        
        rot_dim = rope.shape[-2] * 2
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:] if x.size(-1) > rot_dim else None
        
        x_rot = x_rot.view(x_rot.size(0), x_rot.size(1), x_rot.size(2), rot_dim // 2, 2)
        cos = rope[..., 0]
        sin = rope[..., 1]
        
        real = x_rot[..., 0]
        imag = x_rot[..., 1]
        inv_real = real * cos + imag * sin
        inv_imag = imag * cos - real * sin
        x_out2 = torch.stack([inv_real, inv_imag], dim=-1).flatten(3)
        
        if x_pass is not None and x_pass.numel() > 0:
            return torch.cat([x_out2, x_pass], dim=-1)
        return x_out2



    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with causal mask.

        Args:
            q: [B, H, K, D] or [B, H, T, D] query
            k: [B, H, T, D] full key cache
            v: [B, H, T, D] full value cache
            query_positions: [K] positions of queries (for causal mask), or None for full causal

        Returns:
            [B, H, K, D] or [B, H, T, D] attention output
        """
        # Handle GQA
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # If query_positions is None, use full causal attention
        if query_positions is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            T = k.size(2)
            K = q.size(2)
            device = q.device

            # Create causal mask for specific positions
            # query at position query_positions[i] can attend to keys at positions <= query_positions[i]
            key_positions = torch.arange(T, device=device)  # [T]
            # Broadcast: [1, T] <= [K, 1] → [K, T]
            causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)  # [K, T]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, K, T]

            # Use scaled_dot_product_attention with mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False
            )

        return attn_output

    @torch.no_grad()
    def recompute_cacheblend(
        self,
        kv_data: KVCacheData,
        recompute_ratio: float = 0.15,
    ) -> KVCacheData:
        """
        CacheBlend recomputation strategy for ChatGLM:
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
        
        # Get RoPE cache (ChatGLM specific)
        rope_len = T
        rope_cache = self.model.transformer.rotary_pos_emb(rope_len).to(device=device, dtype=dtype)
        
        # Get model layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'encoder'):
            layers = self.model.transformer.encoder.layers
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
                    layer, hidden_states, k_cache, v_cache, rope_cache,
                    recompute_indices=None, batch_size=batch_size, seq_len=T
                )
                
            elif layer_idx == 1:
                # Layer 1: Full KV recompute + select top positions by V diff
                attn = layer.self_attention
                normed = layer.input_layernorm(hidden_states)
                
                # Compute mixed QKV
                mixed_x_layer = attn.query_key_value(normed)
                
                # Extract K, V
                if attn.multi_query_attention:
                    _, k_new, v_new = mixed_x_layer.split([
                        self.num_heads * self.head_dim,
                        self.num_kv_heads * self.head_dim,
                        self.num_kv_heads * self.head_dim
                    ], dim=-1)
                    k_new = k_new.view(batch_size, T, self.num_kv_heads, self.head_dim)
                    v_new = v_new.view(batch_size, T, self.num_kv_heads, self.head_dim)
                else:
                    _, k_new, v_new = mixed_x_layer.split([
                        self.num_heads * self.head_dim,
                        self.num_heads * self.head_dim,
                        self.num_heads * self.head_dim
                    ], dim=-1)
                    k_new = k_new.view(batch_size, T, self.num_heads, self.head_dim)
                    v_new = v_new.view(batch_size, T, self.num_heads, self.head_dim)
                
                k_new = k_new.transpose(1, 2).contiguous()  # [B, Hk, T, D]
                v_new = v_new.transpose(1, 2).contiguous()  # [B, Hk, T, D]
                
                k_new = self._apply_rope(k_new, rope_cache)
                
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
                mixed_q = attn.query_key_value(normed)
                if attn.multi_query_attention:
                    q = mixed_q[:, :, :self.num_heads * self.head_dim]
                    q = q.view(batch_size, K, self.num_heads, self.head_dim)
                else:
                    q, _, _ = mixed_q.split([
                        self.num_heads * self.head_dim,
                        self.num_heads * self.head_dim,
                        self.num_heads * self.head_dim
                    ], dim=-1)
                    q = q.view(batch_size, K, self.num_heads, self.head_dim)
                
                q = q.transpose(1, 2).contiguous()  # [B, Hq, K, D]
                q = self._apply_rope(q, rope_cache, selected_indices)
                
                # Attention for selected positions
                attn_output = self._compute_attention(q, k_cache, v_cache, selected_indices)
                
                # Project output
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, K, self.hidden_size)
                attn_output = attn.dense(attn_output).to(dtype)
                
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
                    layer, hidden_states, k_cache, v_cache, rope_cache,
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
            attention_mask=attention_mask,
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
        
        # Split input_ids into chunks (list slicing is fast)
        ids_each = []
        offset = 0
        for L in seq_lens:
            ids_each.append(input_ids[offset:offset + int(L)])
            offset += int(L)
        
        # Calculate chunk boundaries for efficient slicing
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
        
        # ChatGLM prefix tokens: [gMASK], <sop>, <|user|>, \n
        prefix_len = len(self.tokenizer.get_prefix_tokens()) + 2
        original_prefix_ids = ids_each[0][:prefix_len]
        
        # Determine reordering
        if num_chunks <= 1:
            order = list(range(num_chunks))
        else:
            if put_higher_ratio_to_tail:
                order = sorted(range(num_chunks), key=lambda i: (ratios[i], i))
            else:
                order = sorted(range(num_chunks), key=lambda i: (-ratios[i], i))
        
        # Reorder metadata (no data copy here)
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
        
        # Prepare RoPE embeddings for GLM
        model_dtype = self.model.dtype
        rope_cache = self.model.transformer.rotary_pos_emb(total_len).to(device=device, dtype=model_dtype)
        
        # Allocate final cache (this is where actual memory is allocated)
        H_kv = self.num_kv_heads
        D = self.head_dim
        
        # Get dtype from original cache
        if isinstance(cache, DynamicCache):
            cache_dtype = cache.key_cache[0].dtype
        else:
            cache_dtype = cache[0][0].dtype
        
        V_final = [torch.empty((1, H_kv, total_len, D), device=device, dtype=cache_dtype) for _ in range(num_layers)]
        K_final = [torch.empty((1, H_kv, total_len, D), device=device, dtype=cache_dtype) for _ in range(num_layers)]
        
        max_chunk_len = max(seq_lens) if seq_lens else 0
        base_pos = torch.arange(max_chunk_len + prefix_len + 1, device=device)
        
        # Process each layer (efficient: use views during read, copy only once during write)
        for layer_idx in range(num_layers):
            K_buf = K_final[layer_idx]
            
            for chunk_idx, orig_chunk_id in enumerate(order):
                T_i = int(new_seq_lens[chunk_idx])
                off = int(offsets[chunk_idx])
                if T_i == 0:
                    continue
                
                # Get chunk view (no data copy - just pointer arithmetic)
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
                    # Copy prefix (first time data is actually copied)
                    K_final[layer_idx][:, :, :prefix_len, :].copy_(K_chunk[:, :, :prefix_len, :])
                    V_final[layer_idx][:, :, :prefix_len, :].copy_(V_chunk[:, :, :prefix_len, :])
                    
                    # Remove RoPE from K (GLM style)
                    K_removed = self._remove_rope(
                        K_chunk[:, :, prefix_len:, :], rope_cache, old_pos
                    )
                    K_buf[:, :, off:off + T_i, :].copy_(K_removed)
                    V_final[layer_idx][:, :, off:off + T_i, :].copy_(V_chunk[:, :, prefix_len:, :])
                else:
                    # Remove RoPE (GLM style)
                    K_removed = self._remove_rope(
                        K_chunk, rope_cache, old_pos
                    )
                    K_buf[:, :, off:off + T_i, :].copy_(K_removed)
                    V_final[layer_idx][:, :, off:off + T_i, :].copy_(V_chunk)
            
            # Apply new RoPE (GLM style)
            new_pos = torch.arange(prefix_len, total_len, device=device)
            K_rebased = self._apply_rope(
                K_buf[:, :, prefix_len:, :], rope_cache, position_ids=new_pos
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


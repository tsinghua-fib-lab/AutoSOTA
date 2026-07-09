"""Chunk-wise prefill for Qwen3-VL KV cache."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass

from .chunker import ChunkInfo
from .base import KVCacheData


@dataclass
class ChunkedKVCache:
    """Chunked KV caches container."""
    chunk_caches: List[DynamicCache]
    chunk_info: ChunkInfo
    prefix_cache: Optional[DynamicCache]  # KV for prefix text
    suffix_cache: Optional[DynamicCache]  # KV for suffix text (computed with full context)
    input_embeds: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    image_start_idx: int
    image_end_idx: int
    seq_len: int


@dataclass
class MultiImageChunkedKVCache:
    """Container for multi-image chunked KV caches."""
    prefix_cache: Optional[DynamicCache]              # KV for text before first image
    image_chunk_caches: List[List[DynamicCache]]      # [img_idx][chunk_idx] - chunks for each image
    inter_caches: List[Optional[DynamicCache]]        # KV for text between images
    suffix_cache: Optional[DynamicCache]              # KV for text after last image
    chunk_infos: List[ChunkInfo]                      # ChunkInfo for each image
    image_ranges: List[Tuple[int, int]]               # (start, end) for each image
    input_embeds: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    seq_len: int


class ChunkPrefiller:
    """Prefill each image chunk separately to get independent KV caches."""

    def __init__(self, model):
        self.model = model
        self.language_model = model.model.language_model
        
        config = self.language_model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.hidden_size = config.hidden_size
    
    @torch.no_grad()
    def prefill_chunks(
        self,
        input_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        chunk_info: ChunkInfo,
        image_start_idx: int,
        image_end_idx: int,
        include_text: bool = True,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> ChunkedKVCache:
        """Prefill each chunk separately and return ChunkedKVCache."""
        device = input_embeds.device
        dtype = input_embeds.dtype
        batch_size = input_embeds.shape[0]
        seq_len = input_embeds.shape[1]
        
        cos_full, sin_full = position_embeddings
        
        chunk_embeds_list = []
        chunk_cos_list = []
        chunk_sin_list = []
        chunk_valid_masks = []
        chunk_visual_masks = []
        chunk_deepstack_list = []

        for chunk_idx in range(chunk_info.num_chunks):
            chunk_indices = chunk_info.chunk_indices[chunk_idx]
            if hasattr(chunk_info, "chunk_valid_masks") and chunk_info.chunk_valid_masks:
                chunk_valid_mask = chunk_info.chunk_valid_masks[chunk_idx].to(chunk_indices.device)
            else:
                chunk_valid_mask = torch.ones_like(chunk_indices, dtype=torch.bool)

            chunk_embeds, chunk_cos, chunk_sin = self._build_padded_chunk(
                input_embeds, cos_full, sin_full, chunk_indices, chunk_valid_mask
            )

            chunk_visual_mask = self._select_visual_mask(
                visual_pos_masks,
                chunk_indices,
                chunk_valid_mask,
                image_start_idx,
                image_end_idx,
            )
            chunk_deepstack = self._select_deepstack_embeds(
                deepstack_visual_embeds,
                chunk_indices,
                chunk_valid_mask,
                image_start_idx,
                image_end_idx,
            )

            chunk_embeds_list.append(chunk_embeds)
            chunk_cos_list.append(chunk_cos)
            chunk_sin_list.append(chunk_sin)
            chunk_valid_masks.append(chunk_valid_mask)
            chunk_visual_masks.append(chunk_visual_mask)
            chunk_deepstack_list.append(chunk_deepstack)

        prefix_cache = None
        suffix_cache = None

        # 1. Prefill prefix FIRST (before image chunks, so chunks can attend to prefix)
        if include_text and image_start_idx > 0:
            prefix_embeds = input_embeds[:, :image_start_idx, :]
            prefix_cos = self._extract_positions(cos_full, torch.arange(0, image_start_idx, device=device))
            prefix_sin = self._extract_positions(sin_full, torch.arange(0, image_start_idx, device=device))
            prefix_pos_emb = (prefix_cos, prefix_sin)

            prefix_cache = self._prefill_single_chunk(
                prefix_embeds,
                prefix_pos_emb,
                batch_size,
                image_start_idx,
                device,
                dtype,
                None,
                None,
                None,
            )

        # 2. Prefill image chunks WITH prefix context
        chunk_caches = self._prefill_chunks_batched(
            chunk_embeds_list,
            chunk_cos_list,
            chunk_sin_list,
            chunk_valid_masks,
            chunk_visual_masks,
            chunk_deepstack_list,
            prefix_cache,
            device,
            dtype,
        )

        if include_text:
            # 3. Prefill suffix WITH full context (prefix + image chunks)
            # Suffix tokens must be able to attend to all previous tokens
            if image_end_idx < seq_len:
                suffix_cache = self._prefill_suffix_with_context(
                    input_embeds=input_embeds,
                    position_embeddings=position_embeddings,
                    prefix_cache=prefix_cache,
                    chunk_caches=chunk_caches,
                    chunk_info=chunk_info,
                    image_start_idx=image_start_idx,
                    image_end_idx=image_end_idx,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )

        return ChunkedKVCache(
            chunk_caches=chunk_caches,
            chunk_info=chunk_info,
            prefix_cache=prefix_cache,
            suffix_cache=suffix_cache,
            input_embeds=input_embeds,
            position_embeddings=position_embeddings,
            image_start_idx=image_start_idx,
            image_end_idx=image_end_idx,
            seq_len=seq_len,
        )

    @torch.no_grad()
    def prefill_multi_image(
        self,
        input_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        chunk_infos: List[ChunkInfo],
        image_ranges: List[Tuple[int, int]],
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> MultiImageChunkedKVCache:
        """Process multiple images sequentially with accumulated context.

        Each image chunk attends to all previous context (prefix + prior images + inter-texts).
        Text between images attends to all prior context including the preceding image.

        Args:
            input_embeds: Full sequence embeddings [batch, seq_len, hidden]
            position_embeddings: (cos, sin) for full sequence
            chunk_infos: List of ChunkInfo for each image
            image_ranges: List of (start, end) for each image
            visual_pos_masks: Visual position masks for all images (full sequence)
            deepstack_visual_embeds: Deepstack embeddings for all images (full sequence)

        Returns:
            MultiImageChunkedKVCache containing all segment caches
        """
        device = input_embeds.device
        dtype = input_embeds.dtype
        batch_size = input_embeds.shape[0]
        seq_len = input_embeds.shape[1]
        num_images = len(image_ranges)

        cos_full, sin_full = position_embeddings

        # Accumulated context cache (grows as we process each segment)
        context_cache = None
        all_chunk_caches = []  # List of lists: [img_idx][chunk_idx]
        inter_caches = []      # KV for text between images
        prefix_cache = None

        # 1. Process prefix (before first image)
        first_image_start = image_ranges[0][0]
        if first_image_start > 0:
            prefix_cache = self._prefill_text_segment(
                input_embeds, position_embeddings,
                start_idx=0, end_idx=first_image_start,
                context_cache=None,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            context_cache = prefix_cache

        # Compute cumulative visual token offsets for multi-image
        # visual_pos_masks may contain all visual tokens concatenated: [img0_tokens | img1_tokens | ...]
        visual_offsets = [0]
        for img_start, img_end in image_ranges[:-1]:
            visual_offsets.append(visual_offsets[-1] + (img_end - img_start))

        # Total visual tokens across all images
        total_visual_tokens = sum(img_end - img_start for img_start, img_end in image_ranges)

        # 2. Process each image and inter-text
        for img_idx, (img_start, img_end) in enumerate(image_ranges):
            chunk_info = chunk_infos[img_idx]
            visual_offset = visual_offsets[img_idx]

            # Build chunk data for this image
            chunk_embeds_list = []
            chunk_cos_list = []
            chunk_sin_list = []
            chunk_valid_masks = []
            chunk_visual_masks = []
            chunk_deepstack_list = []

            for chunk_idx in range(chunk_info.num_chunks):
                chunk_indices = chunk_info.chunk_indices[chunk_idx]
                if hasattr(chunk_info, "chunk_valid_masks") and chunk_info.chunk_valid_masks:
                    chunk_valid_mask = chunk_info.chunk_valid_masks[chunk_idx].to(chunk_indices.device)
                else:
                    chunk_valid_mask = torch.ones_like(chunk_indices, dtype=torch.bool)

                chunk_embeds, chunk_cos, chunk_sin = self._build_padded_chunk(
                    input_embeds, cos_full, sin_full, chunk_indices, chunk_valid_mask
                )

                # Select visual masks/deepstack for this chunk
                # These methods handle both "visual-only" and "full-sequence" tensor shapes
                chunk_visual_mask = self._select_visual_mask_multi(
                    visual_pos_masks,
                    chunk_indices,
                    chunk_valid_mask,
                    img_start,
                    img_end,
                    visual_offset,
                    total_visual_tokens,
                )
                chunk_deepstack = self._select_deepstack_embeds_multi(
                    deepstack_visual_embeds,
                    chunk_indices,
                    chunk_valid_mask,
                    img_start,
                    img_end,
                    visual_offset,
                    total_visual_tokens,
                )

                chunk_embeds_list.append(chunk_embeds)
                chunk_cos_list.append(chunk_cos)
                chunk_sin_list.append(chunk_sin)
                chunk_valid_masks.append(chunk_valid_mask)
                chunk_visual_masks.append(chunk_visual_mask)
                chunk_deepstack_list.append(chunk_deepstack)

            # 2a. Process image chunks WITH accumulated context
            chunk_caches = self._prefill_chunks_batched(
                chunk_embeds_list,
                chunk_cos_list,
                chunk_sin_list,
                chunk_valid_masks,
                chunk_visual_masks,
                chunk_deepstack_list,
                context_cache,
                device,
                dtype,
            )
            all_chunk_caches.append(chunk_caches)

            # 2b. Update context: add image chunks to accumulated cache
            context_cache = self._merge_caches(context_cache, chunk_caches)

            # 2c. Process inter-text (if not last image)
            if img_idx < num_images - 1:
                next_img_start = image_ranges[img_idx + 1][0]
                if img_end < next_img_start:
                    inter_cache = self._prefill_text_segment(
                        input_embeds, position_embeddings,
                        start_idx=img_end, end_idx=next_img_start,
                        context_cache=context_cache,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    )
                    inter_caches.append(inter_cache)
                    if inter_cache is not None:
                        context_cache = self._merge_caches(context_cache, [inter_cache])
                else:
                    inter_caches.append(None)

        # 3. Process suffix (after last image)
        last_image_end = image_ranges[-1][1]
        suffix_cache = None
        if last_image_end < seq_len:
            suffix_cache = self._prefill_text_segment(
                input_embeds, position_embeddings,
                start_idx=last_image_end, end_idx=seq_len,
                context_cache=context_cache,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return MultiImageChunkedKVCache(
            prefix_cache=prefix_cache,
            image_chunk_caches=all_chunk_caches,
            inter_caches=inter_caches,
            suffix_cache=suffix_cache,
            chunk_infos=chunk_infos,
            image_ranges=image_ranges,
            input_embeds=input_embeds,
            position_embeddings=position_embeddings,
            seq_len=seq_len,
        )

    def _prefill_chunks_batched(
        self,
        chunk_embeds_list: List[torch.Tensor],
        chunk_cos_list: List[torch.Tensor],
        chunk_sin_list: List[torch.Tensor],
        chunk_valid_masks: List[torch.Tensor],
        chunk_visual_masks: List[Optional[torch.Tensor]],
        chunk_deepstack_list: List[Optional[List[torch.Tensor]]],
        prefix_cache: Optional[DynamicCache],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[DynamicCache]:
        """Prefill all chunks with attention to prefix (if provided).

        Each image chunk can attend to:
        1. All prefix tokens (if prefix_cache provided)
        2. All previous tokens within the chunk (causal)
        """
        num_chunks = len(chunk_embeds_list)
        if num_chunks == 0:
            return []

        chunk_embeds = torch.stack(chunk_embeds_list, dim=0).to(device=device, dtype=dtype)
        chunk_cos = torch.stack(chunk_cos_list, dim=0).to(device=device, dtype=dtype)
        chunk_sin = torch.stack(chunk_sin_list, dim=0).to(device=device, dtype=dtype)
        # _build_padded_chunk returns shape [batch, chunk_size, hidden]; assume batch=1
        if chunk_embeds.dim() == 4:
            assert chunk_embeds.shape[1] == 1, "Only batch_size=1 supported in chunk prefill"
            chunk_embeds = chunk_embeds[:, 0]
        if chunk_cos.dim() == 4:
            assert chunk_cos.shape[1] == 1, "Only batch_size=1 supported in chunk prefill"
            chunk_cos = chunk_cos[:, 0]
            chunk_sin = chunk_sin[:, 0]

        chunk_valid = torch.stack(chunk_valid_masks, dim=0).to(device=device)
        chunk_pos_emb = (chunk_cos, chunk_sin)

        caches = [DynamicCache() for _ in range(num_chunks)]
        hidden_states = chunk_embeds
        chunk_size = chunk_embeds.shape[1]

        # Determine prefix length for attention mask
        prefix_len = 0
        if prefix_cache is not None:
            prefix_k, _ = prefix_cache[0]
            prefix_len = prefix_k.shape[2]

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]

            # Get prefix KV for this layer
            prefix_k = None
            prefix_v = None
            if prefix_cache is not None:
                prefix_k, prefix_v = prefix_cache[layer_idx]

            hidden_states, k_all, v_all = self._forward_layer_with_prefix(
                layer, hidden_states, chunk_pos_emb, num_chunks, chunk_size,
                chunk_valid, prefix_k, prefix_v, prefix_len
            )

            for ci in range(num_chunks):
                hidden_states_ci = hidden_states[ci:ci + 1]
                hidden_states_ci = self._apply_deepstack(
                    hidden_states_ci,
                    chunk_visual_masks[ci],
                    chunk_deepstack_list[ci],
                    layer_idx,
                )
                hidden_states[ci:ci + 1] = hidden_states_ci

                valid_idx = torch.nonzero(chunk_valid[ci], as_tuple=False).squeeze(-1)
                caches[ci].update(
                    k_all[ci:ci + 1, :, valid_idx, :],
                    v_all[ci:ci + 1, :, valid_idx, :],
                    layer_idx,
                )

        return caches

    def _forward_layer_with_prefix(
        self,
        layer,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        seq_len: int,
        valid_mask: Optional[torch.Tensor],
        prefix_k: Optional[torch.Tensor],
        prefix_v: Optional[torch.Tensor],
        prefix_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through a single layer with prefix KV for cross-attention.

        This allows image chunk tokens to attend to prefix tokens.
        """
        dtype = hidden_states.dtype
        attn = layer.self_attn
        cos, sin = position_embeddings

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

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_cache = k.to(dtype)
        v_cache = v.to(dtype)

        # Build full KV by prepending prefix
        if prefix_k is not None and prefix_v is not None:
            # prefix_k/v: [1, num_kv_heads, prefix_len, head_dim]
            # Expand to batch_size (num_chunks)
            prefix_k_expanded = prefix_k.expand(batch_size, -1, -1, -1)
            prefix_v_expanded = prefix_v.expand(batch_size, -1, -1, -1)
            full_k = torch.cat([prefix_k_expanded, k], dim=2)
            full_v = torch.cat([prefix_v_expanded, v], dim=2)
        else:
            full_k = k
            full_v = v

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            full_k_attn = full_k.repeat_interleave(num_groups, dim=1)
            full_v_attn = full_v.repeat_interleave(num_groups, dim=1)
        else:
            full_k_attn = full_k
            full_v_attn = full_v

        # Build attention mask: chunk tokens can see prefix + causal within chunk
        total_len = prefix_len + seq_len

        # Check for padding in valid_mask
        mask_has_padding = valid_mask is not None and valid_mask.dim() > 0 and (not valid_mask.all().item())

        # Build mask: [batch, 1, seq_len, total_len]
        # chunk_to_prefix: all ones (can see all prefix tokens)
        # chunk_to_chunk: lower triangular (causal within chunk) + valid mask
        chunk_to_prefix = torch.ones(batch_size, seq_len, prefix_len, device=hidden_states.device, dtype=torch.bool)
        chunk_causal = torch.tril(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
        ).unsqueeze(0).expand(batch_size, -1, -1)

        if mask_has_padding:
            valid_mask = valid_mask.to(device=hidden_states.device, dtype=torch.bool)
            # Mask out invalid positions in both query and key dimensions
            chunk_causal = chunk_causal & valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)

        attn_mask = torch.cat([chunk_to_prefix, chunk_causal], dim=2)
        attn_mask = attn_mask.unsqueeze(1)  # [batch, 1, seq_len, total_len]

        attn_output = F.scaled_dot_product_attention(
            q, full_k_attn, full_v_attn, attn_mask=attn_mask, is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = attn.o_proj(attn_output).to(dtype)

        hidden_states = hidden_states + attn_output

        residual = hidden_states
        mlp_out = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states = residual + mlp_out

        if mask_has_padding:
            hidden_states = hidden_states * valid_mask.unsqueeze(-1)

        return hidden_states, k_cache, v_cache
    
    def _extract_positions(self, pos_emb: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Extract position embeddings at given indices."""
        if pos_emb.dim() == 3:
            return pos_emb[:, indices, :]
        elif pos_emb.dim() == 4:
            return pos_emb[:, :, indices, :]
        else:
            raise ValueError(f"Unexpected pos_emb dim: {pos_emb.dim()}")

    def _build_padded_chunk(
        self,
        input_embeds: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        chunk_indices: torch.Tensor,
        chunk_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build padded embeddings and RoPE positions for a chunk."""
        chunk_size = chunk_indices.numel()
        valid_indices = chunk_indices[chunk_valid_mask]

        chunk_embeds = torch.zeros(
            input_embeds.shape[0],
            chunk_size,
            input_embeds.shape[2],
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        if valid_indices.numel() > 0:
            chunk_embeds[:, chunk_valid_mask, :] = input_embeds.index_select(1, valid_indices)

        chunk_cos = self._pad_positions(cos_full, valid_indices, chunk_valid_mask, chunk_size)
        chunk_sin = self._pad_positions(sin_full, valid_indices, chunk_valid_mask, chunk_size)

        return chunk_embeds, chunk_cos, chunk_sin

    def _pad_positions(
        self,
        pos_emb: torch.Tensor,
        valid_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        if pos_emb.dim() == 3:
            padded = torch.zeros(
                pos_emb.shape[0],
                chunk_size,
                pos_emb.shape[2],
                device=pos_emb.device,
                dtype=pos_emb.dtype,
            )
            if valid_indices.numel() > 0:
                padded[:, valid_mask, :] = pos_emb.index_select(1, valid_indices)
            return padded
        if pos_emb.dim() == 4:
            padded = torch.zeros(
                pos_emb.shape[0],
                pos_emb.shape[1],
                chunk_size,
                pos_emb.shape[3],
                device=pos_emb.device,
                dtype=pos_emb.dtype,
            )
            if valid_indices.numel() > 0:
                padded[:, :, valid_mask, :] = pos_emb.index_select(2, valid_indices)
            return padded
        raise ValueError(f"Unexpected pos_emb dim: {pos_emb.dim()}")
    
    def _prefill_suffix_with_context(
        self,
        input_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        prefix_cache: Optional[DynamicCache],
        chunk_caches: List[DynamicCache],
        chunk_info: ChunkInfo,
        image_start_idx: int,
        image_end_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DynamicCache:
        """Prefill suffix tokens with attention to prefix + image chunks.

        This is the key fix: suffix tokens need to see all previous context,
        not just themselves. We build a combined KV cache from prefix and
        image chunks, then compute suffix KV with cross-attention to that context.
        """
        seq_len = input_embeds.shape[1]
        suffix_len = seq_len - image_end_idx
        cos_full, sin_full = position_embeddings

        # Extract suffix embeddings and position embeddings
        suffix_indices = torch.arange(image_end_idx, seq_len, device=device)
        suffix_embeds = input_embeds[:, image_end_idx:, :]
        suffix_cos = self._extract_positions(cos_full, suffix_indices)
        suffix_sin = self._extract_positions(sin_full, suffix_indices)

        # Build context KV cache: [prefix][image_chunks_reordered]
        # We need to concatenate prefix and image chunk KV for attention
        context_len = image_start_idx  # prefix length
        for cache in chunk_caches:
            # Each chunk cache has shape [batch, num_kv_heads, chunk_tokens, head_dim]
            k, v = cache[0]  # Get first layer to determine length
            context_len += k.shape[2]

        # Build position embeddings for context (prefix + reordered image)
        prefix_indices = torch.arange(0, image_start_idx, device=device)
        # Note: reorder_indices already includes image_start_idx offset
        context_indices = torch.cat([prefix_indices, chunk_info.reorder_indices])
        context_cos = self._extract_positions(cos_full, context_indices)
        context_sin = self._extract_positions(sin_full, context_indices)

        suffix_cache = DynamicCache()
        hidden_states = suffix_embeds.to(dtype=dtype)

        # Pre-compute attention mask (same for all layers)
        # Context length = prefix + all image chunk tokens
        total_context_len = image_start_idx
        for cache in chunk_caches:
            k, _ = cache[0]
            total_context_len += k.shape[2]
        total_len = total_context_len + suffix_len

        # Build mask once: suffix can see all context + causally see suffix
        suffix_causal = torch.tril(
            torch.ones(suffix_len, suffix_len, device=device, dtype=torch.bool)
        )
        suffix_to_context = torch.ones(suffix_len, total_context_len, device=device, dtype=torch.bool)
        attn_mask = torch.cat([suffix_to_context, suffix_causal], dim=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, suffix_len, total_len]

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]
            attn = layer.self_attn

            # Get context KV from prefix and chunks
            context_k_parts = []
            context_v_parts = []

            if prefix_cache is not None:
                prefix_k, prefix_v = prefix_cache[layer_idx]
                context_k_parts.append(prefix_k)
                context_v_parts.append(prefix_v)

            for cache in chunk_caches:
                chunk_k, chunk_v = cache[layer_idx]
                context_k_parts.append(chunk_k)
                context_v_parts.append(chunk_v)

            # Concatenate context KV: [batch, num_kv_heads, context_len, head_dim]
            context_k = torch.cat(context_k_parts, dim=2) if context_k_parts else None
            context_v = torch.cat(context_v_parts, dim=2) if context_v_parts else None

            # Compute suffix Q, K, V
            normed = layer.input_layernorm(hidden_states)

            q = attn.q_proj(normed).view(batch_size, suffix_len, self.num_heads, self.head_dim)
            k = attn.k_proj(normed).view(batch_size, suffix_len, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(normed).view(batch_size, suffix_len, self.num_kv_heads, self.head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm") and attn.k_norm is not None:
                k = attn.k_norm(k)

            # Apply RoPE with suffix positions
            q = self._apply_rope(q, suffix_cos, suffix_sin)
            k = self._apply_rope(k, suffix_cos, suffix_sin)

            # Store suffix KV for cache
            k_cache = k.to(dtype)
            v_cache = v.to(dtype)
            suffix_cache.update(k_cache, v_cache, layer_idx)

            # For attention, combine context KV with suffix KV
            # Full KV: [context_k, suffix_k], [context_v, suffix_v]
            if context_k is not None:
                full_k = torch.cat([context_k, k], dim=2)
                full_v = torch.cat([context_v, v], dim=2)
            else:
                full_k = k
                full_v = v

            # GQA expansion for attention
            if self.num_kv_heads < self.num_heads:
                num_groups = self.num_heads // self.num_kv_heads
                full_k = full_k.repeat_interleave(num_groups, dim=1)
                full_v = full_v.repeat_interleave(num_groups, dim=1)

            # Use pre-computed attention mask
            attn_output = F.scaled_dot_product_attention(
                q, full_k, full_v, attn_mask=attn_mask, is_causal=False
            )

            attn_output = attn_output.transpose(1, 2).reshape(batch_size, suffix_len, self.hidden_size)
            attn_output = attn.o_proj(attn_output).to(dtype)

            hidden_states = hidden_states + attn_output

            residual = hidden_states
            mlp_out = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(mlp_out).to(dtype)
            hidden_states = residual + mlp_out

        return suffix_cache

    def _prefill_text_segment(
        self,
        input_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        start_idx: int,
        end_idx: int,
        context_cache: Optional[DynamicCache],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DynamicCache:
        """Prefill a text segment with attention to accumulated context.

        This is a generalized version of _prefill_suffix_with_context that handles
        any text segment (prefix, inter-text, or suffix) with cross-attention to context.

        Args:
            input_embeds: Full sequence embeddings [batch, seq_len, hidden]
            position_embeddings: (cos, sin) for full sequence
            start_idx: Start position of text segment
            end_idx: End position of text segment
            context_cache: Accumulated KV cache from previous segments (or None for prefix)
            batch_size: Batch size
            device: Device
            dtype: Data type

        Returns:
            DynamicCache for this text segment
        """
        segment_len = end_idx - start_idx
        if segment_len <= 0:
            return None

        cos_full, sin_full = position_embeddings

        # Extract embeddings and position embeddings for this segment
        segment_indices = torch.arange(start_idx, end_idx, device=device)
        segment_embeds = input_embeds[:, start_idx:end_idx, :]
        segment_cos = self._extract_positions(cos_full, segment_indices)
        segment_sin = self._extract_positions(sin_full, segment_indices)

        # If no context, just do causal self-attention
        if context_cache is None:
            return self._prefill_single_chunk(
                segment_embeds,
                (segment_cos, segment_sin),
                batch_size,
                segment_len,
                device,
                dtype,
                None,
                None,
                None,
            )

        # Otherwise, do cross-attention to context + causal within segment
        segment_cache = DynamicCache()
        hidden_states = segment_embeds.to(dtype=dtype)

        # Get context length from first layer
        context_k, _ = context_cache[0]
        context_len = context_k.shape[2]
        total_len = context_len + segment_len

        # Build mask: segment sees all context + causal within segment
        segment_causal = torch.tril(
            torch.ones(segment_len, segment_len, device=device, dtype=torch.bool)
        )
        segment_to_context = torch.ones(segment_len, context_len, device=device, dtype=torch.bool)
        attn_mask = torch.cat([segment_to_context, segment_causal], dim=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, segment_len, total_len]

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]
            attn = layer.self_attn

            # Get context KV
            context_k, context_v = context_cache[layer_idx]

            # Compute segment Q, K, V
            normed = layer.input_layernorm(hidden_states)

            q = attn.q_proj(normed).view(batch_size, segment_len, self.num_heads, self.head_dim)
            k = attn.k_proj(normed).view(batch_size, segment_len, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(normed).view(batch_size, segment_len, self.num_kv_heads, self.head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm") and attn.k_norm is not None:
                k = attn.k_norm(k)

            # Apply RoPE with segment positions
            q = self._apply_rope(q, segment_cos, segment_sin)
            k = self._apply_rope(k, segment_cos, segment_sin)

            # Store segment KV for cache
            k_cache = k.to(dtype)
            v_cache = v.to(dtype)
            segment_cache.update(k_cache, v_cache, layer_idx)

            # For attention, combine context KV with segment KV
            full_k = torch.cat([context_k, k], dim=2)
            full_v = torch.cat([context_v, v], dim=2)

            # GQA expansion for attention
            if self.num_kv_heads < self.num_heads:
                num_groups = self.num_heads // self.num_kv_heads
                full_k = full_k.repeat_interleave(num_groups, dim=1)
                full_v = full_v.repeat_interleave(num_groups, dim=1)

            attn_output = F.scaled_dot_product_attention(
                q, full_k, full_v, attn_mask=attn_mask, is_causal=False
            )

            attn_output = attn_output.transpose(1, 2).reshape(batch_size, segment_len, self.hidden_size)
            attn_output = attn.o_proj(attn_output).to(dtype)

            hidden_states = hidden_states + attn_output

            residual = hidden_states
            mlp_out = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(mlp_out).to(dtype)
            hidden_states = residual + mlp_out

        return segment_cache

    def _merge_caches(
        self,
        context_cache: Optional[DynamicCache],
        new_caches: List[DynamicCache],
    ) -> DynamicCache:
        """Merge new caches into accumulated context by concatenating KV.

        Args:
            context_cache: Existing accumulated context (or None)
            new_caches: List of new caches to append

        Returns:
            Merged DynamicCache with all KV concatenated along sequence dimension
        """
        if not new_caches:
            return context_cache

        # Filter out None caches
        valid_caches = [c for c in new_caches if c is not None]
        if not valid_caches:
            return context_cache

        if context_cache is None:
            # Just merge new_caches together
            if len(valid_caches) == 1:
                return valid_caches[0]

            merged = DynamicCache()
            first_cache = valid_caches[0]
            num_layers = len(first_cache)

            for layer_idx in range(num_layers):
                k_parts = []
                v_parts = []
                for cache in valid_caches:
                    k, v = cache[layer_idx]
                    k_parts.append(k)
                    v_parts.append(v)
                k_concat = torch.cat(k_parts, dim=2)
                v_concat = torch.cat(v_parts, dim=2)
                merged.update(k_concat, v_concat, layer_idx)

            return merged

        # Merge context_cache with new_caches
        merged = DynamicCache()
        num_layers = len(context_cache)

        for layer_idx in range(num_layers):
            k_parts = []
            v_parts = []

            # Start with context
            ctx_k, ctx_v = context_cache[layer_idx]
            k_parts.append(ctx_k)
            v_parts.append(ctx_v)

            # Add new caches
            for cache in valid_caches:
                k, v = cache[layer_idx]
                k_parts.append(k)
                v_parts.append(v)

            k_concat = torch.cat(k_parts, dim=2)
            v_concat = torch.cat(v_parts, dim=2)
            merged.update(k_concat, v_concat, layer_idx)

        return merged

    def _prefill_single_chunk(
        self,
        chunk_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        chunk_len: int,
        device: torch.device,
        dtype: torch.dtype,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        valid_mask: Optional[torch.Tensor],
    ) -> DynamicCache:
        """Forward pass through all layers for a single chunk."""
        hidden_states = chunk_embeds.to(dtype=dtype)
        cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            layer = self.language_model.layers[layer_idx]
            hidden_states, k, v = self._forward_layer_with_cache(
                layer,
                hidden_states,
                position_embeddings,
                batch_size,
                chunk_len,
                valid_mask,
            )

            hidden_states = self._apply_deepstack(
                hidden_states, visual_pos_masks, deepstack_visual_embeds, layer_idx
            )
            cache.update(k, v, layer_idx)

        return cache

    def _apply_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        layer_idx: int,
    ) -> torch.Tensor:
        if deepstack_visual_embeds is None or visual_pos_masks is None:
            return hidden_states
        if not hasattr(self.language_model, "_deepstack_process"):
            return hidden_states
        if layer_idx >= len(deepstack_visual_embeds):
            return hidden_states
        if visual_pos_masks.shape[1] != hidden_states.shape[1]:
            raise ValueError("visual_pos_masks length does not match hidden_states.")
        return self.language_model._deepstack_process(
            hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
        )

    def _select_visual_mask(
        self,
        visual_pos_masks: Optional[torch.Tensor],
        chunk_indices: torch.Tensor,
        chunk_valid_mask: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
    ) -> Optional[torch.Tensor]:
        if visual_pos_masks is None:
            return None
        if visual_pos_masks.dim() != 2:
            return None
        chunk_size = chunk_indices.numel()
        seq_len = visual_pos_masks.shape[1]
        num_image_tokens = image_end_idx - image_start_idx
        padded_mask = torch.zeros(
            visual_pos_masks.shape[0],
            chunk_size,
            device=visual_pos_masks.device,
            dtype=visual_pos_masks.dtype,
        )
        valid_indices = chunk_indices[chunk_valid_mask]
        if seq_len == num_image_tokens:
            local_indices = valid_indices - image_start_idx
            padded_mask[:, chunk_valid_mask] = visual_pos_masks.index_select(1, local_indices)
        else:
            padded_mask[:, chunk_valid_mask] = visual_pos_masks.index_select(1, valid_indices)
        return padded_mask

    def _select_deepstack_embeds(
        self,
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        chunk_indices: torch.Tensor,
        chunk_valid_mask: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
    ) -> Optional[List[torch.Tensor]]:
        if deepstack_visual_embeds is None:
            return None
        num_image_tokens = image_end_idx - image_start_idx
        valid_indices = chunk_indices[chunk_valid_mask]
        local_indices = valid_indices - image_start_idx
        chunked = []
        for embeds in deepstack_visual_embeds:
            if embeds.dim() == 3:
                seq_len = embeds.shape[1]
                if seq_len == num_image_tokens:
                    chunked.append(embeds.index_select(1, local_indices))
                else:
                    chunked.append(embeds.index_select(1, valid_indices))
            elif embeds.dim() == 2:
                seq_len = embeds.shape[0]
                if seq_len == num_image_tokens:
                    chunked.append(embeds.index_select(0, local_indices))
                else:
                    chunked.append(embeds.index_select(0, valid_indices))
            else:
                chunked.append(embeds)
        return chunked

    def _select_visual_mask_multi(
        self,
        visual_pos_masks: Optional[torch.Tensor],
        chunk_indices: torch.Tensor,
        chunk_valid_mask: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        visual_offset: int,
        total_visual_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Select visual mask for a chunk in multi-image setting.

        Handles two possible shapes for visual_pos_masks:
        1. [batch, total_visual_tokens] - only visual tokens concatenated
        2. [batch, seq_len] - full sequence with True at visual positions

        Args:
            visual_offset: Where this image's tokens start in concatenated visual tokens
            total_visual_tokens: Total number of visual tokens across all images
        """
        if visual_pos_masks is None:
            return None
        if visual_pos_masks.dim() != 2:
            return None

        chunk_size = chunk_indices.numel()
        seq_len = visual_pos_masks.shape[1]
        num_image_tokens = image_end_idx - image_start_idx

        padded_mask = torch.zeros(
            visual_pos_masks.shape[0],
            chunk_size,
            device=visual_pos_masks.device,
            dtype=visual_pos_masks.dtype,
        )

        valid_indices = chunk_indices[chunk_valid_mask]
        local_indices = valid_indices - image_start_idx

        # Determine which indexing to use based on visual_pos_masks shape
        if seq_len == total_visual_tokens:
            # visual_pos_masks contains only visual tokens (concatenated)
            # Use visual_offset to index correctly
            visual_indices = visual_offset + local_indices
            padded_mask[:, chunk_valid_mask] = visual_pos_masks.index_select(1, visual_indices)
        else:
            # visual_pos_masks is full sequence length
            # Use valid_indices directly (sequence positions)
            padded_mask[:, chunk_valid_mask] = visual_pos_masks.index_select(1, valid_indices)

        return padded_mask

    def _select_deepstack_embeds_multi(
        self,
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        chunk_indices: torch.Tensor,
        chunk_valid_mask: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        visual_offset: int,
        total_visual_tokens: int,
    ) -> Optional[List[torch.Tensor]]:
        """Select deepstack embeddings for a chunk in multi-image setting.

        Handles two possible shapes for deepstack_visual_embeds tensors:
        1. [batch, total_visual_tokens, hidden] - only visual tokens concatenated
        2. [batch, seq_len, hidden] - full sequence

        Args:
            visual_offset: Where this image's tokens start in concatenated visual tokens
            total_visual_tokens: Total number of visual tokens across all images
        """
        if deepstack_visual_embeds is None:
            return None

        valid_indices = chunk_indices[chunk_valid_mask]
        local_indices = valid_indices - image_start_idx

        chunked = []
        for embeds in deepstack_visual_embeds:
            if embeds.dim() == 3:
                # [batch, seq_or_visual, hidden]
                embed_seq_len = embeds.shape[1]
                if embed_seq_len == total_visual_tokens:
                    # Contains only visual tokens - use visual_offset
                    visual_indices = visual_offset + local_indices
                    chunked.append(embeds.index_select(1, visual_indices))
                else:
                    # Full sequence - use valid_indices directly
                    chunked.append(embeds.index_select(1, valid_indices))
            elif embeds.dim() == 2:
                # [seq_or_visual, hidden]
                embed_seq_len = embeds.shape[0]
                if embed_seq_len == total_visual_tokens:
                    visual_indices = visual_offset + local_indices
                    chunked.append(embeds.index_select(0, visual_indices))
                else:
                    chunked.append(embeds.index_select(0, valid_indices))
            else:
                chunked.append(embeds)
        return chunked

    def _forward_layer_with_cache(
        self,
        layer,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        seq_len: int,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through a single layer, return (hidden_states, K, V)."""
        dtype = hidden_states.dtype
        attn = layer.self_attn
        cos, sin = position_embeddings

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

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        k_cache = k.to(dtype)
        v_cache = v.to(dtype)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k_attn = k.repeat_interleave(num_groups, dim=1)
            v_attn = v.repeat_interleave(num_groups, dim=1)
        else:
            k_attn = k
            v_attn = v

        mask_has_padding = valid_mask is not None and (not valid_mask.all().item())
        if mask_has_padding:
            valid_mask = valid_mask.to(device=hidden_states.device, dtype=torch.bool)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
            )  # [T, T]
            # attn_mask shape: [B, 1, T, T] so it can broadcast over heads
            attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn_mask = attn_mask & valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            attn_mask = attn_mask.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=attn_mask, is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = attn.o_proj(attn_output).to(dtype)

        hidden_states = hidden_states + attn_output

        residual = hidden_states
        mlp_out = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(mlp_out).to(dtype)
        hidden_states = residual + mlp_out

        if mask_has_padding:
            hidden_states = hidden_states * valid_mask.unsqueeze(-1)

        return hidden_states, k_cache, v_cache
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class KVCacheConcatenator:
    """Concatenate chunked KV caches into a single cache.

    Single-image order: [prefix] [chunk0] [chunk1] ... [chunkN] [suffix]
    Multi-image order:  [prefix] [img1_chunks] [inter1] [img2_chunks] [inter2] ... [imgN_chunks] [suffix]
    """

    def concatenate_multi_image(self, multi_cache: MultiImageChunkedKVCache) -> DynamicCache:
        """Concatenate multi-image caches into a single DynamicCache.

        Args:
            multi_cache: MultiImageChunkedKVCache containing all segment caches

        Returns:
            DynamicCache with KV concatenated in order:
            [prefix][img1_chunks][inter1][img2_chunks][inter2]...[imgN_chunks][suffix]
        """
        # Get number of layers from first available cache
        num_layers = None
        if multi_cache.prefix_cache is not None:
            num_layers = len(multi_cache.prefix_cache)
        elif multi_cache.image_chunk_caches and multi_cache.image_chunk_caches[0]:
            num_layers = len(multi_cache.image_chunk_caches[0][0])
        elif multi_cache.suffix_cache is not None:
            num_layers = len(multi_cache.suffix_cache)

        if num_layers is None:
            raise ValueError("No caches found to concatenate")

        full_cache = DynamicCache()

        for layer_idx in range(num_layers):
            parts_k = []
            parts_v = []

            # 1. Prefix (if exists)
            if multi_cache.prefix_cache is not None:
                prefix_k, prefix_v = multi_cache.prefix_cache[layer_idx]
                parts_k.append(prefix_k)
                parts_v.append(prefix_v)

            # 2. For each image: chunks + inter-text
            for img_idx, chunk_caches in enumerate(multi_cache.image_chunk_caches):
                # Add image chunks
                for chunk_cache in chunk_caches:
                    k, v = chunk_cache[layer_idx]
                    parts_k.append(k)
                    parts_v.append(v)

                # Add inter-text (if exists and not last image)
                if img_idx < len(multi_cache.inter_caches):
                    inter_cache = multi_cache.inter_caches[img_idx]
                    if inter_cache is not None:
                        inter_k, inter_v = inter_cache[layer_idx]
                        parts_k.append(inter_k)
                        parts_v.append(inter_v)

            # 3. Suffix (if exists)
            if multi_cache.suffix_cache is not None:
                suffix_k, suffix_v = multi_cache.suffix_cache[layer_idx]
                parts_k.append(suffix_k)
                parts_v.append(suffix_v)

            if parts_k:
                k_concat = torch.cat(parts_k, dim=2)
                v_concat = torch.cat(parts_v, dim=2)
                full_cache.update(k_concat, v_concat, layer_idx)

        return full_cache

    def concatenate(self, chunked_cache: ChunkedKVCache) -> DynamicCache:
        """Concatenate all chunk caches in order: [prefix][chunks][suffix].

        Args:
            chunked_cache: ChunkedKVCache containing chunk caches, prefix_cache and suffix_cache

        Returns:
            DynamicCache with concatenated KV in order [prefix][chunks][suffix]
        """
        chunk_caches = chunked_cache.chunk_caches

        if len(chunk_caches) == 0:
            raise ValueError("No chunk caches to concatenate")

        first_cache = chunk_caches[0]
        num_layers = len(first_cache)
        full_cache = DynamicCache()

        for layer_idx in range(num_layers):
            parts_k = []
            parts_v = []

            # 1. Prefix (if exists)
            if chunked_cache.prefix_cache is not None:
                prefix_k, prefix_v = chunked_cache.prefix_cache[layer_idx]
                parts_k.append(prefix_k)
                parts_v.append(prefix_v)

            # 2. Image chunks in order
            for cache in chunk_caches:
                k, v = cache[layer_idx]
                parts_k.append(k)
                parts_v.append(v)

            # 3. Suffix (if exists)
            if chunked_cache.suffix_cache is not None:
                suffix_k, suffix_v = chunked_cache.suffix_cache[layer_idx]
                parts_k.append(suffix_k)
                parts_v.append(suffix_v)

            k_concat = torch.cat(parts_k, dim=2)
            v_concat = torch.cat(parts_v, dim=2)

            full_cache.update(k_concat, v_concat, layer_idx)

        return full_cache

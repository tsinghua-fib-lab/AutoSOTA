"""Distributed KV cache extraction for sequence parallelism."""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass, field
from transformers.cache_utils import DynamicCache

from .config import DistributedConfig


@dataclass
class DistributedKVCacheData:
    """
    Container for distributed KV cache data.

    This extends the base KVCacheData with distributed-specific fields.

    Attributes:
        past_key_values: DynamicCache or tuple of K/V tensors (local partition)
        input_ids: [batch, local_seq_len] local input token IDs
        attention_mask: [batch, local_seq_len] local attention mask
        chunk_lens: Chunk lengths (int or List[int])
        global_offset: Global starting position for this partition
        global_total_len: Total sequence length across all ranks
        local_seq_len: Length of local sequence partition
    """

    past_key_values: Any  # DynamicCache or tuple of K/V tensors
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chunk_lens: Any  # int or List[int]
    global_offset: int = 0
    global_total_len: int = 0
    local_seq_len: int = 0
    gathered_full_kv: Any = None  # Pre-gathered full KV cache (DynamicCache), skips all_gather_kv

    @property
    def total_len(self) -> int:
        """Get total sequence length (local partition)."""
        if isinstance(self.chunk_lens, list):
            return sum(self.chunk_lens)
        return self.chunk_lens

    @property
    def num_layers(self) -> int:
        """Number of layers in the cache."""
        if isinstance(self.past_key_values, DynamicCache):
            return len(self.past_key_values.key_cache)
        elif isinstance(self.past_key_values, (tuple, list)):
            return len(self.past_key_values)
        return 0

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.input_ids.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensors."""
        if isinstance(self.past_key_values, DynamicCache):
            return self.past_key_values.key_cache[0].dtype
        elif isinstance(self.past_key_values, (tuple, list)):
            return self.past_key_values[0][0].dtype
        return torch.float32

    def to_device(self, device: torch.device) -> "DistributedKVCacheData":
        """Move all tensors to specified device."""
        input_ids = self.input_ids.to(device)
        attention_mask = self.attention_mask.to(device)

        if isinstance(self.past_key_values, DynamicCache):
            for layer_idx in range(len(self.past_key_values.key_cache)):
                if self.past_key_values.key_cache[layer_idx] is not None:
                    self.past_key_values.key_cache[layer_idx] = (
                        self.past_key_values.key_cache[layer_idx].to(device)
                    )
                if self.past_key_values.value_cache[layer_idx] is not None:
                    self.past_key_values.value_cache[layer_idx] = (
                        self.past_key_values.value_cache[layer_idx].to(device)
                    )
        elif isinstance(self.past_key_values, (tuple, list)):
            past_key_values = []
            for layer_kv in self.past_key_values:
                k, v = layer_kv
                past_key_values.append((k.to(device), v.to(device)))
            self.past_key_values = past_key_values

        return DistributedKVCacheData(
            past_key_values=self.past_key_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_lens=self.chunk_lens,
            global_offset=self.global_offset,
            global_total_len=self.global_total_len,
            local_seq_len=self.local_seq_len,
        )

    def clone(self) -> "DistributedKVCacheData":
        """Deep clone DistributedKVCacheData to avoid in-place modifications."""
        input_ids_clone = self.input_ids.clone()
        attention_mask_clone = self.attention_mask.clone()

        if isinstance(self.past_key_values, DynamicCache):
            cloned_cache = DynamicCache()
            for layer_idx in range(len(self.past_key_values.key_cache)):
                if self.past_key_values.key_cache[layer_idx] is not None:
                    cloned_cache.key_cache.append(
                        self.past_key_values.key_cache[layer_idx].clone()
                    )
                else:
                    cloned_cache.key_cache.append(None)

                if self.past_key_values.value_cache[layer_idx] is not None:
                    cloned_cache.value_cache.append(
                        self.past_key_values.value_cache[layer_idx].clone()
                    )
                else:
                    cloned_cache.value_cache.append(None)
            past_key_values_clone = cloned_cache
        elif isinstance(self.past_key_values, (tuple, list)):
            past_key_values_clone = []
            for layer_kv in self.past_key_values:
                k, v = layer_kv
                past_key_values_clone.append((k.clone(), v.clone()))
            past_key_values_clone = tuple(past_key_values_clone)
        else:
            raise TypeError(f"Unsupported past_key_values type: {type(self.past_key_values)}")

        if isinstance(self.chunk_lens, list):
            chunk_lens_clone = self.chunk_lens.copy()
        else:
            chunk_lens_clone = self.chunk_lens

        return DistributedKVCacheData(
            past_key_values=past_key_values_clone,
            input_ids=input_ids_clone,
            attention_mask=attention_mask_clone,
            chunk_lens=chunk_lens_clone,
            global_offset=self.global_offset,
            global_total_len=self.global_total_len,
            local_seq_len=self.local_seq_len,
        )


class DistributedExtractor:
    """
    Distributed KV cache extraction with sequence parallelism.

    Each GPU extracts KV cache for its partition of the sequence,
    enabling parallel processing without communication during extraction.

    Example:
        extractor = DistributedExtractor(base_extractor, config)
        local_kv = extractor.extract_distributed(context_ids)
    """

    def __init__(self, base_extractor, config: DistributedConfig):
        """
        Args:
            base_extractor: KVCacheExtractor instance for the model
            config: DistributedConfig with process info
        """
        self.base = base_extractor
        self.config = config
        self.device = base_extractor.device
        self.num_layers = base_extractor.num_layers
        self.num_kv_heads = base_extractor.num_kv_heads
        self.kv_head_dim = base_extractor.kv_head_dim

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
    ) -> torch.Tensor:
        """Apply RoPE to tensor x."""
        if position_ids is not None:
            if position_ids.dim() == 1:
                cos = cos.index_select(1, position_ids)
                sin = sin.index_select(1, position_ids)
            elif position_ids.dim() == 2:
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)
                cos = torch.gather(cos, 1, idx)
                sin = torch.gather(sin, 1, idx)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed

    def _remove_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Remove RoPE from tensor x."""
        if position_ids is not None:
            if position_ids.dim() == 1:
                cos = cos.index_select(1, position_ids)
                sin = sin.index_select(1, position_ids)
            elif position_ids.dim() == 2:
                B, K, D = position_ids.size(0), position_ids.size(1), cos.size(-1)
                idx = position_ids.unsqueeze(-1).expand(B, K, D)
                cos = torch.gather(cos, 1, idx)
                sin = torch.gather(sin, 1, idx)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x = (x * cos) - (self._rotate_half(x) * sin)
        return x

    @torch.no_grad()
    def extract_distributed(
        self,
        context_ids: torch.Tensor,
        apply_global_rope: bool = True,
    ) -> DistributedKVCacheData:
        """
        Extract KV cache for this GPU's partition of the sequence.

        Each GPU processes only its local chunk, enabling parallel extraction
        without any inter-GPU communication.

        Args:
            context_ids: Full context token IDs [1, T]
            apply_global_rope: If True, correct RoPE positions to global indices

        Returns:
            DistributedKVCacheData with local partition and global position info
        """
        device = self.device
        T = context_ids.shape[1]

        # Set up sequence partitioning
        self.config.set_sequence_partition(T)
        start = self.config.local_seq_start
        end = self.config.local_seq_end
        local_len = end - start

        # Extract only local chunk tokens
        local_ids = context_ids[:, start:end].to(device)

        # Use base extractor's method to extract KV
        # For the first rank, we need to handle the prefix token
        model = self.base.model
        tokenizer = self.base.tokenizer
        decoder = self.base.decoder

        # Get prefix tokens for model
        prefix_token_ids = self.base._get_prefix_tokens()
        prefix_len = len(prefix_token_ids)

        # Prepare input with prefix
        if self.config.rank == 0:
            # First rank: include prefix
            prefix_tensor = torch.tensor(
                [prefix_token_ids], dtype=torch.long, device=device
            )
            input_ids = torch.cat([prefix_tensor, local_ids], dim=1)
            actual_local_len = local_len + prefix_len
        else:
            # Other ranks: add prefix for attention context but will be stripped
            prefix_tensor = torch.tensor(
                [prefix_token_ids], dtype=torch.long, device=device
            )
            input_ids = torch.cat([prefix_tensor, local_ids], dim=1)
            actual_local_len = local_len

        attention_mask = torch.ones_like(input_ids)

        # Build position_ids for the forward pass.
        # Rank 0: default contiguous [0, prefix_len + local_len)
        # Rank > 0: prefix gets [0..prefix_len), local tokens get global positions
        #   [global_offset..global_offset+local_len) so keys have correct RoPE
        #   from the start, eliminating the _correct_rope_to_global() loop.
        if apply_global_rope and self.config.rank > 0:
            global_offset = start + prefix_len
            position_ids = torch.cat([
                torch.arange(prefix_len, device=device),
                torch.arange(global_offset, global_offset + local_len, device=device),
            ]).unsqueeze(0)
        else:
            position_ids = None  # Use default contiguous positions

        # Forward pass to get KV cache
        fwd_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids

        outputs = decoder(**fwd_kwargs)

        past = outputs.past_key_values

        # Extract KV tensors
        kv_cache = DynamicCache()
        kv_cache.key_cache = []
        kv_cache.value_cache = []

        for layer_idx in range(self.num_layers):
            if isinstance(past, DynamicCache):
                k = past.key_cache[layer_idx]
                v = past.value_cache[layer_idx]
            else:
                k, v = past[layer_idx]

            if self.config.rank == 0:
                # First rank: keep all including prefix
                kv_cache.key_cache.append(k.clone())
                kv_cache.value_cache.append(v.clone())
            else:
                # Other ranks: strip prefix
                kv_cache.key_cache.append(k[:, :, prefix_len:, :].clone())
                kv_cache.value_cache.append(v[:, :, prefix_len:, :].clone())

        # RoPE correction: when position_ids were passed to the forward pass
        # (apply_global_rope=True, rank > 0), keys already have correct global
        # RoPE — skip the expensive remove+reapply loop.
        # Only fall back to _correct_rope_to_global if position_ids were NOT used.
        if apply_global_rope and self.config.rank > 0 and position_ids is None:
            kv_cache = self._correct_rope_to_global(
                kv_cache, local_len, start, prefix_len
            )

        # Prepare output
        # global_offset is the starting position for this rank's KV cache
        # Rank 0: starts at position 0 (includes prefix at position 0)
        # Rank > 0: starts at (partition_start + prefix_len) to account for prefix
        if self.config.rank == 0:
            output_ids = input_ids
            output_len = actual_local_len
            global_offset = 0
        else:
            output_ids = local_ids
            output_len = local_len
            global_offset = start + prefix_len  # Account for prefix that rank 0 added

        # Total length includes prefix
        total_len_with_prefix = T + prefix_len

        return DistributedKVCacheData(
            past_key_values=kv_cache,
            input_ids=output_ids,
            attention_mask=torch.ones_like(output_ids),
            chunk_lens=output_len,
            global_offset=global_offset,
            global_total_len=total_len_with_prefix,
            local_seq_len=output_len,
        )

    def _correct_rope_to_global(
        self,
        kv_cache: DynamicCache,
        local_len: int,
        global_start: int,
        prefix_len: int,
    ) -> DynamicCache:
        """
        Correct RoPE positions from local to global indices.

        Args:
            kv_cache: DynamicCache with local RoPE
            local_len: Local sequence length
            global_start: Global starting position for this partition
            prefix_len: Length of prefix tokens

        Returns:
            DynamicCache with globally-corrected RoPE
        """
        device = self.device
        model = self.base.model

        # Global positions must account for prefix that rank 0 added
        # Rank 0 uses positions [0, prefix_len + local_len), with position 0 being prefix
        # Rank 1+ must start at (global_start + prefix_len) to avoid position collision
        # Example: prefix_len=1, T=1000, 4 GPUs
        #   Rank 0: positions [0, 251) for prefix + context[0:250]
        #   Rank 1: positions [251, 501) for context[250:500]
        #   etc.
        global_offset = global_start + prefix_len

        # Get RoPE embeddings for full sequence
        max_pos = global_offset + local_len + 1
        position_ids = torch.arange(max_pos, device=device).unsqueeze(0)

        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            cos_full, sin_full = model.model.rotary_emb(
                x=torch.empty(1, 1, 1, 1, device=device, dtype=model.dtype),
                position_ids=position_ids,
            )
        else:
            raise ValueError("Cannot find rotary_emb for model")

        # Local positions (what was used during extraction)
        # Non-rank-0 extraction adds prefix for attention but strips it from output
        # So positions during extraction were [prefix_len, prefix_len + local_len)
        local_positions = torch.arange(prefix_len, prefix_len + local_len, device=device)

        # Global positions (what should be used) - offset by prefix_len
        global_positions = torch.arange(global_offset, global_offset + local_len, device=device)

        for layer_idx in range(self.num_layers):
            k = kv_cache.key_cache[layer_idx]

            # Remove local RoPE
            k_removed = self._remove_rope(k, cos_full, sin_full, local_positions)

            # Apply global RoPE
            k_global = self._apply_rope(k_removed, cos_full, sin_full, global_positions)

            kv_cache.key_cache[layer_idx] = k_global

        return kv_cache

    @torch.no_grad()
    def extract_distributed_with_ring_attention(
        self,
        context_ids: torch.Tensor,
        process_group=None,
    ) -> DistributedKVCacheData:
        """
        Extract KV cache using ring attention for proper cross-GPU attention.

        Unlike extract_distributed(), this method uses ring attention during
        the forward pass, ensuring each token properly attends to ALL previous
        tokens across all GPUs. This fixes the quality degradation caused by
        local-only attention during extraction.

        IMPORTANT: The caller must have already called substitute_hf_flash_attn()
        to set up ring attention BEFORE calling this method. This method only
        updates the ring attention parameters for each extraction.

        Args:
            context_ids: Full context token IDs [1, T]
            process_group: Distributed process group for ring attention

        Returns:
            DistributedKVCacheData with local partition and global position info
        """
        try:
            from ring_flash_attn import update_ring_flash_attn_params
        except ImportError:
            raise ImportError(
                "ring-flash-attn not installed. Install with: pip install ring-flash-attn"
            )

        device = self.device
        T = context_ids.shape[1]
        world_size = self.config.world_size
        rank = self.config.rank

        # Get prefix tokens for model
        prefix_token_ids = self.base._get_prefix_tokens()
        prefix_len = len(prefix_token_ids)

        # Add prefix to full context
        prefix_tensor = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
        full_ids = torch.cat([prefix_tensor, context_ids.to(device)], dim=1)
        total_len = full_ids.shape[1]  # T + prefix_len

        # Pad to be divisible by world_size (required by ring attention)
        if total_len % world_size != 0:
            pad_len = world_size - (total_len % world_size)
            tokenizer = self.base.tokenizer
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            full_ids = torch.cat([
                full_ids,
                torch.full((1, pad_len), pad_id, device=device, dtype=full_ids.dtype)
            ], dim=1)
            padded_total_len = total_len + pad_len
        else:
            padded_total_len = total_len
            pad_len = 0

        # Setup ring attention parameters (assumes substitute_hf_flash_attn was already called)
        if process_group is None:
            process_group = self.config.process_group

        cu_seqlens = torch.tensor([0, padded_total_len], device=device, dtype=torch.int32)
        update_ring_flash_attn_params(cu_seqlens, process_group)

        # Chunk input across GPUs (using padded length)
        chunk_size = padded_total_len // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size

        input_ids_chunk = full_ids[:, start_idx:end_idx]
        position_ids = torch.arange(padded_total_len, device=device).unsqueeze(0)
        position_ids_chunk = position_ids[:, start_idx:end_idx]

        attention_mask = torch.ones_like(input_ids_chunk)

        # Forward pass with ring attention
        model = self.base.model
        decoder = self.base.decoder

        outputs = decoder(
            input_ids=input_ids_chunk,
            position_ids=position_ids_chunk,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        past = outputs.past_key_values

        # Extract KV tensors (already have correct RoPE from ring attention)
        kv_cache = DynamicCache()
        kv_cache.key_cache = []
        kv_cache.value_cache = []

        # Calculate actual local length (excluding padding)
        # For last rank, may need to trim padding
        if rank == world_size - 1 and pad_len > 0:
            actual_chunk_len = chunk_size - pad_len
        else:
            actual_chunk_len = chunk_size

        for layer_idx in range(self.num_layers):
            if isinstance(past, DynamicCache):
                k = past.key_cache[layer_idx]
                v = past.value_cache[layer_idx]
            else:
                k, v = past[layer_idx]

            # Trim padding if needed
            if actual_chunk_len < chunk_size:
                k = k[:, :, :actual_chunk_len, :].clone()
                v = v[:, :, :actual_chunk_len, :].clone()
            else:
                k = k.clone()
                v = v.clone()

            kv_cache.key_cache.append(k)
            kv_cache.value_cache.append(v)

        # Calculate global offset and lengths
        # Rank 0 starts at position 0 (includes prefix)
        # Other ranks start at their chunk position
        global_offset = start_idx
        local_seq_len = actual_chunk_len

        # Output IDs for this rank (without padding)
        output_ids = full_ids[:, start_idx:start_idx + actual_chunk_len]

        return DistributedKVCacheData(
            past_key_values=kv_cache,
            input_ids=output_ids,
            attention_mask=torch.ones_like(output_ids),
            chunk_lens=local_seq_len,
            global_offset=global_offset,
            global_total_len=total_len,  # Original length without padding
            local_seq_len=local_seq_len,
        )

    @torch.no_grad()
    def gather_full_kv(
        self,
        local_kv: DistributedKVCacheData,
    ) -> DynamicCache:
        """
        All-gather KV cache from all GPUs to reconstruct full cache.

        This is used after recomputation to prepare for generation.

        Args:
            local_kv: Local KV cache partition

        Returns:
            DynamicCache containing full KV cache (same on all ranks)
        """
        if not self.config.enabled or self.config.world_size == 1:
            return local_kv.past_key_values

        device = self.device
        world_size = self.config.world_size
        total_len = local_kv.global_total_len

        # Handle prefix for rank 0
        prefix_len = len(self.base._get_prefix_tokens()) if self.config.rank == 0 else 0

        # Gather sequence lengths from all ranks
        local_len = torch.tensor([local_kv.local_seq_len], device=device)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len, group=self.config.process_group)
        all_lens = [int(l.item()) for l in all_lens]

        # Calculate offsets
        offsets = [0]
        for l in all_lens[:-1]:
            offsets.append(offsets[-1] + l)

        # Allocate full cache
        full_cache = DynamicCache()
        full_cache.key_cache = []
        full_cache.value_cache = []

        dtype = local_kv.dtype

        for layer_idx in range(self.num_layers):
            k_local = local_kv.past_key_values.key_cache[layer_idx]
            v_local = local_kv.past_key_values.value_cache[layer_idx]
            _, H, _, D = k_local.shape

            # Allocate full tensors
            k_full = torch.zeros(1, H, total_len, D, device=device, dtype=dtype)
            v_full = torch.zeros(1, H, total_len, D, device=device, dtype=dtype)

            # All-gather K and V
            for rank in range(world_size):
                rank_len = all_lens[rank]
                rank_offset = offsets[rank]

                if rank == self.config.rank:
                    k_full[:, :, rank_offset : rank_offset + rank_len, :] = k_local
                    v_full[:, :, rank_offset : rank_offset + rank_len, :] = v_local

            # Broadcast each rank's portion
            for rank in range(world_size):
                rank_len = all_lens[rank]
                rank_offset = offsets[rank]

                k_slice = k_full[:, :, rank_offset : rank_offset + rank_len, :].contiguous()
                v_slice = v_full[:, :, rank_offset : rank_offset + rank_len, :].contiguous()

                dist.broadcast(k_slice, src=rank, group=self.config.process_group)
                dist.broadcast(v_slice, src=rank, group=self.config.process_group)

                k_full[:, :, rank_offset : rank_offset + rank_len, :] = k_slice
                v_full[:, :, rank_offset : rank_offset + rank_len, :] = v_slice

            full_cache.key_cache.append(k_full)
            full_cache.value_cache.append(v_full)

        return full_cache

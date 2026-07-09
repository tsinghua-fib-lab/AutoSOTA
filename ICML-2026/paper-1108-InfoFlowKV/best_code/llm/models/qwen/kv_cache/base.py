"""Data containers for KV cache recomputation pipeline."""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from transformers.cache_utils import DynamicCache


@dataclass
class KVCacheData:
    """
    Container for extracted KV cache and associated metadata.

    Attributes:
        past_key_values: The DynamicCache or tuple containing K and V tensors
        input_ids: [batch, seq_len] original input token IDs
        attention_mask: [batch, seq_len] attention mask
        chunk_lens: Chunk lengths (int for single chunk, List[int] for multiple chunks)
        global_offset: Global starting position for distributed sequence parallelism
    """
    past_key_values: Any  # DynamicCache or tuple of K/V tensors
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chunk_lens: Any  # int or List[int]
    global_offset: int = 0  # For sequence parallelism: global position offset
    
    @property
    def total_len(self) -> int:
        """Get total sequence length."""
        if isinstance(self.chunk_lens, list):
            return sum(self.chunk_lens)
        return self.chunk_lens

    def to_device(self, device: torch.device) -> "KVCacheData":
        """Move all tensors to specified device."""
        # Move input_ids
        input_ids = self.input_ids.to(device)

        # Move attention_mask
        attention_mask = self.attention_mask.to(device)

        # Move cache
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

        return KVCacheData(
            past_key_values=self.past_key_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_lens=self.chunk_lens,
        )

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

    def clone(self) -> "KVCacheData":
        """
        Deep clone KVCacheData to avoid in-place modifications.
        
        This is important when passing kv_data to scorer.compute(), 
        which may modify past_key_values during model forward pass.
        """
        # Clone input_ids and attention_mask
        input_ids_clone = self.input_ids.clone()
        attention_mask_clone = self.attention_mask.clone()
        
        # Deep clone past_key_values
        if isinstance(self.past_key_values, DynamicCache):
            # Clone DynamicCache
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
            # Clone tuple/list of (K, V) pairs
            past_key_values_clone = []
            for layer_kv in self.past_key_values:
                k, v = layer_kv
                past_key_values_clone.append((k.clone(), v.clone()))
            past_key_values_clone = tuple(past_key_values_clone)
        else:
            raise TypeError(f"Unsupported past_key_values type: {type(self.past_key_values)}")
        
        # Clone chunk_lens (handle both int and list)
        if isinstance(self.chunk_lens, list):
            chunk_lens_clone = self.chunk_lens.copy()
        else:
            chunk_lens_clone = self.chunk_lens
        
        return KVCacheData(
            past_key_values=past_key_values_clone,
            input_ids=input_ids_clone,
            attention_mask=attention_mask_clone,
            chunk_lens=chunk_lens_clone,
        )


@dataclass
class ImportanceScores:
    """
    Container for per-position importance scores.

    Attributes:
        scores: [seq_len] tensor or numpy array of importance scores (higher = more important)
        method: Scoring method used ("norm", "entropy", "mass", "vatp")
        layer_indices: Which layers were used for scoring
    """
    scores: Any  # torch.Tensor or np.ndarray
    method: str
    layer_indices: List[int] = field(default_factory=list)

    def to_tensor(self) -> torch.Tensor:
        """Convert scores to torch tensor."""
        if isinstance(self.scores, np.ndarray):
            return torch.from_numpy(self.scores)
        return self.scores

    def to_numpy(self) -> np.ndarray:
        """Convert scores to numpy array."""
        if isinstance(self.scores, torch.Tensor):
            return self.scores.cpu().numpy()
        return self.scores

    def top_k_indices(self, k: int) -> torch.Tensor:
        """Get indices of top-k most important positions."""
        scores_tensor = self.to_tensor()
        k = min(k, scores_tensor.numel())
        _, indices = torch.topk(scores_tensor, k)
        return indices.sort().values

    def top_ratio_indices(self, ratio: float) -> torch.Tensor:
        """Get indices of top ratio% most important positions."""
        scores_tensor = self.to_tensor()
        k = max(1, int(scores_tensor.numel() * ratio))
        return self.top_k_indices(k)


@dataclass
class RecomputeConfig:
    """
    Configuration for KV cache recomputation.

    Attributes:
        recompute_ratio: Fraction of positions to recompute (0.0-1.0)
        recompute_k: Alternative: fixed number of positions (overrides ratio if set)
        method: Importance scoring method ("norm", "entropy", "mass", "vatp", "combined")
        layer_indices: Which layers to use for scoring (None = last 2 layers)
        exclude_first_tokens: Number of first tokens to exclude from recomputation
        use_vatp: Whether to include VATP in combined scoring
    """
    recompute_ratio: float = 0.15
    recompute_k: Optional[int] = None
    method: str = "norm"
    layer_indices: Optional[List[int]] = None
    exclude_first_tokens: int = 0
    use_vatp: bool = True

    def get_num_positions(self, seq_len: int) -> int:
        """Calculate number of positions to recompute."""
        if self.recompute_k is not None:
            return min(self.recompute_k, seq_len)
        return max(1, int(seq_len * self.recompute_ratio))

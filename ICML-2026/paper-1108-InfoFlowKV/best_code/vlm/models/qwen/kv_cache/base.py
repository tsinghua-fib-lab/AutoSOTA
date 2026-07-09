"""Data containers for KV cache recomputation pipeline."""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers.cache_utils import DynamicCache


@dataclass
class KVCacheData:
    """
    Container for extracted KV cache and associated metadata.

    Attributes:
        past_key_values: The DynamicCache containing K and V tensors
        position_ids: [3, batch, seq_len] for MRoPE (t, h, w components)
        position_embeddings: (cos, sin) tuple from rotary embedding
        input_embeds: [batch, seq_len, hidden_size] token embeddings
        input_ids: [batch, seq_len] original input token IDs
        seq_len: Sequence length
        image_ranges: List of (start, end) tuples for image token positions
    """
    past_key_values: DynamicCache
    position_ids: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    input_embeds: torch.Tensor
    input_ids: torch.Tensor
    seq_len: int
    image_ranges: List[Tuple[int, int]] = field(default_factory=list)

    def to_device(self, device: torch.device) -> "KVCacheData":
        """Move all tensors to specified device."""
        # Move position_ids
        position_ids = self.position_ids.to(device)

        # Move position_embeddings
        position_embeddings = (
            self.position_embeddings[0].to(device),
            self.position_embeddings[1].to(device),
        )

        # Move input_embeds
        input_embeds = self.input_embeds.to(device)

        # Move input_ids
        input_ids = self.input_ids.to(device)

        # Move cache (DynamicCache uses self.layers, each layer has .keys and .values)
        for layer in self.past_key_values.layers:
            if hasattr(layer, 'keys') and layer.keys is not None:
                layer.keys = layer.keys.to(device)
            if hasattr(layer, 'values') and layer.values is not None:
                layer.values = layer.values.to(device)

        return KVCacheData(
            past_key_values=self.past_key_values,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            input_embeds=input_embeds,
            input_ids=input_ids,
            seq_len=self.seq_len,
            image_ranges=self.image_ranges,
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in the cache."""
        return len(self.past_key_values.layers)

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.input_embeds.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensors."""
        return self.input_embeds.dtype

    def clone(self) -> "KVCacheData":
        """Deep clone to avoid in-place modifications."""
        cloned_cache = DynamicCache()
        for layer_idx in range(len(self.past_key_values)):
            k, v = self.past_key_values[layer_idx]
            cloned_cache.update(k.clone(), v.clone(), layer_idx)
        return KVCacheData(
            past_key_values=cloned_cache,
            position_ids=self.position_ids.clone() if self.position_ids is not None else None,
            position_embeddings=(
                self.position_embeddings[0].clone(),
                self.position_embeddings[1].clone(),
            ) if self.position_embeddings else None,
            input_embeds=self.input_embeds.clone() if self.input_embeds is not None else None,
            input_ids=self.input_ids.clone() if self.input_ids is not None else None,
            seq_len=self.seq_len,
            image_ranges=self.image_ranges.copy() if self.image_ranges else None,
        )


@dataclass
class ImportanceScores:
    """
    Container for per-position importance scores.

    Attributes:
        scores: [seq_len] tensor or numpy array of importance scores (higher = more important)
        method: Scoring method used ("norm", "entropy", "mass", "vatp", "combined")
        layer_indices: Which layers were used for scoring
    """
    scores: Union[torch.Tensor, np.ndarray]
    method: str
    layer_indices: List[int] = field(default_factory=list)

    def to_numpy(self) -> np.ndarray:
        """Convert scores to numpy array."""
        if isinstance(self.scores, np.ndarray):
            return self.scores
        return self.scores.cpu().numpy()

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert scores to torch tensor."""
        if isinstance(self.scores, torch.Tensor):
            return self.scores.to(device) if device else self.scores
        tensor = torch.from_numpy(self.scores)
        return tensor.to(device) if device else tensor

    @property
    def numel(self) -> int:
        """Number of elements in scores."""
        if isinstance(self.scores, torch.Tensor):
            return self.scores.numel()
        return self.scores.size

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
        strategy: Recomputation strategy ("no_recompute", "lego", "cacheblend", "guided_recompute")
        recompute_ratio: Fraction of positions to recompute (0.0-1.0)
        recompute_k: Alternative: fixed number of positions (overrides ratio if set)
        method: Importance scoring method ("norm", "entropy", "mass", "vatp")
        layer_indices: Which layers to use for scoring (None = last 2 layers)
        exclude_image_tokens: Whether to exclude image tokens from recomputation
    """
    strategy: str = "guided_recompute"
    recompute_ratio: float = 0.15
    recompute_k: Optional[int] = None
    method: str = "norm"
    layer_indices: Optional[List[int]] = None
    exclude_image_tokens: bool = False

    def get_num_positions(self, seq_len: int) -> int:
        """Calculate number of positions to recompute."""
        if self.recompute_k is not None:
            return min(self.recompute_k, seq_len)
        return max(1, int(seq_len * self.recompute_ratio))

"""Importance scoring for KV cache positions using KV cache + query approach."""

import torch
from typing import List, Optional, Tuple

from .base import KVCacheData, ImportanceScores, RecomputeConfig


class ImportanceScorer:
    """
    Compute importance scores for each token position using KV cache + query.

    This scorer uses the cache-dev approach: run a forward pass with the existing
    KV cache and query token(s) to get attention weights, then compute importance
    scores from those weights.

    Supports multiple scoring methods:
    - norm: L2 norm of attention weights (how much each key is attended to)
    - entropy: Attention distribution entropy (diversity of attention)
    - mass: Total attention sum per key position
    - vatp: Value-Aware Token Pruning (attention weighted by value norms)
    - combined: Weighted combination of entropy + norm + optionally vatp

    Usage:
        scorer = ImportanceScorer(model, method="norm")
        scores = scorer.compute(kv_data, query_input_ids=query_ids)
        indices = scorer.select_positions(scores, ratio=0.15)
    """

    def __init__(
        self,
        model,
        method: str = "norm",
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            model: The Qwen3-VL model
            method: Scoring method ("norm", "entropy", "mass", "vatp", "combined")
            layer_indices: Which layers to use for scoring (None = last 2)
        """
        self.model = model
        self.language_model = model.model.language_model
        self.method = method

        # Get model config
        config = self.language_model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.hidden_size = config.hidden_size

        # Default to last 2 layers
        if layer_indices is None:
            layer_indices = [-2, -1]
        # Convert negative indices
        self.layer_indices = [
            idx if idx >= 0 else self.num_layers + idx for idx in layer_indices
        ]

    @torch.no_grad()
    def get_attention_weights(
        self,
        kv_data: KVCacheData,
        query_input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get attention weights by running forward with KV cache + query.

        Args:
            kv_data: Extracted KV cache data
            query_input_ids: [batch, num_query_tokens] input IDs for query

        Returns:
            List of attention weight tensors per layer [batch, num_heads, num_query, seq_len]
        """
        # Clone cache to avoid in-place modifications
        cloned_kv = kv_data.clone()

        device = kv_data.device
        kv_seq_len = kv_data.seq_len
        query_len = query_input_ids.shape[1]

        # Create cache_position for proper position handling (needed for MRoPE in Qwen3-VL)
        cache_position = torch.arange(
            kv_seq_len, kv_seq_len + query_len, device=device
        )

        # Flash Attention 2 doesn't support output_attentions=True
        # Use set_attn_implementation to temporarily switch to eager
        self.language_model.set_attn_implementation("eager")

        # Run forward with output_attentions=True
        outputs = self.language_model(
            input_ids=query_input_ids,
            past_key_values=cloned_kv.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            output_attentions=True,
        )

        # Restore flash attention for extraction/recompute
        self.language_model.set_attn_implementation("flash_attention_2")

        return outputs.attentions

    @torch.no_grad()
    def compute(
        self,
        kv_data: KVCacheData,
        query_input_ids: Optional[torch.Tensor] = None,
    ) -> ImportanceScores:
        """
        Compute importance scores for all positions using KV cache + query.

        Args:
            kv_data: Extracted KV cache data
            query_input_ids: [batch, num_query_tokens] input IDs for query.
                            If None, uses the last token from kv_data.input_ids.

        Returns:
            ImportanceScores with per-position scores
        """
        # If no query provided, use last token from input_ids
        if query_input_ids is None:
            if kv_data.input_ids is None:
                raise ValueError("query_input_ids required when kv_data.input_ids is None")
            query_input_ids = kv_data.input_ids[:, -1:]

        device = kv_data.device
        seq_len = kv_data.seq_len

        # Get attention weights from forward pass
        attentions = self.get_attention_weights(kv_data, query_input_ids)

        # Accumulate scores across scoring layers
        all_scores = torch.zeros(seq_len, device=device, dtype=torch.float32)

        for layer_idx in self.layer_indices:
            if layer_idx < len(attentions):
                attn_weights = attentions[layer_idx]  # [B, H, Q, S]
                # Get scores from the KV portion (first seq_len positions)
                kv_attn = attn_weights[:, :, :, :seq_len]

                if self.method == "vatp":
                    # Get value cache for this layer
                    v_cache = kv_data.past_key_values[layer_idx][1]  # [B, H_kv, T, D]
                    layer_scores = self._compute_vatp_scores(kv_attn, v_cache)
                elif self.method == "combined":
                    v_cache = kv_data.past_key_values[layer_idx][1]
                    layer_scores = self._compute_combined_scores(kv_attn, v_cache)
                else:
                    layer_scores = self._compute_scores_from_attention(kv_attn)

                all_scores += layer_scores

        # Average across scoring layers
        all_scores /= len(self.layer_indices)

        return ImportanceScores(
            scores=all_scores,
            method=self.method,
            layer_indices=self.layer_indices,
        )

    def _compute_scores_from_attention(
        self,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance scores from pre-computed attention weights.

        Args:
            attn_weights: [B, H, Q, S] attention weights

        Returns:
            [S] importance scores
        """
        if self.method == "norm":
            # L2 norm per key position
            scores = (attn_weights ** 2).sum(dim=(0, 1, 2))
            scores = torch.sqrt(scores)
        elif self.method == "mass":
            # Sum of attention per key position
            scores = attn_weights.sum(dim=(0, 1, 2))
        elif self.method == "entropy":
            # Entropy contribution per key
            eps = 1e-10
            log_weights = torch.log(attn_weights + eps)
            scores = -(attn_weights * log_weights).sum(dim=(0, 1, 2))
        else:
            # Default to mass
            scores = attn_weights.sum(dim=(0, 1, 2))

        return scores

    def _compute_vatp_scores(
        self,
        attn_weights: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Value-Aware Token Pruning scores.

        VATP weights attention by value norms to identify tokens that
        contribute most to the output.

        Args:
            attn_weights: [B, H, Q, S] attention weights
            v_cache: [B, H_kv, S, D] value cache

        Returns:
            [S] VATP importance scores
        """
        # Compute value norms: [B, H_kv, S]
        v_norms = v_cache.norm(dim=-1)

        # Expand v_norms to match attention heads if GQA
        if v_norms.shape[1] < attn_weights.shape[1]:
            num_groups = attn_weights.shape[1] // v_norms.shape[1]
            v_norms = v_norms.repeat_interleave(num_groups, dim=1)

        # Weight attention by value norms: [B, H, Q, S]
        # attn_weights: [B, H, Q, S], v_norms: [B, H, S] -> [B, H, 1, S]
        weighted_attn = attn_weights * v_norms.unsqueeze(2)

        # Sum across batch, heads, and queries
        scores = weighted_attn.sum(dim=(0, 1, 2))

        return scores

    def _compute_combined_scores(
        self,
        attn_weights: torch.Tensor,
        v_cache: torch.Tensor,
        entropy_weight: float = 0.4,
        norm_weight: float = 0.4,
        vatp_weight: float = 0.2,
    ) -> torch.Tensor:
        """
        Compute combined importance scores.

        Weighted combination of entropy, norm, and optionally VATP scores.

        Args:
            attn_weights: [B, H, Q, S] attention weights
            v_cache: [B, H_kv, S, D] value cache
            entropy_weight: Weight for entropy scores
            norm_weight: Weight for norm scores
            vatp_weight: Weight for VATP scores

        Returns:
            [S] combined importance scores
        """
        # Compute individual scores
        # Entropy
        eps = 1e-10
        log_weights = torch.log(attn_weights + eps)
        entropy_scores = -(attn_weights * log_weights).sum(dim=(0, 1, 2))

        # Norm
        norm_scores = torch.sqrt((attn_weights ** 2).sum(dim=(0, 1, 2)))

        # VATP
        vatp_scores = self._compute_vatp_scores(attn_weights, v_cache)

        # Normalize each score type to [0, 1]
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 0:
                return (x - x_min) / (x_max - x_min)
            return torch.zeros_like(x)

        entropy_norm = normalize(entropy_scores)
        norm_norm = normalize(norm_scores)
        vatp_norm = normalize(vatp_scores)

        # Weighted combination
        combined = (
            entropy_weight * entropy_norm +
            norm_weight * norm_norm +
            vatp_weight * vatp_norm
        )

        return combined

    def select_positions(
        self,
        scores: ImportanceScores,
        config: Optional[RecomputeConfig] = None,
        ratio: Optional[float] = None,
        k: Optional[int] = None,
        image_ranges: Optional[List[Tuple[int, int]]] = None,
        exclude_image_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Select positions to recompute based on importance scores.

        Args:
            scores: Computed importance scores
            config: Optional RecomputeConfig
            ratio: Optional ratio of positions to select
            k: Optional fixed number of positions to select
            image_ranges: Optional image token ranges to exclude
            exclude_image_tokens: Whether to exclude image tokens

        Returns:
            Tensor of position indices to recompute (sorted)
        """
        score_tensor = scores.to_tensor().clone()
        seq_len = score_tensor.numel()

        # Determine exclusion settings
        if config is not None:
            exclude = config.exclude_image_tokens
        else:
            exclude = exclude_image_tokens

        # Optionally exclude image tokens
        if exclude and image_ranges:
            for start, end in image_ranges:
                score_tensor[start:end] = float("-inf")

        # Determine number of positions to select
        if config is not None:
            num_positions = config.get_num_positions(seq_len)
        elif k is not None:
            num_positions = min(k, seq_len)
        elif ratio is not None:
            num_positions = max(1, int(seq_len * ratio))
        else:
            raise ValueError("Must provide config, ratio, or k parameter")

        # Select top-k
        _, indices = torch.topk(score_tensor, num_positions)

        # Sort indices for sequential access
        return indices.sort().values

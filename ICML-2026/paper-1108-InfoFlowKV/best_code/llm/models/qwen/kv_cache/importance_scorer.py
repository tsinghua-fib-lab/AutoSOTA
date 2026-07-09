"""Importance scoring for KV cache positions.

Supports multiple scoring methods:
- norm: L2 norm of attention weights (accumulated attention per key)
- entropy: Attention distribution entropy 
- mass: Total attention sum per key position
- vatp: Value-Aware Token Pruning (attention weighted by value norms)
- combined: Weighted combination of multiple metrics
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Any
from transformers.cache_utils import DynamicCache
from .base import KVCacheData, ImportanceScores, RecomputeConfig


class ImportanceScorer:
    """
    Compute importance scores for each token position.

    Usage:
        scorer = ImportanceScorer(model, method="vatp")
        scores = scorer.compute(kv_data, attention_weights)
        indices = scorer.select_positions(scores, config)
    """

    def __init__(
        self,
        model,
        method: str = "norm",
        layer_indices: Optional[List[int]] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ):
        """
        Args:
            model: The language model
            method: Scoring method ("norm", "entropy", "mass", "vatp", "combined")
            layer_indices: Which layers to use for scoring (None = last 2 layers, e.g., [-2, -1])
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
        """
        self.model = model
        self.method = method

        # Get model config
        config = getattr(model, 'config', model)
        self.num_layers = getattr(config, "num_layers", 
                                  getattr(config, "num_hidden_layers", 28))
        self.num_heads = num_heads or getattr(config, "num_attention_heads", 32)
        self.num_kv_heads = num_kv_heads or getattr(config, "multi_query_group_num",
                                                     getattr(config, "num_key_value_heads", self.num_heads))

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
        Get attention weights by running forward pass with KV cache + query.

        Uses selective eager patching: only scoring layers compute eager attention
        (returning real weights), while all other layers use fast flash attention.
        This saves significant VRAM for long contexts.

        Args:
            kv_data: KVCacheData with context KV cache
            query_input_ids: Query tokens [B, Q]

        Returns:
            Tuple of attention tensors per layer; scoring layers have
            [B, H, Q, K+Q], others are None.
        """
        cloned_cache = DynamicCache()
        cloned_cache.key_cache = list(kv_data.past_key_values.key_cache)
        cloned_cache.value_cache = list(kv_data.past_key_values.value_cache)
        cloned_cache._seen_tokens = kv_data.total_len

        # Try selective eager patching (works when model uses flash_attention_2)
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ImportError:
            ALL_ATTENTION_FUNCTIONS = None

        patched = False
        captured_weights = {}
        early_exit_hook = None

        class _EarlyExit(Exception):
            pass

        if ALL_ATTENTION_FUNCTIONS is not None and "flash_attention_2" in ALL_ATTENTION_FUNCTIONS:
            orig_flash_fn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            scoring_layer_ids = set()
            module_to_layer = {}
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                for idx in self.layer_indices:
                    attn_id = id(self.model.model.layers[idx].self_attn)
                    scoring_layer_ids.add(attn_id)
                    module_to_layer[attn_id] = idx

            def _selective_eager(module, query, key, value, attention_mask, **kwargs):
                if id(module) in scoring_layer_ids:
                    # Use flash attention for the output (memory efficient, exact)
                    attn_output = orig_flash_fn(module, query, key, value, attention_mask, **kwargs)[0]

                    # Compute norm scores incrementally over Q chunks to avoid
                    # materializing the full [B, H, Q, K] attention matrix.
                    # L2 norm is decomposable: norm_k = sqrt(sum_q(mean_h(attn)^2))
                    scaling = kwargs.get("scaling", module.scaling)
                    num_kv_heads = key.shape[1]
                    num_heads = query.shape[1]
                    K_len = key.shape[2]

                    # Expand K for GQA once (~1.3GB for 131K context, reused across chunks)
                    if num_heads != num_kv_heads:
                        groups = num_heads // num_kv_heads
                        key_exp = key[:, :, None, :, :].expand(
                            key.shape[0], num_kv_heads, groups, *key.shape[2:]
                        ).reshape(key.shape[0], num_heads, *key.shape[2:])
                    else:
                        key_exp = key

                    norm_sq_sum = torch.zeros(K_len, device=query.device, dtype=torch.float32)
                    q_chunk_size = 512

                    for q_start in range(0, query.shape[2], q_chunk_size):
                        q_chunk = query[:, :, q_start:q_start + q_chunk_size, :]
                        C = q_chunk.shape[2]

                        chunk_scores = torch.matmul(q_chunk, key_exp.transpose(2, 3)) * scaling
                        if attention_mask is not None:
                            chunk_scores = chunk_scores + attention_mask[:, :, q_start:q_start + C, :K_len]
                        chunk_attn = torch.nn.functional.softmax(chunk_scores, dim=-1, dtype=torch.float32)
                        del chunk_scores

                        # Average over heads then accumulate squared norm over Q
                        chunk_mean = chunk_attn[0].mean(dim=0)  # [C, K]
                        norm_sq_sum += (chunk_mean ** 2).sum(dim=0)  # [K]
                        del chunk_attn, chunk_mean

                    if num_heads != num_kv_heads:
                        del key_exp

                    # Store pre-computed norm² for _compute_norm
                    captured_weights[module_to_layer[id(module)]] = ("norm_sq", norm_sq_sum)
                    return attn_output, None
                return orig_flash_fn(module, query, key, value, attention_mask, **kwargs)

            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _selective_eager
            patched = True

            # Early exit: skip layers after the last scoring layer
            max_scoring = max(self.layer_indices)
            if max_scoring + 1 < self.num_layers:
                early_exit_hook = self.model.model.layers[max_scoring + 1].register_forward_pre_hook(
                    lambda m, args: (_ for _ in ()).throw(_EarlyExit())
                )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*does not support.*output_attentions.*")
                try:
                    outputs = self.model(
                        input_ids=query_input_ids,
                        past_key_values=cloned_cache,
                        use_cache=True,
                        output_attentions=True,
                    )
                    return outputs.attentions
                except _EarlyExit:
                    attentions = [None] * self.num_layers
                    for layer_idx, weights in captured_weights.items():
                        attentions[layer_idx] = weights
                    return tuple(attentions)
        finally:
            if patched:
                ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = orig_flash_fn
            if early_exit_hook is not None:
                early_exit_hook.remove()

    @torch.no_grad()
    def compute(
        self,
        kv_data: KVCacheData,
        attention_weights: Optional[List[torch.Tensor]] = None,
        query_input_ids: Optional[torch.Tensor] = None,
        use_vatp: bool = None,
    ) -> ImportanceScores:
        """
        Compute importance scores for all positions.

        Args:
            kv_data: KVCacheData with context KV cache
            attention_weights: Optional pre-computed attention tensors [B, H, Q, K] per layer
            query_input_ids: Optional query tokens [B, Q] (required if attention_weights is None)
            use_vatp: Override to enable/disable VATP in combined scoring

        Returns:
            ImportanceScores with per-position scores
        """
        # Record original KV cache length before get_attention_weights
        original_cache_len = kv_data.past_key_values.key_cache[0].shape[2]
        
        # Get attention weights if not provided
        if attention_weights is None:
            if query_input_ids is None:
                raise ValueError("Either attention_weights or query_input_ids must be provided")
            attention_weights = self.get_attention_weights(kv_data, query_input_ids)

        # Compute scores using the appropriate method
        if self.method == "combined":
            scores = self._compute_combined(
                attention_weights,
                kv_data.past_key_values,
                use_vatp=use_vatp if use_vatp is not None else True
            )
        elif self.method == "vatp":
            scores = self._compute_vatp(attention_weights, kv_data.past_key_values)
        elif self.method == "entropy":
            scores = self._compute_entropy(attention_weights)
        elif self.method == "norm":
            scores = self._compute_norm(attention_weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Truncate scores to original cache length
        # (attention K dim may include query tokens added during get_attention_weights)
        scores_array = scores.to_numpy()
        if len(scores_array) > original_cache_len:
            scores_array = scores_array[:original_cache_len]
            scores = ImportanceScores(
                scores=scores_array,
                method=scores.method,
                layer_indices=scores.layer_indices,
            )
        
        return scores

    def _compute_combined(
        self,
        attention_weights: List[torch.Tensor],
        past_key_values: Any,
        use_vatp: bool = True,
    ) -> ImportanceScores:
        """Combine multiple metrics with weighted average."""
        # Get individual metrics
        entropy_scores = self._compute_entropy(attention_weights)
        norm_scores = self._compute_norm(attention_weights)
        
        entropy = entropy_scores.to_numpy()
        norm = norm_scores.to_numpy()
        
        # Normalize to [0, 1]
        def normalize(scores):
            min_val = scores.min()
            max_val = scores.max()
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
            return scores

        entropy_norm = normalize(entropy)
        norm_norm = normalize(norm)
        
        if use_vatp and past_key_values is not None:
            vatp_scores = self._compute_vatp(attention_weights, past_key_values)
            vatp = vatp_scores.to_numpy()
            vatp_norm = normalize(vatp)
            combined = (entropy_norm + norm_norm + vatp_norm) / 3.0
        else:
            combined = (entropy_norm + norm_norm) / 2.0

        return ImportanceScores(
            scores=combined,
            method="combined",
            layer_indices=self.layer_indices,
        )

    def _compute_vatp(
        self,
        attention_weights: List[torch.Tensor],
        past_key_values: Any,
    ) -> ImportanceScores:
        """
        Compute VATP (Value-Aware Token Pruning) score.
        
        For each layer:
        1. w[h,k] = sum_q attn[h,q,k]  (accumulated attention per head)
        2. v_norm[h,k] = ||v[h,k,:]||_2  (L2 norm of value vectors)
        3. score[k] = sum_h w[h,k] * v_norm[h,k]
        
        Final score is averaged over selected layers.
        """
        if len(self.layer_indices) == 0:
            raise ValueError("layer_indices must contain at least one layer")

        # Get dimensions from first layer
        first_layer = self.layer_indices[0]
        device = attention_weights[first_layer].device
        dtype = attention_weights[first_layer].dtype
        _, H0, Q0, K0 = attention_weights[first_layer].shape

        # Detect actual KV length from past_key_values
        # Get first layer's value to infer H_kv and actual K
        if isinstance(past_key_values, DynamicCache):
            v0 = past_key_values.value_cache[first_layer]
        elif isinstance(past_key_values, (tuple, list)):
            v0 = past_key_values[first_layer][1]
        else:
            raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")
        
        if v0.dim() == 4:
            v0 = v0[0]  # [H_kv, K, D]
        
        v0 = v0.to(device=device, dtype=torch.float32)
        H_kv = v0.size(0)
        kv_len = v0.size(1)
        
        # Use actual KV length for score initialization
        score = torch.zeros(kv_len, device=device, dtype=torch.float32)

        # Compute head-to-group mapping for GQA
        if self.num_heads % H_kv != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"H_kv ({H_kv}) for GQA"
            )
        
        r = self.num_heads // H_kv
        h2g = torch.arange(self.num_heads, device=device) // r

        # Accumulate VATP over layers
        for lid in self.layer_indices:
            # Attention: [B, H, Q, K] → [H, Q, K]
            attn = attention_weights[lid][0].float()
            
            # Sum over query dimension: [H, K]
            w = attn.sum(dim=1)

            # Get values
            if isinstance(past_key_values, DynamicCache):
                v = past_key_values.value_cache[lid]
            elif isinstance(past_key_values, (tuple, list)):
                v = past_key_values[lid][1]
            else:
                raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")

            # v shape depends on format: [B, H_kv, K, D] or [H_kv, K, D]
            if v.dim() == 4:
                v = v[0]  # Take first batch: [H_kv, K, D]
            
            v = v.to(device=device, dtype=torch.float32)

            # L2 norm per KV head and position: [H_kv, K]
            v_norm_grp = v.norm(dim=-1)

            # Expand to attention-head dimension: [H, K]
            v_norm = v_norm_grp[h2g]
            
            # Align w and v_norm to kv_len to prevent dimension mismatch
            if w.shape[1] > kv_len:
                w = w[:, :kv_len]
            if v_norm.shape[1] > kv_len:
                v_norm = v_norm[:, :kv_len]

            # Layer contribution: [K]
            contrib = (w * v_norm).sum(dim=0)
            score += contrib

        # Average over layers
        score = score / float(len(self.layer_indices))

        return ImportanceScores(
            scores=score.cpu().numpy(),
            method="vatp",
            layer_indices=self.layer_indices,
        )

    def _compute_entropy(
        self,
        attention_weights: List[torch.Tensor],
    ) -> ImportanceScores:
        """
        Compute attention entropy per position.
        
        Reference: small_model_guided implementation
        1. Average over heads: attn_mean = attn.mean(dim=0)  # [Q, K]
        2. Compute entropy: -(attn_mean * log(attn_mean)).sum(dim=0)  # [K]
        
        Higher entropy = position receives diverse attention from queries.
        """
        first_layer = self.layer_indices[0]
        device = attention_weights[first_layer].device
        _, _, _, K = attention_weights[first_layer].shape

        entropy = torch.zeros(K, device=device, dtype=torch.float32)

        for lid in self.layer_indices:
            # [B, H, Q, K] → [H, Q, K]
            attn = attention_weights[lid][0].float()
            
            # Average over heads: [Q, K]
            attn_mean = attn.mean(dim=0)
            
            # Add epsilon for numerical stability
            attn_dist = attn_mean + 1e-10
            
            # Entropy: -sum(p * log(p)) over query dimension
            layer_entropy = -(attn_dist * attn_dist.log()).sum(dim=0)  # [K]
            
            entropy += layer_entropy

        # Average over layers
        entropy = entropy / float(len(self.layer_indices))

        return ImportanceScores(
            scores=entropy.cpu().numpy(),
            method="entropy",
            layer_indices=self.layer_indices,
        )

    def _compute_norm(
        self,
        attention_weights: List[torch.Tensor],
    ) -> ImportanceScores:
        """
        Compute L2 norm of attention weights per position.

        Reference: small_model_guided implementation
        1. Average over heads: attn_mean = attn.mean(dim=0)  # [Q, K]
        2. L2 norm over query dimension: norm_k = ||attn_mean[:, k]||_2

        This measures how strongly each position is attended to across all queries.
        Supports pre-computed norm² from selective eager (chunked Q computation).
        """
        first_layer = self.layer_indices[0]
        first_weight = attention_weights[first_layer]

        # Check for pre-computed norm² from selective eager
        if isinstance(first_weight, tuple) and first_weight[0] == "norm_sq":
            _, norm_sq = first_weight
            K = norm_sq.shape[0]
            device = norm_sq.device
            norm_scores = torch.zeros(K, device=device, dtype=torch.float32)
            for lid in self.layer_indices:
                _, layer_norm_sq = attention_weights[lid]
                norm_scores += layer_norm_sq.sqrt()
            norm_scores = norm_scores / float(len(self.layer_indices))
            return ImportanceScores(
                scores=norm_scores.cpu().numpy(),
                method="norm",
                layer_indices=self.layer_indices,
            )

        # Standard path: full attention weights [B, H, Q, K]
        device = first_weight.device
        _, _, _, K = first_weight.shape

        norm_scores = torch.zeros(K, device=device, dtype=torch.float32)

        for lid in self.layer_indices:
            # [B, H, Q, K] → [H, Q, K]
            attn = attention_weights[lid][0].float()

            # Average over heads: [Q, K]
            attn_mean = attn.mean(dim=0)

            # L2 norm over query dimension for each key position: [K]
            # norm_k = ||attn_mean[:, k]||_2
            layer_norm = attn_mean.norm(p=2, dim=0)  # [K]

            norm_scores += layer_norm

        # Average over layers
        norm_scores = norm_scores / float(len(self.layer_indices))

        return ImportanceScores(
            scores=norm_scores.cpu().numpy(),
            method="norm",
            layer_indices=self.layer_indices,
        )


    def select_positions(
        self,
        scores: ImportanceScores,
        ratio: Optional[float] = None,
        k: Optional[int] = None,
        exclude_first_tokens: int = 0,
        exclude_ranges: Optional[List[Tuple[int, int]]] = None,
        descending: bool = False,
    ) -> np.ndarray:
        """
        Select positions to recompute based on importance scores.

        Args:
            scores: Computed importance scores
            ratio: Fraction of positions to select (0.0-1.0)
            k: Alternative: fixed number of positions (overrides ratio if set)
            exclude_first_tokens: Number of first tokens to exclude
            exclude_ranges: Optional ranges [(start, end), ...] to exclude
            descending: If True, sort positions in descending order (latest first)

        Returns:
            Numpy array of position indices to recompute (sorted)
        """
        score_array = scores.to_numpy().copy()
        seq_len = len(score_array)

        # Exclude first tokens if configured
        if exclude_first_tokens > 0:
            score_array[:exclude_first_tokens] = float('-inf')

        # Exclude specified ranges
        if exclude_ranges:
            for start, end in exclude_ranges:
                score_array[start:end] = float('-inf')

        # Get number of positions to select
        if k is not None:
            num_positions = min(k, seq_len)
        elif ratio is not None:
            num_positions = max(1, int(seq_len * ratio))
        else:
            raise ValueError("Either ratio or k must be provided")

        # Select top-k
        indices = np.argpartition(score_array, -num_positions)[-num_positions:]
        indices = indices[np.argsort(score_array[indices])[::-1]]

        # Sort by position index
        indices = np.sort(indices)
        if descending:
            indices = indices[::-1].copy()

        return indices

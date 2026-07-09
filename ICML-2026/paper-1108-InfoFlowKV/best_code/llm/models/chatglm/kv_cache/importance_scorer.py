"""Importance scoring for KV cache positions.

Supports multiple scoring methods:
- norm: L2 norm of attention weights (accumulated attention per key)
- entropy: Attention distribution entropy 
- mass: Total attention sum per key position
- vatp: Value-Aware Token Pruning (attention weighted by value norms)
- combined: Weighted combination of multiple metrics
"""

import math
import types
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Any
from transformers.cache_utils import DynamicCache

from .base import KVCacheData, ImportanceScores, RecomputeConfig


def patch_glm_core_attention(model):
    """
    Patch GLM CoreAttention modules to save attention weights.
    GLM uses CoreAttention for the actual attention computation in eager mode.
    """
    patched_count = 0
    for layer_idx, layer in enumerate(model.transformer.encoder.layers):
        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'core_attention'):
            core_attention = layer.self_attention.core_attention
            
            # Skip if already patched
            if hasattr(core_attention, '_original_forward'):
                continue
                
            original_forward = core_attention.forward
            core_attention._original_forward = original_forward  # Save original
            core_attention._capture_attn = False  # Default OFF to save memory

            def make_patched_forward(original_fn, module, idx):
                def patched_forward(query_layer, key_layer, value_layer, attention_mask):
                    if not getattr(module, "_capture_attn", False):
                        return original_fn(query_layer, key_layer, value_layer, attention_mask)
                    # Call original forward to get context_layer
                    context_layer = original_fn(query_layer, key_layer, value_layer, attention_mask)

                    # Compute attention weights for storage
                    batch_size = query_layer.size(0)
                    num_heads = query_layer.size(1)
                    seq_len_q = query_layer.size(2)
                    seq_len_k = key_layer.size(2)
                    head_dim = query_layer.size(3)

                    # Reshape for attention computation
                    q = query_layer.view(batch_size * num_heads, seq_len_q, head_dim)
                    k = key_layer.view(batch_size * num_heads, seq_len_k, head_dim)

                    # Compute attention scores
                    scores = torch.bmm(q, k.transpose(1, 2))
                    scores = scores / (head_dim ** 0.5)

                    # Apply mask if provided
                    if attention_mask is not None:
                        scores = scores.view(batch_size, num_heads, seq_len_q, seq_len_k)
                        scores = scores.masked_fill(attention_mask, float("-inf"))
                        scores = scores.view(batch_size * num_heads, seq_len_q, seq_len_k)

                    # Compute attention weights
                    attention_probs = torch.nn.functional.softmax(scores, dim=-1)
                    attention_probs = attention_probs.view(batch_size, num_heads, seq_len_q, seq_len_k)

                    # Store attention weights
                    module._saved_attention_weights = attention_probs.detach()

                    return context_layer
                return patched_forward

            core_attention.forward = make_patched_forward(original_forward, core_attention, layer_idx)
            patched_count += 1

    return patched_count


def set_glm_attention_capture(model, flag: bool):
    """Enable/disable attention capture on patched GLM core attention modules."""
    for layer in model.transformer.encoder.layers:
        if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "core_attention"):
            setattr(layer.self_attention.core_attention, "_capture_attn", flag)


def patch_glm_model_forward(model):
    """
    Patch GLM model's forward method to collect and return attention weights.
    """
    # Skip if already patched
    if hasattr(model, '_original_forward'):
        return
        
    original_forward = model.forward
    model._original_forward = original_forward  # Save original

    def new_forward(input_ids=None, position_ids=None, attention_mask=None,
                   past_key_values=None, inputs_embeds=None, labels=None,
                   use_cache=None, output_attentions=None, output_hidden_states=None,
                   return_dict=None, **kwargs):

        # Call original forward
        outputs = original_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # If attention was requested, collect the saved weights
        if output_attentions:
            all_attentions = []
            for layer in model.transformer.encoder.layers:
                if hasattr(layer.self_attention.core_attention, '_saved_attention_weights'):
                    attn = layer.self_attention.core_attention._saved_attention_weights
                    if attn is not None:
                        all_attentions.append(attn)
                        # Clear for next forward
                        layer.self_attention.core_attention._saved_attention_weights = None

            if all_attentions:
                outputs.attentions = tuple(all_attentions)

        return outputs

    model.forward = new_forward
    return model


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
        
        # Patch GLM model for attention capture
        self._is_patched = False
        self._patch_model()

    def _patch_model(self):
        """Patch GLM model to capture attention weights."""
        if not self._is_patched:
            # Patch CoreAttention modules
            patch_glm_core_attention(self.model)
            # Patch model forward to collect attention weights
            patch_glm_model_forward(self.model)
            self._is_patched = True

    @torch.no_grad()
    def get_attention_weights(
        self,
        kv_data: KVCacheData,
        query_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Get attention weights by running forward pass with KV cache + query.

        Args:
            kv_data: KVCacheData with context KV cache
            query_input_ids: Query tokens [B, Q]
            attention_mask: Optional attention mask [B, K+Q]

        Returns:
            List of attention tensors [B, H, Q, K] per layer
        """
        # For ChatGLM, need to handle position_ids
        cache_len = kv_data.past_key_values.key_cache[0].size(2)
        batch_size = query_input_ids.size(0)
        query_len = query_input_ids.size(1)
        
        position_ids = torch.arange(
            cache_len, cache_len + query_len,
            dtype=torch.long,
            device=query_input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Enable attention capture
        set_glm_attention_capture(self.model, True)
        
        # Run forward pass with KV cache + query to get attention weights
        # Note: attention_mask not passed to model (not used for importance scoring)
        # Clone kv_data to avoid modifying the original past_key_values
        cloned_cache = kv_data.clone()
        
        # Suppress the SDPA fallback warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not support.*output_attentions.*")
            outputs = self.model(
                input_ids=query_input_ids,
                position_ids=position_ids,
                past_key_values=cloned_cache.past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        
        # Disable attention capture
        set_glm_attention_capture(self.model, False)
        
        return outputs.attentions

    @torch.no_grad()
    def compute(
        self,
        kv_data: KVCacheData,
        attention_weights: Optional[List[torch.Tensor]] = None,
        query_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_vatp: bool = None,
    ) -> ImportanceScores:
        """
        Compute importance scores for all positions.

        Args:
            kv_data: KVCacheData with context KV cache
            attention_weights: Optional pre-computed attention tensors [B, H, Q, K] per layer
            query_input_ids: Optional query tokens [B, Q] (required if attention_weights is None)
            attention_mask: Optional attention mask [B, K+Q]
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
            attention_weights = self.get_attention_weights(kv_data, query_input_ids, attention_mask)

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
        elif self.method == "mass":
            scores = self._compute_mass(attention_weights)
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
        """
        first_layer = self.layer_indices[0]
        device = attention_weights[first_layer].device
        _, _, _, K = attention_weights[first_layer].shape

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
    ) -> np.ndarray:
        """
        Select positions to recompute based on importance scores.

        Args:
            scores: Computed importance scores
            ratio: Fraction of positions to select (0.0-1.0)
            k: Alternative: fixed number of positions (overrides ratio if set)
            exclude_first_tokens: Number of first tokens to exclude
            exclude_ranges: Optional ranges [(start, end), ...] to exclude

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

        # Sort for sequential access
        indices = np.sort(indices)

        return indices

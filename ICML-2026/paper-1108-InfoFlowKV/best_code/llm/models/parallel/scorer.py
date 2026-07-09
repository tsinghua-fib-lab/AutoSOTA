"""Distributed importance scoring with minimal communication."""

import warnings

import torch
import torch.distributed as dist
from typing import Optional, Tuple, List
from transformers.cache_utils import DynamicCache

from .config import DistributedConfig
from .extractor import DistributedKVCacheData


class DistributedScorer:
    """
    Distributed importance scoring for sequence parallel inference.

    Key optimization: Broadcasts Q (small) and gathers scores (much smaller than KV).
    This avoids the expensive all-gather of full KV tensors.

    Communication pattern:
        1. Broadcast Q from rank 0 (O(Q_len × dim))
        2. Each GPU computes local scores (no communication)
        3. All-gather scores (O(Q_len × T) - much smaller than full KV)

    Example:
        scorer = DistributedScorer(model, config)
        local_important, global_important = scorer.score_distributed(local_kv, query_ids)
    """

    def __init__(
        self,
        model,
        config: DistributedConfig,
        method: str = "norm",
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            model: The language model
            config: DistributedConfig with process info
            method: Scoring method ("norm", "entropy", "vatp", "combined")
            layer_indices: Which layers to use for scoring (default: last 2)
        """
        self.model = model
        self.config = config
        self.method = method
        self.device = next(model.parameters()).device

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

        # Default to last 2 layers
        if layer_indices is None:
            layer_indices = [-2, -1]

        self.layer_indices = [
            idx if idx >= 0 else self.num_layers + idx for idx in layer_indices
        ]

    @torch.no_grad()
    def _get_attention_weights(
        self,
        query_input_ids: torch.Tensor,
        local_kv: DistributedKVCacheData,
    ) -> tuple:
        """
        Get attention weights by running forward pass with output_attentions=True.

        In distributed mode, substitute_hf_flash_attn() replaces the flash_attention_2
        dispatch with a function that always returns None for attention weights. We
        patch it so only the scoring layers (self.layer_indices) use eager attention
        (returning real weights), while all other layers keep using fast flash attention.

        Args:
            query_input_ids: Query token IDs [B, Q_len]
            local_kv: Local KV cache partition

        Returns:
            Tuple of attention weight tensors per layer; entries for scoring layers
            are [B, H, Q, K+Q], others are None.
        """
        # Shallow copy cache lists (DynamicCache.update() reassigns list elements
        # via torch.cat, so originals are never modified in-place)
        cloned_cache = DynamicCache()
        cloned_cache.key_cache = list(local_kv.past_key_values.key_cache)
        cloned_cache.value_cache = list(local_kv.past_key_values.value_cache)
        cloned_cache._seen_tokens = local_kv.local_seq_len

        # Query tokens must have position IDs starting after the GLOBAL context,
        # not after the local cache. Otherwise RoPE encodes wrong relative
        # distances (e.g., query appears at pos 250 instead of 1001).
        device = query_input_ids.device
        global_total = getattr(local_kv, 'global_total_len', 0) or local_kv.local_seq_len
        Q_len = query_input_ids.shape[1]
        position_ids = torch.arange(
            global_total, global_total + Q_len, device=device
        ).unsqueeze(0)

        # Patch flash_attention_2 so only scoring layers use eager attention.
        # This avoids the cost of eager on all layers while getting real
        # attention weights (with correct RoPE) for the layers we score.
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ImportError:
            ALL_ATTENTION_FUNCTIONS = None

        patched = False
        captured_weights = {}  # layer_idx -> attn_weights tensor
        early_exit_hook = None

        class _EarlyExit(Exception):
            pass

        if ALL_ATTENTION_FUNCTIONS is not None and "flash_attention_2" in ALL_ATTENTION_FUNCTIONS:
            orig_flash_fn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            scoring_layer_ids = set()
            module_to_layer = {}  # id(self_attn) -> layer_idx
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                for idx in self.layer_indices:
                    attn_id = id(self.model.model.layers[idx].self_attn)
                    scoring_layer_ids.add(attn_id)
                    module_to_layer[attn_id] = idx

            def _selective_eager(module, query, key, value, attention_mask, **kwargs):
                if id(module) in scoring_layer_ids:
                    # Eager: Q @ K^T + softmax, returns real attention weights
                    key_states = key
                    value_states = value
                    if hasattr(module, 'num_key_value_groups') and module.num_key_value_groups > 1:
                        key_states = key_states[:, :, None, :, :].expand(
                            *key_states.shape[:2], module.num_key_value_groups,
                            *key_states.shape[2:]
                        ).reshape(key_states.shape[0], -1, *key_states.shape[2:])
                        value_states = value_states[:, :, None, :, :].expand(
                            *value_states.shape[:2], module.num_key_value_groups,
                            *value_states.shape[2:]
                        ).reshape(value_states.shape[0], -1, *value_states.shape[2:])
                    scaling = kwargs.get("scaling", module.scaling)
                    attn_weights = torch.matmul(
                        query, key_states.transpose(2, 3)
                    ) * scaling
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask[:, :, :, :key_states.shape[-2]]
                    attn_weights = torch.nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query.dtype)
                    captured_weights[module_to_layer[id(module)]] = attn_weights
                    attn_output = torch.matmul(attn_weights, value_states)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    return attn_output, attn_weights
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
                        position_ids=position_ids,
                        past_key_values=cloned_cache,
                        use_cache=True,
                        output_attentions=True,
                    )
                    return outputs.attentions
                except _EarlyExit:
                    # Build attentions tuple from captured weights
                    attentions = [None] * self.num_layers
                    for layer_idx, weights in captured_weights.items():
                        attentions[layer_idx] = weights
                    return tuple(attentions)
        finally:
            if patched:
                ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = orig_flash_fn
            if early_exit_hook is not None:
                early_exit_hook.remove()

    def _compute_local_scores(
        self,
        local_kv: DistributedKVCacheData,
        query_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance scores using real attention weights from the model.

        Uses output_attentions=True to get model-computed attention weights with
        correct RoPE on both Q and K. This matches the single-GPU scorer exactly.

        Args:
            local_kv: Local KV cache partition
            query_input_ids: Query token IDs [B, Q_len]

        Returns:
            Importance scores [local_seq_len] - 1D score per K position
        """
        local_T = local_kv.local_seq_len
        attention_weights = self._get_attention_weights(query_input_ids, local_kv)

        scores_accum = None

        for layer_idx in self.layer_indices:
            # Truncate to context-only: [B, H, Q, K+Q] -> [B, H, Q, K]
            attn_probs = attention_weights[layer_idx][:, :, :, :local_T].float()

            if self.method == "norm":
                attn_mean = attn_probs.mean(dim=1)[0]  # [Q, local_T]
                layer_scores = attn_mean.norm(p=2, dim=0)  # [local_T]
            elif self.method == "entropy":
                attn_mean = attn_probs.mean(dim=1)[0] + 1e-10
                layer_scores = -(attn_mean * attn_mean.log()).sum(dim=0)
            else:
                layer_scores = attn_probs.sum(dim=(0, 1, 2))

            if scores_accum is None:
                scores_accum = layer_scores
            else:
                scores_accum += layer_scores

        scores_accum = scores_accum / len(self.layer_indices)
        return scores_accum

    @torch.no_grad()
    def score_distributed(
        self,
        local_kv: DistributedKVCacheData,
        query_input_ids: torch.Tensor,
        top_k: Optional[int] = None,
        top_ratio: Optional[float] = None,
        exclude_first_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global importance scores with minimal communication.

        Communication cost analysis:
        - Broadcast Q hidden: O(Q_len × hidden_size) - small
        - All-gather scores: O(T) - much smaller than O(T × num_heads × head_dim)

        Args:
            local_kv: Local KV cache partition
            query_input_ids: Query token IDs [B, Q_len]
            top_k: Number of positions to select (overrides config)
            top_ratio: Ratio of positions to select (overrides config)
            exclude_first_tokens: Number of tokens at the start of the global
                sequence to exclude from selection (matching single-GPU
                exclude_first_tokens behavior). These positions get -inf scores.

        Returns:
            Tuple of:
                - local_important: Important positions within this GPU's partition (local indices)
                - global_important: Important positions in global indices (same on all GPUs)
        """
        device = self.device
        world_size = self.config.world_size

        # Compute local scores via forward pass (correct Q at each layer).
        # Softmax is local (over local_K), which is approximate but uses
        # properly-transformed Q. No Q broadcast needed.
        scores_local = self._compute_local_scores(local_kv, query_input_ids.to(device))  # [local_T]

        if not self.config.enabled or world_size == 1:
            # Single GPU: scores_local is global scores
            global_scores = scores_local

            # Exclude first tokens from selection
            if exclude_first_tokens > 0:
                global_scores = global_scores.clone()
                global_scores[:exclude_first_tokens] = float('-inf')

            # Select important positions
            num_pos = self._get_num_positions(
                len(global_scores), top_k, top_ratio
            )
            important_positions = self._select_top_positions(global_scores, num_pos)

            # All positions are local in single-GPU case
            return important_positions, important_positions

        # Multi-GPU: all-gather scores
        # First, gather sequence lengths to know tensor sizes
        local_len = torch.tensor([scores_local.numel()], device=device)
        all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_lens, local_len, group=self.config.process_group)
        all_lens = [int(l.item()) for l in all_lens]

        # Pad scores to max length for all_gather
        max_len = max(all_lens)
        scores_padded = torch.zeros(max_len, device=device, dtype=scores_local.dtype)
        scores_padded[: scores_local.numel()] = scores_local

        # All-gather padded scores
        all_scores_padded = [
            torch.zeros(max_len, device=device, dtype=scores_local.dtype)
            for _ in range(world_size)
        ]
        dist.all_gather(all_scores_padded, scores_padded, group=self.config.process_group)

        # Concatenate and trim to get global scores
        global_scores = torch.cat([
            all_scores_padded[r][: all_lens[r]] for r in range(world_size)
        ])

        # Calculate global offsets based on actual gathered lengths
        # This accounts for prefix tokens that rank 0 adds
        offsets = [0]
        for l in all_lens[:-1]:
            offsets.append(offsets[-1] + l)

        # Exclude first tokens from selection (matching single-GPU behavior)
        if exclude_first_tokens > 0:
            global_scores[:exclude_first_tokens] = float('-inf')

        # Select important positions (same on all GPUs due to broadcast/gather)
        total_len = global_scores.numel()
        num_pos = self._get_num_positions(total_len, top_k, top_ratio)
        important_positions = self._select_top_positions(global_scores, num_pos)

        # Ensure all ranks have identical positions. torch.topk can break ties
        # non-deterministically across GPUs even with identical inputs, causing
        # per-rank K_local to diverge → NCCL all_gather size mismatches downstream.
        dist.broadcast(important_positions, src=0, group=self.config.process_group)

        # Filter to local positions for this GPU
        # Use actual offsets, not config values (which don't account for prefix)
        local_start = offsets[self.config.rank]
        local_end = local_start + all_lens[self.config.rank]

        local_mask = (important_positions >= local_start) & (important_positions < local_end)
        local_important_global = important_positions[local_mask]

        # Convert to local indices
        local_important = local_important_global - local_start

        return local_important, important_positions

    def _get_num_positions(
        self,
        total_len: int,
        top_k: Optional[int] = None,
        top_ratio: Optional[float] = None,
    ) -> int:
        """Get number of positions to select."""
        if top_k is not None:
            return min(top_k, total_len)
        if top_ratio is not None:
            return max(1, int(total_len * top_ratio))
        return self.config.get_num_recompute_positions(total_len)

    def _select_top_positions(
        self,
        scores: torch.Tensor,
        num_positions: int,
    ) -> torch.Tensor:
        """
        Select top-k positions by importance score.

        Args:
            scores: Importance scores [seq_len]
            num_positions: Number of positions to select

        Returns:
            Sorted tensor of position indices
        """
        num_positions = min(num_positions, scores.numel())
        _, indices = torch.topk(scores, num_positions)
        return indices.sort().values

    @torch.no_grad()
    def compute_full_attention_scores(
        self,
        local_kv: DistributedKVCacheData,
        query_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full attention matrix for analysis/debugging.

        Args:
            local_kv: Local KV cache partition
            query_input_ids: Query token IDs

        Returns:
            Full attention scores [num_layers, num_heads, Q_len, local_T]
        """
        query_input_ids = query_input_ids.to(self.device)
        local_T = local_kv.local_seq_len
        attention_weights = self._get_attention_weights(query_input_ids, local_kv)

        all_attn = []
        for layer_idx in self.layer_indices:
            # Truncate to context-only: [B, H, Q, K+Q] -> [B, H, Q, K]
            attn_probs = attention_weights[layer_idx][:, :, :, :local_T]
            all_attn.append(attn_probs[0])

        return torch.stack(all_attn)  # [num_layers, num_heads, Q, K]

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import attribution as llm_attr
from .core import IFRAggregate, MultiHopIFRResult, compute_ifr_sentence_aggregate

##########################################
# Stop-token configuration (edit here)
##########################################

# Tokens to be treated as "stop tokens" and skipped (soft-deleted) from the full attribution flow.
# You can modify this list for different experiments.
STOP_TOKENS: List[str] = [",", "."]

# Treat pure-whitespace tokens as stop tokens.
SKIP_WHITESPACE: bool = True

# Match stop tokens after stripping leading/trailing whitespace.
STRIP_BEFORE_MATCH: bool = True


def is_stop_token(token: str) -> bool:
    if token is None:
        return False
    t = str(token)
    if STRIP_BEFORE_MATCH:
        t = t.strip()
    if SKIP_WHITESPACE and t == "":
        return True
    return t in STOP_TOKENS


def keep_token_indices(tokens: Sequence[str]) -> List[int]:
    return [i for i, tok in enumerate(tokens) if not is_stop_token(tok)]


def _last_attributable_generation_index(
    tokenizer: Any,
    generation_ids: torch.Tensor,
    generation_tokens: Sequence[str],
) -> int:
    end = int(generation_ids.shape[1]) - 1
    if end < 0:
        return -1

    last_id = int(generation_ids[0, end].item())
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
        if any(last_id == int(eos_id) for eos_id in eos_ids if eos_id is not None):
            return end - 1

    if end < len(generation_tokens) and is_stop_token(generation_tokens[end]):
        return end - 1

    return end


def _stop_keep_mask(tokens: Sequence[str]) -> torch.Tensor:
    keep = [0.0 if is_stop_token(tok) else 1.0 for tok in tokens]
    return torch.as_tensor(keep, dtype=torch.float32)


def _build_stop_keep_mask_full(
    *,
    prompt_len_full: int,
    gen_len: int,
    user_prompt_indices: Sequence[int],
    user_prompt_tokens: Sequence[str],
    generation_tokens: Sequence[str],
) -> torch.Tensor:
    """Return a float32 mask over the full sequence (chat template + user prompt + generation)."""
    total_len = int(prompt_len_full) + int(gen_len)
    mask = torch.ones((total_len,), dtype=torch.float32)

    prompt_keep = _stop_keep_mask(user_prompt_tokens)
    for j, abs_idx in enumerate(user_prompt_indices):
        if 0 <= int(abs_idx) < int(prompt_len_full) and j < int(prompt_keep.numel()):
            mask[int(abs_idx)] = prompt_keep[j]

    gen_keep = _stop_keep_mask(generation_tokens)
    for g in range(int(gen_keep.numel())):
        abs_idx = int(prompt_len_full) + g
        if 0 <= abs_idx < total_len:
            mask[abs_idx] = gen_keep[g]

    return mask


def faithfulness_test_skip_tokens(
    llm_evaluator: Any,
    attribution: torch.Tensor,
    prompt: str,
    generation: str,
    *,
    keep_prompt_token_indices: Sequence[int],
    user_prompt_indices: Optional[Sequence[int]] = None,
    k: int = 20,
) -> Tuple[float, float, float]:
    """Token-level MAS/RISE faithfulness via guided deletion in k perturbation steps, skipping specified prompt tokens.

    This is a drop-in replacement for llm_attr_eval.LLMAttributionEvaluator.faithfulness_test
    when an experimental protocol wants to soft-delete some prompt tokens (e.g., stop tokens)
    from the perturbation path.
    """

    def auc(arr: np.ndarray) -> float:
        return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / max(1, (arr.shape[0] - 1))

    pad_token_id = llm_evaluator._ensure_pad_token_id()

    user_prompt = prompt
    formatted_prompt = llm_evaluator.format_prompt(user_prompt)

    formatted_ids = llm_evaluator.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_ids = formatted_ids.to(llm_evaluator.device)
    prompt_ids_perturbed = prompt_ids.clone()

    generation_ids = llm_evaluator.tokenizer(
        generation + llm_evaluator.tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(llm_evaluator.device)

    attr_cpu = attribution.detach().cpu()
    w = attr_cpu.sum(0)
    P = int(w.numel())

    keep: List[int] = []
    seen: set[int] = set()
    for raw in keep_prompt_token_indices:
        try:
            idx = int(raw)
        except Exception:
            continue
        if 0 <= idx < P and idx not in seen:
            keep.append(idx)
            seen.add(idx)
    keep.sort()

    K = len(keep)
    if K > 0:
        steps = int(k) if k is not None else 0
        if steps <= 0:
            steps = 1
        steps = min(steps, K)
    else:
        steps = 0

    scores = np.zeros(steps + 1, dtype=np.float64)
    density = np.zeros(steps + 1, dtype=np.float64)

    if user_prompt_indices is not None:
        prompt_positions = [int(x) for x in user_prompt_indices]
        if len(prompt_positions) != P:
            raise ValueError(
                "user_prompt_indices length does not match prompt-side attribution length: "
                f"indices P={len(prompt_positions)}, attr P={P}."
            )
        if P and max(prompt_positions) >= int(prompt_ids_perturbed.shape[1]):
            raise ValueError("user_prompt_indices contains an out-of-bounds index for formatted prompt ids.")
    else:
        user_ids = llm_evaluator.tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        user_start = llm_evaluator._find_subsequence_start(formatted_ids[0], user_ids[0])
        if user_start is None:
            raise RuntimeError("Failed to locate user prompt token span inside formatted chat prompt.")
        prompt_positions = [int(user_start) + j for j in range(P)]

    scores[0] = (
        llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
    )
    density[0] = 1.0

    if K == 0:
        return auc(scores), auc(scores), auc(scores)

    w_keep = w.index_select(0, torch.as_tensor(keep, dtype=torch.long))
    sorted_local = torch.argsort(w_keep, descending=True)
    sorted_keep = [keep[int(i.item())] for i in sorted_local]
    attr_sum = float(w_keep.sum().item())

    if attr_sum <= 0:
        density = np.linspace(1.0, 0.0, steps + 1)

    base = K // steps
    remainder = K % steps
    start = 0
    for step in range(steps):
        size = base + (1 if step < remainder else 0)
        group = sorted_keep[start : start + size]
        start += size

        for idx in group:
            abs_pos = int(prompt_positions[int(idx)])
            prompt_ids_perturbed[0, abs_pos] = pad_token_id
        scores[step + 1] = (
            llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
        )
        if attr_sum > 0:
            dec = float(w.index_select(0, torch.as_tensor(group, dtype=torch.long)).sum().item()) / attr_sum
            density[step + 1] = density[step] - dec

    min_normalized_pred = 1.0
    normalized_model_response = scores.copy()
    for i in range(len(scores)):
        normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    alignment_penalty = np.abs(normalized_model_response - density)
    corrected_scores = normalized_model_response + alignment_penalty
    corrected_scores = corrected_scores.clip(0.0, 1.0)
    corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

    if np.isnan(corrected_scores).any():
        corrected_scores = np.linspace(1.0, 0.0, len(scores))

    return auc(normalized_model_response), auc(corrected_scores), auc(normalized_model_response + alignment_penalty)


def evaluate_attr_recovery_skip_tokens(
    attribution_prompt: torch.Tensor,
    *,
    keep_prompt_token_indices: Sequence[int],
    gold_prompt_token_indices: Sequence[int],
    top_fraction: float = 0.1,
) -> float:
    """Recall of gold prompt tokens among top-attributed prompt tokens, skipping specified tokens."""
    if attribution_prompt.ndim != 2:
        raise ValueError("Expected 2D prompt-side attribution matrix [R, P].")

    P = int(attribution_prompt.shape[1])
    keep: List[int] = []
    seen: set[int] = set()
    for raw in keep_prompt_token_indices or []:
        try:
            idx = int(raw)
        except Exception:
            continue
        if 0 <= idx < P and idx not in seen:
            keep.append(idx)
            seen.add(idx)
    keep.sort()
    if not keep:
        return float("nan")

    gold: set[int] = set()
    for raw in gold_prompt_token_indices or []:
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx in seen:
            gold.add(idx)
    if not gold:
        return float("nan")

    w = torch.nan_to_num(attribution_prompt.sum(0).to(dtype=torch.float32), nan=0.0).clamp(min=0.0)
    w_keep = w.index_select(0, torch.as_tensor(keep, dtype=torch.long))

    frac = float(top_fraction)
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    k = max(1, int(math.ceil(float(len(keep)) * frac)))
    k = min(k, int(len(keep)))
    topk_local = torch.topk(w_keep, k, largest=True).indices.tolist()
    topk = {keep[int(i)] for i in topk_local}
    hit = len(topk.intersection(gold))
    return float(hit) / float(len(gold))


@dataclass
class StopTokenConfig:
    stop_tokens: List[str]
    skip_whitespace: bool
    strip_before_match: bool


class LLMIFRAttributionImproved(llm_attr.LLMIFRAttribution):
    """Experimental FT-IFR (ifr_multi_hop_stop_words) variant with stop-token soft deletion."""

    def _stop_config(self) -> StopTokenConfig:
        return StopTokenConfig(
            stop_tokens=list(STOP_TOKENS),
            skip_whitespace=bool(SKIP_WHITESPACE),
            strip_before_match=bool(STRIP_BEFORE_MATCH),
        )

    @torch.no_grad()
    def calculate_ifr_multi_hop_stop_words(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> llm_attr.LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len_full, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_stop_words",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "stop_config": self._stop_config().__dict__,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        if sink_span is None:
            sink_span = (0, gen_len - 1)
        span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end >= gen_len:
            raise ValueError(f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len}.")

        if thinking_span is None:
            thinking_span = sink_span
        think_start, think_end = thinking_span
        if think_start < 0 or think_end < think_start or think_end >= gen_len:
            raise ValueError(f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len}.")

        sink_start_abs = int(prompt_len_full) + int(span_start)
        sink_end_abs = int(prompt_len_full) + int(span_end)
        think_start_abs = int(prompt_len_full) + int(think_start)
        think_end_abs = int(prompt_len_full) + int(think_end)

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[int(prompt_len_full) : int(prompt_len_full) + int(gen_len)] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        stop_keep_mask_full = _build_stop_keep_mask_full(
            prompt_len_full=int(prompt_len_full),
            gen_len=int(gen_len),
            user_prompt_indices=list(getattr(self, "user_prompt_indices", []) or []),
            user_prompt_tokens=list(getattr(self, "user_prompt_tokens", []) or []),
            generation_tokens=list(getattr(self, "generation_tokens", []) or []),
        )

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        hop_count = max(0, int(n_hops))

        # Base: sink-span aggregate, with sink positions masked if they are stop tokens.
        sink_gen_tokens = list(getattr(self, "generation_tokens", []) or [])
        sink_stops = []
        for gi in range(int(span_start), int(span_end) + 1):
            tok = sink_gen_tokens[gi] if 0 <= gi < len(sink_gen_tokens) else ""
            sink_stops.append(is_stop_token(tok))
        base_weights = None
        if any(sink_stops):
            base_weights = torch.as_tensor(
                [0.0 if st else 1.0 for st in sink_stops],
                dtype=params.model_dtype,
            )

        base_ifr_raw = compute_ifr_sentence_aggregate(
            sink_start=sink_start_abs,
            sink_end=sink_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            sink_weights=base_weights,
            rotary_emb=metadata.rotary_emb,
        )

        base_total = base_ifr_raw.token_importance_total.to(dtype=torch.float32) * stop_keep_mask_full
        base_ifr = IFRAggregate(
            per_layer=base_ifr_raw.per_layer,
            token_importance_total=base_total,
            head_importance_total=base_ifr_raw.head_importance_total,
            ffn_importance_per_layer=base_ifr_raw.ffn_importance_per_layer,
            resid_ffn_importance_per_layer=base_ifr_raw.resid_ffn_importance_per_layer,
        )

        raw_attributions: List[IFRAggregate] = [base_ifr]

        # Observation mask: respect provided observation_mask, otherwise follow the core FT behavior,
        # and then apply stop-token masking so stop tokens are fully removed from the observation path.
        if obs_mask_tensor is None:
            obs_mask = torch.ones((total_len,), dtype=torch.float32)
            obs_mask[int(think_start_abs) : min(int(think_end_abs) + 1, total_len)] = 0.0
            obs_mask[int(sink_start_abs) : min(int(sink_end_abs) + 1, total_len)] = 0.0
            if int(think_end_abs) + 1 < total_len:
                obs_mask[int(think_end_abs) + 1 :] = 0.0
        else:
            obs_mask = obs_mask_tensor.clone().to(dtype=torch.float32)
            if int(obs_mask.shape[0]) != int(total_len):
                raise ValueError("observation_mask must match sequence length.")

        obs_mask = obs_mask * stop_keep_mask_full

        base_obs = base_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        denom_base = float(base_total.sum().item())
        thinking_slice = base_total[int(think_start_abs) : int(think_end_abs) + 1]
        w_thinking = thinking_slice.detach().clone().to(params.model_dtype)
        current_ratio = float(w_thinking.sum().item()) / (denom_base + 1e-12) if denom_base > 0 else 0.0
        ratios: List[float] = [current_ratio]

        # Multi-hop: thinking-span re-aggregation with masked weights.
        for hop in range(1, hop_count + 1):
            if float(w_thinking.sum().item()) <= 0.0 or float(current_ratio) <= 0.0:
                # Terminate remaining hops with zero vectors to keep shapes stable.
                zeros = torch.zeros_like(base_total)
                for _ in range(hop, hop_count + 1):
                    raw_attributions.append(
                        IFRAggregate(
                            per_layer=[],
                            token_importance_total=zeros,
                            head_importance_total=torch.zeros_like(base_ifr.head_importance_total),
                            ffn_importance_per_layer=torch.zeros_like(base_ifr.ffn_importance_per_layer),
                            resid_ffn_importance_per_layer=torch.zeros_like(base_ifr.resid_ffn_importance_per_layer),
                        )
                    )
                    per_hop_obs.append(torch.zeros_like(base_total))
                    ratios.append(0.0)
                break

            hop_ifr_raw = compute_ifr_sentence_aggregate(
                sink_start=think_start_abs,
                sink_end=think_end_abs,
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_weights=w_thinking,
                rotary_emb=metadata.rotary_emb,
            )

            hop_total = hop_ifr_raw.token_importance_total.to(dtype=torch.float32) * stop_keep_mask_full
            hop_ifr = IFRAggregate(
                per_layer=hop_ifr_raw.per_layer,
                token_importance_total=hop_total,
                head_importance_total=hop_ifr_raw.head_importance_total,
                ffn_importance_per_layer=hop_ifr_raw.ffn_importance_per_layer,
                resid_ffn_importance_per_layer=hop_ifr_raw.resid_ffn_importance_per_layer,
            )
            raw_attributions.append(hop_ifr)

            obs_only = hop_total * obs_mask * float(current_ratio)
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            thinking_slice = hop_total[int(think_start_abs) : int(think_end_abs) + 1]
            w_thinking = thinking_slice.detach().clone().to(params.model_dtype)

            hop_denom = float(hop_total.sum().item())
            if hop_denom <= 0.0:
                current_ratio = 0.0
            else:
                current_ratio *= float(w_thinking.sum().item()) / (hop_denom + 1e-12)
            ratios.append(float(current_ratio))

        obs_avg = obs_accum / float(max(1, hop_count))
        observation: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        multi_hop = MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=ratios, observation=observation)

        eval_vector = multi_hop.observation["sum"]
        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)
        for offset in range(int(span_start), int(span_end) + 1):
            tok = sink_gen_tokens[offset] if 0 <= offset < len(sink_gen_tokens) else ""
            if is_stop_token(tok):
                continue
            score_array[offset] = eval_vector

        projected_per_hop = [self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(self.prompt_tokens, obs["mask"].view(1, -1))[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "multi_hop_stop_words",
                "sink_span_generation": (int(span_start), int(span_end)),
                "sink_span_absolute": (int(sink_start_abs), int(sink_end_abs)),
                "thinking_span_generation": (int(think_start), int(think_end)),
                "thinking_span_absolute": (int(think_start_abs), int(think_end_abs)),
                "renorm_threshold": float(renorm),
                "n_hops": int(n_hops),
                "thinking_ratios": ratios,
                "per_hop_projected": projected_per_hop,
                "observation_projected": observation_projected,
                "stop_config": self._stop_config().__dict__,
                "raw": multi_hop,
            }
        }

        return self._finalize_result(score_array, metadata=meta)


##########################################
# Split-hop configuration (edit here)
##########################################

# Default number of equal-length segments to split the thinking span into (token-based).
# Used only when n_hops is not provided.
SPLIT_HOP_NUM_SEGMENTS: int = 5


def _split_span_equal(start: int, end: int, num_segments: int) -> List[Tuple[int, int]]:
    """Split an inclusive span [start, end] into up to num_segments equal-size segments."""
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        return []

    length = end_i - start_i + 1
    n = max(1, int(num_segments))
    base = length // n
    rem = length % n

    segments: List[Tuple[int, int]] = []
    cur = start_i
    for i in range(n):
        seg_len = base + (1 if i < rem else 0)
        if seg_len <= 0:
            continue
        seg_start = cur
        seg_end = cur + seg_len - 1
        segments.append((seg_start, seg_end))
        cur = seg_end + 1
    return segments


@dataclass
class SplitHopConfig:
    num_segments: int


class LLMIFRAttributionSplitHop(llm_attr.LLMIFRAttribution):
    """Experimental FT-IFR variant that split-hops over a segmented thinking span.

    This implementation follows "scheme B":
    - The model forward pass is unchanged (no token deletion).
    - Attribution remains token-level and prompt-only (via the same observation mask as multi-hop FT-IFR).
    - The thinking span is split into equal-length token segments; we propagate attribution mass backward
      segment-by-segment, redistributing each segment's pending mass to earlier segments and the prompt.
    """

    def _split_hop_config(self, *, num_segments: int) -> SplitHopConfig:
        return SplitHopConfig(num_segments=int(num_segments))

    @torch.no_grad()
    def calculate_ifr_multi_hop_split_hop(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> llm_attr.LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len_full, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        num_segments = int(n_hops) if n_hops is not None else int(SPLIT_HOP_NUM_SEGMENTS)
        if num_segments < 0:
            num_segments = 0

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_split_hop",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "split_hop_config": self._split_hop_config(num_segments=num_segments).__dict__,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        if sink_span is None:
            sink_span = (0, gen_len - 1)
        span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end >= gen_len:
            raise ValueError(f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len}.")

        if thinking_span is None:
            thinking_span = sink_span
        think_start, think_end = thinking_span
        if think_start < 0 or think_end < think_start or think_end >= gen_len:
            raise ValueError(f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len}.")

        sink_start_abs = int(prompt_len_full) + int(span_start)
        sink_end_abs = int(prompt_len_full) + int(span_end)
        think_start_abs = int(prompt_len_full) + int(think_start)
        think_end_abs = int(prompt_len_full) + int(think_end)

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[int(prompt_len_full) : int(prompt_len_full) + int(gen_len)] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        base_ifr_raw = compute_ifr_sentence_aggregate(
            sink_start=sink_start_abs,
            sink_end=sink_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            rotary_emb=metadata.rotary_emb,
        )
        base_total = base_ifr_raw.token_importance_total.to(dtype=torch.float32)
        base_ifr = IFRAggregate(
            per_layer=base_ifr_raw.per_layer,
            token_importance_total=base_total,
            head_importance_total=base_ifr_raw.head_importance_total,
            ffn_importance_per_layer=base_ifr_raw.ffn_importance_per_layer,
            resid_ffn_importance_per_layer=base_ifr_raw.resid_ffn_importance_per_layer,
        )

        if obs_mask_tensor is None:
            obs_mask = torch.ones((total_len,), dtype=torch.float32)
            obs_mask[int(think_start_abs) : min(int(think_end_abs) + 1, total_len)] = 0.0
            obs_mask[int(sink_start_abs) : min(int(sink_end_abs) + 1, total_len)] = 0.0
            if int(think_end_abs) + 1 < total_len:
                obs_mask[int(think_end_abs) + 1 :] = 0.0
        else:
            obs_mask = obs_mask_tensor.clone().to(dtype=torch.float32)
            if int(obs_mask.shape[0]) != int(total_len):
                raise ValueError("observation_mask must match sequence length.")

        base_obs = base_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        if num_segments <= 0:
            segments_gen = []
        else:
            segments_gen = _split_span_equal(int(think_start), int(think_end), num_segments)
        segments_abs: List[Tuple[int, int]] = [
            (int(prompt_len_full) + int(s), int(prompt_len_full) + int(e)) for s, e in segments_gen
        ]

        # Pending mass (scheme B): start from hop0 mass on each segment and redistribute backward.
        pending_weights: List[torch.Tensor] = [
            base_total[s_abs : e_abs + 1].clone().to(dtype=torch.float32) for (s_abs, e_abs) in segments_abs
        ]

        raw_attributions: List[IFRAggregate] = [base_ifr]
        hop_details: List[Dict[str, Any]] = []

        # Process segments from last to first: cot[K-1] -> cot[K-2] -> ... -> cot[0] -> prompt
        for seg_idx in range(len(segments_abs) - 1, -1, -1):
            seg_start_abs, seg_end_abs = segments_abs[seg_idx]
            w_pending = pending_weights[seg_idx]
            pending_mass = float(w_pending.sum().item())

            if pending_mass <= 0.0:
                raw_attributions.append(
                    IFRAggregate(
                        per_layer=[],
                        token_importance_total=torch.zeros_like(base_total),
                        head_importance_total=torch.zeros_like(base_ifr.head_importance_total),
                        ffn_importance_per_layer=torch.zeros_like(base_ifr.ffn_importance_per_layer),
                        resid_ffn_importance_per_layer=torch.zeros_like(base_ifr.resid_ffn_importance_per_layer),
                    )
                )
                per_hop_obs.append(torch.zeros_like(base_total))
                hop_details.append(
                    {
                        "segment_index": int(seg_idx),
                        "segment_span_generation": segments_gen[seg_idx],
                        "segment_span_absolute": (int(seg_start_abs), int(seg_end_abs)),
                        "pending_mass": float(pending_mass),
                        "masked_denom": 0.0,
                        "alpha": 0.0,
                    }
                )
                continue

            hop_ifr_raw = compute_ifr_sentence_aggregate(
                sink_start=int(seg_start_abs),
                sink_end=int(seg_end_abs),
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_weights=w_pending.to(dtype=params.model_dtype),
                rotary_emb=metadata.rotary_emb,
            )

            hop_total = hop_ifr_raw.token_importance_total.to(dtype=torch.float32)
            hop_total_masked = hop_total.clone()
            hop_total_masked[int(seg_start_abs) : int(seg_end_abs) + 1] = 0.0

            masked_denom = float(hop_total_masked.sum().item())
            alpha = float(pending_mass) / (masked_denom + 1e-12) if masked_denom > 0.0 else 0.0
            distributed = hop_total_masked * alpha

            # Prompt-only observation accumulation (generation+thinking excluded by obs_mask).
            obs_only = distributed * obs_mask
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            # Redistribute mass to earlier segments for further propagation.
            if alpha > 0.0:
                for j in range(seg_idx):
                    js_abs, je_abs = segments_abs[j]
                    pending_weights[j] = pending_weights[j] + distributed[js_abs : je_abs + 1]
            pending_weights[seg_idx] = torch.zeros_like(pending_weights[seg_idx])

            hop_ifr = IFRAggregate(
                per_layer=hop_ifr_raw.per_layer,
                token_importance_total=hop_total_masked,
                head_importance_total=hop_ifr_raw.head_importance_total,
                ffn_importance_per_layer=hop_ifr_raw.ffn_importance_per_layer,
                resid_ffn_importance_per_layer=hop_ifr_raw.resid_ffn_importance_per_layer,
            )
            raw_attributions.append(hop_ifr)
            hop_details.append(
                {
                    "segment_index": int(seg_idx),
                    "segment_span_generation": segments_gen[seg_idx],
                    "segment_span_absolute": (int(seg_start_abs), int(seg_end_abs)),
                    "pending_mass": float(pending_mass),
                    "masked_denom": float(masked_denom),
                    "alpha": float(alpha),
                }
            )

        obs_avg = obs_accum / float(max(1, len(per_hop_obs)))
        observation: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        multi_hop = MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=[], observation=observation)
        eval_vector = multi_hop.observation["sum"]

        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)
        for offset in range(int(span_start), int(span_end) + 1):
            score_array[offset] = eval_vector

        projected_per_hop = [self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(self.prompt_tokens, obs["mask"].view(1, -1))[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "multi_hop_split_hop",
                "sink_span_generation": (int(span_start), int(span_end)),
                "sink_span_absolute": (int(sink_start_abs), int(sink_end_abs)),
                "thinking_span_generation": (int(think_start), int(think_end)),
                "thinking_span_absolute": (int(think_start_abs), int(think_end_abs)),
                "renorm_threshold": float(renorm),
                "split_hop_config": self._split_hop_config(num_segments=num_segments).__dict__,
                "segments_generation": [(int(s), int(e)) for (s, e) in segments_gen],
                "segments_absolute": [(int(s), int(e)) for (s, e) in segments_abs],
                "hop_details": hop_details,
                "per_hop_projected": projected_per_hop,
                "observation_projected": observation_projected,
                "raw": multi_hop,
            }
        }

        return self._finalize_result(score_array, metadata=meta)

### both

class LLMIFRAttributionBoth(llm_attr.LLMIFRAttribution):
    """Experimental FT-IFR variant combining stop-token soft deletion with in_all_gen multi-hop.

    Notes
    -----
    - Hop 0 aggregates over sink_span (= output/answer).
    - Recursive hops aggregate over all_gen_span (= CoT + output).
    - The EOS token (assumed to be the last generation token) is excluded from all spans.
    - Stop tokens are soft-deleted from hop aggregation weights and observation vectors.
    - Returned attribution matrix follows scheme B (rows filled for sink_span_generation).
    """

    def _stop_config(self) -> StopTokenConfig:
        return StopTokenConfig(
            stop_tokens=list(STOP_TOKENS),
            skip_whitespace=bool(SKIP_WHITESPACE),
            strip_before_match=bool(STRIP_BEFORE_MATCH),
        )

    @torch.no_grad()
    def calculate_ifr_multi_hop_both(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> llm_attr.LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len_full, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_both",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "stop_config": self._stop_config().__dict__,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        end_no_eos = _last_attributable_generation_index(
            self.tokenizer,
            self.generation_ids,
            list(getattr(self, "generation_tokens", []) or []),
        )
        if end_no_eos < 0:
            score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop_both",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "stop_config": self._stop_config().__dict__,
                    "note": "No non-EOS generation tokens; returning NaN attribution matrix.",
                }
            }
            return self._finalize_result(score_array, metadata=metadata)

        # Scheme B: fill only sink_span rows; default to all_gen_span.
        if sink_span is None:
            span_start, span_end = (0, end_no_eos)
        else:
            span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end > end_no_eos:
            raise ValueError(
                f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len} with EOS excluded."
            )

        think_start_abs: Optional[int] = None
        think_end_abs: Optional[int] = None
        if thinking_span is not None:
            think_start, think_end = thinking_span
            if think_start < 0 or think_end < think_start or think_end > end_no_eos:
                raise ValueError(
                    f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len} with EOS excluded."
                )
            think_start_abs = int(prompt_len_full) + int(think_start)
            think_end_abs = int(prompt_len_full) + int(think_end)

        # Internal hop span: CoT+output when both spans are provided; otherwise default all_gen.
        if sink_span is not None and thinking_span is not None:
            all_gen_start = min(int(span_start), int(thinking_span[0]))
            all_gen_end = max(int(span_end), int(thinking_span[1]))
        else:
            all_gen_start = 0
            all_gen_end = int(end_no_eos)

        all_gen_start = max(0, int(all_gen_start))
        all_gen_end = min(int(end_no_eos), int(all_gen_end))
        if all_gen_end < all_gen_start:
            raise ValueError("Derived all_gen_span is empty after EOS exclusion and bounds checking.")

        sink_start_abs = int(prompt_len_full) + int(span_start)
        sink_end_abs = int(prompt_len_full) + int(span_end)
        all_gen_start_abs = int(prompt_len_full) + int(all_gen_start)
        all_gen_end_abs = int(prompt_len_full) + int(all_gen_end)

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[int(prompt_len_full) : int(prompt_len_full) + int(gen_len)] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        stop_keep_mask_full = _build_stop_keep_mask_full(
            prompt_len_full=int(prompt_len_full),
            gen_len=int(gen_len),
            user_prompt_indices=list(getattr(self, "user_prompt_indices", []) or []),
            user_prompt_tokens=list(getattr(self, "user_prompt_tokens", []) or []),
            generation_tokens=list(getattr(self, "generation_tokens", []) or []),
        )

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        hop_count = max(0, int(n_hops))

        # Base hop: aggregate over the selected output span, excluding stop tokens from the sink weights.
        gen_tokens = list(getattr(self, "generation_tokens", []) or [])
        sink_stops = []
        for gi in range(int(span_start), int(span_end) + 1):
            tok = gen_tokens[gi] if 0 <= gi < len(gen_tokens) else ""
            sink_stops.append(is_stop_token(tok))
        base_weights = None
        if any(sink_stops):
            base_weights = torch.as_tensor(
                [0.0 if st else 1.0 for st in sink_stops],
                dtype=params.model_dtype,
            )

        base_ifr_raw = compute_ifr_sentence_aggregate(
            sink_start=sink_start_abs,
            sink_end=sink_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            sink_weights=base_weights,
            rotary_emb=metadata.rotary_emb,
        )
        base_total = torch.nan_to_num(
            base_ifr_raw.token_importance_total.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_total = base_total * stop_keep_mask_full
        base_ifr = IFRAggregate(
            per_layer=base_ifr_raw.per_layer,
            token_importance_total=base_total,
            head_importance_total=base_ifr_raw.head_importance_total,
            ffn_importance_per_layer=base_ifr_raw.ffn_importance_per_layer,
            resid_ffn_importance_per_layer=base_ifr_raw.resid_ffn_importance_per_layer,
        )

        # Observation mask: respect provided observation_mask, otherwise follow in_all_gen scheme B,
        # and then apply stop-token masking so stop tokens are removed from the observation path.
        if obs_mask_tensor is None:
            obs_mask = torch.ones((total_len,), dtype=torch.float32)
            obs_mask[int(all_gen_start_abs) : min(int(all_gen_end_abs) + 1, total_len)] = 0.0
            if int(all_gen_end_abs) + 1 < total_len:
                obs_mask[int(all_gen_end_abs) + 1 :] = 0.0
        else:
            obs_mask = obs_mask_tensor.clone().to(dtype=torch.float32)
            if int(obs_mask.shape[0]) != int(total_len):
                raise ValueError("observation_mask must match sequence length.")

        obs_mask = obs_mask * stop_keep_mask_full

        base_obs = base_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        raw_attributions: List[IFRAggregate] = [base_ifr]

        denom_base = float(base_total.sum().item())
        w_span_raw = base_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
        w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
        w_span_sum = float(w_span_raw.sum().item())
        w_span_weights = (
            (w_span_raw / (w_span_sum + 1e-12))
            if w_span_sum > 0.0
            else torch.zeros_like(w_span_raw, dtype=torch.float32)
        )
        current_ratio = float(w_span_sum) / (denom_base + 1e-12) if denom_base > 0.0 else 0.0
        ratios: List[float] = [float(current_ratio)]

        for hop in range(1, hop_count + 1):
            if float(w_span_sum) <= 0.0 or float(current_ratio) <= 0.0:
                zeros = torch.zeros_like(base_total)
                for _ in range(hop, hop_count + 1):
                    raw_attributions.append(
                        IFRAggregate(
                            per_layer=[],
                            token_importance_total=zeros,
                            head_importance_total=torch.zeros_like(base_ifr.head_importance_total),
                            ffn_importance_per_layer=torch.zeros_like(base_ifr.ffn_importance_per_layer),
                            resid_ffn_importance_per_layer=torch.zeros_like(base_ifr.resid_ffn_importance_per_layer),
                        )
                    )
                    per_hop_obs.append(torch.zeros_like(base_total))
                    ratios.append(0.0)
                break

            hop_ifr_raw = compute_ifr_sentence_aggregate(
                sink_start=all_gen_start_abs,
                sink_end=all_gen_end_abs,
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_weights=w_span_weights,
                rotary_emb=metadata.rotary_emb,
            )

            hop_total = torch.nan_to_num(
                hop_ifr_raw.token_importance_total.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            hop_total = hop_total * stop_keep_mask_full
            hop_ifr = IFRAggregate(
                per_layer=hop_ifr_raw.per_layer,
                token_importance_total=hop_total,
                head_importance_total=hop_ifr_raw.head_importance_total,
                ffn_importance_per_layer=hop_ifr_raw.ffn_importance_per_layer,
                resid_ffn_importance_per_layer=hop_ifr_raw.resid_ffn_importance_per_layer,
            )
            raw_attributions.append(hop_ifr)

            obs_only = hop_total * obs_mask * float(current_ratio)
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            w_span_raw = hop_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
            w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
            w_span_sum = float(w_span_raw.sum().item())
            w_span_weights = (
                (w_span_raw / (w_span_sum + 1e-12))
                if w_span_sum > 0.0
                else torch.zeros_like(w_span_raw, dtype=torch.float32)
            )

            hop_denom = float(hop_total.sum().item())
            if hop_denom <= 0.0:
                current_ratio = 0.0
            else:
                current_ratio *= float(w_span_sum) / (hop_denom + 1e-12)
            ratios.append(float(current_ratio))

        obs_avg = obs_accum / float(max(1, hop_count))
        observation: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        multi_hop = MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=ratios, observation=observation)
        eval_vector = multi_hop.observation["sum"]

        score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
        for offset in range(int(span_start), int(span_end) + 1):
            tok = gen_tokens[offset] if 0 <= offset < len(gen_tokens) else ""
            if is_stop_token(tok):
                continue
            score_array[offset] = eval_vector

        generation_start_abs = int(prompt_len_full)
        generation_end_abs = int(prompt_len_full) + int(gen_len)

        def _generation_slice(vector: torch.Tensor) -> List[float]:
            clean = torch.nan_to_num(
                vector.detach().cpu().to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return [float(x) for x in clean[generation_start_abs:generation_end_abs].tolist()]

        projected_per_hop = [self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions]
        per_hop_generation = [
            _generation_slice(result.token_importance_total) for result in multi_hop.raw_attributions
        ]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(self.prompt_tokens, obs["mask"].view(1, -1))[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }
        observation_generation = _generation_slice(obs["sum"])

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "multi_hop_both",
                "sink_span_generation": (int(span_start), int(span_end)),
                "sink_span_absolute": (int(sink_start_abs), int(sink_end_abs)),
                "thinking_span_generation": (
                    (int(thinking_span[0]), int(thinking_span[1])) if thinking_span is not None else None
                ),
                "thinking_span_absolute": (
                    (int(think_start_abs), int(think_end_abs)) if think_start_abs is not None else None
                ),
                "all_gen_span_generation": (int(all_gen_start), int(all_gen_end)),
                "all_gen_span_absolute": (int(all_gen_start_abs), int(all_gen_end_abs)),
                "renorm_threshold": float(renorm),
                "n_hops": int(n_hops),
                "thinking_ratios": multi_hop.thinking_ratios,
                "per_hop_projected": projected_per_hop,
                "per_hop_generation": per_hop_generation,
                "observation_projected": observation_projected,
                "observation_generation": observation_generation,
                "stop_config": self._stop_config().__dict__,
                "raw": multi_hop,
                "note": (
                    "both = stop_words + in_all_gen (scheme B; hop0 over sink_span; "
                    "recursive hops over all_gen_span with EOS excluded)."
                ),
            }
        }

        return self._finalize_result(score_array, metadata=meta)

### in_all_gen

class LLMIFRAttributionInAllGen(llm_attr.LLMIFRAttribution):
    """Experimental FT-IFR variant that runs all hops over the full generation span (CoT + output).

    Notes
    -----
    This method follows scheme B for compatibility:
    - Internally, hop0 and subsequent hops aggregate over all_gen_span (= CoT + output).
    - The returned attribution matrix is still written only for `sink_span` rows (answer tokens),
      falling back to all_gen_span when `sink_span` is not provided.
    - The EOS token (assumed to be the last generation token) is excluded from all spans.
    """

    @torch.no_grad()
    def calculate_ifr_in_all_gen(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> llm_attr.LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len_full, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "in_all_gen",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        end_no_eos = _last_attributable_generation_index(
            self.tokenizer,
            self.generation_ids,
            list(getattr(self, "generation_tokens", []) or []),
        )
        if end_no_eos < 0:
            score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "in_all_gen",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "note": "No non-EOS generation tokens; returning NaN attribution matrix.",
                }
            }
            return self._finalize_result(score_array, metadata=metadata)

        # Scheme B: fill only sink_span rows; default to all_gen_span.
        if sink_span is None:
            span_start, span_end = (0, end_no_eos)
        else:
            span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end > end_no_eos:
            raise ValueError(
                f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len} with EOS excluded."
            )

        think_start_abs: Optional[int] = None
        think_end_abs: Optional[int] = None
        if thinking_span is not None:
            think_start, think_end = thinking_span
            if think_start < 0 or think_end < think_start or think_end > end_no_eos:
                raise ValueError(
                    f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len} with EOS excluded."
                )
            think_start_abs = int(prompt_len_full) + int(think_start)
            think_end_abs = int(prompt_len_full) + int(think_end)

        # Internal hop span: CoT+output when both spans are provided; otherwise default all_gen.
        if sink_span is not None and thinking_span is not None:
            all_gen_start = min(int(span_start), int(thinking_span[0]))
            all_gen_end = max(int(span_end), int(thinking_span[1]))
        else:
            all_gen_start = 0
            all_gen_end = int(end_no_eos)

        all_gen_start = max(0, int(all_gen_start))
        all_gen_end = min(int(end_no_eos), int(all_gen_end))
        if all_gen_end < all_gen_start:
            raise ValueError("Derived all_gen_span is empty after EOS exclusion and bounds checking.")

        sink_start_abs = int(prompt_len_full) + int(span_start)
        sink_end_abs = int(prompt_len_full) + int(span_end)
        all_gen_start_abs = int(prompt_len_full) + int(all_gen_start)
        all_gen_end_abs = int(prompt_len_full) + int(all_gen_end)

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[int(prompt_len_full) : int(prompt_len_full) + int(gen_len)] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        # Multi-hop IFR over all_gen_span (scheme B). We implement the hop loop inline
        # to robustly handle rare NaNs in intermediate IFR vectors on long sequences.
        hop_count = max(0, int(n_hops))

        base_ifr_raw = compute_ifr_sentence_aggregate(
            sink_start=all_gen_start_abs,
            sink_end=all_gen_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            rotary_emb=metadata.rotary_emb,
        )
        base_total = torch.nan_to_num(
            base_ifr_raw.token_importance_total.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_ifr = IFRAggregate(
            per_layer=base_ifr_raw.per_layer,
            token_importance_total=base_total,
            head_importance_total=base_ifr_raw.head_importance_total,
            ffn_importance_per_layer=base_ifr_raw.ffn_importance_per_layer,
            resid_ffn_importance_per_layer=base_ifr_raw.resid_ffn_importance_per_layer,
        )

        if obs_mask_tensor is None:
            obs_mask = torch.ones((total_len,), dtype=torch.float32)
            obs_mask[int(all_gen_start_abs) : min(int(all_gen_end_abs) + 1, total_len)] = 0.0
            if int(all_gen_end_abs) + 1 < total_len:
                obs_mask[int(all_gen_end_abs) + 1 :] = 0.0
        else:
            obs_mask = obs_mask_tensor.clone().to(dtype=torch.float32)
            if int(obs_mask.shape[0]) != int(total_len):
                raise ValueError("observation_mask must match sequence length.")

        base_obs = base_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        raw_attributions: List[IFRAggregate] = [base_ifr]

        denom_base = float(base_total.sum().item())
        w_span_raw = base_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
        w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
        w_span_sum = float(w_span_raw.sum().item())
        w_span_weights = (
            (w_span_raw / (w_span_sum + 1e-12))
            if w_span_sum > 0.0
            else torch.zeros_like(w_span_raw, dtype=torch.float32)
        )
        current_ratio = float(w_span_sum) / (denom_base + 1e-12) if denom_base > 0.0 else 0.0
        ratios: List[float] = [float(current_ratio)]

        for hop in range(1, hop_count + 1):
            if float(w_span_sum) <= 0.0 or float(current_ratio) <= 0.0:
                zeros = torch.zeros_like(base_total)
                for _ in range(hop, hop_count + 1):
                    raw_attributions.append(
                        IFRAggregate(
                            per_layer=[],
                            token_importance_total=zeros,
                            head_importance_total=torch.zeros_like(base_ifr.head_importance_total),
                            ffn_importance_per_layer=torch.zeros_like(base_ifr.ffn_importance_per_layer),
                            resid_ffn_importance_per_layer=torch.zeros_like(base_ifr.resid_ffn_importance_per_layer),
                        )
                    )
                    per_hop_obs.append(torch.zeros_like(base_total))
                    ratios.append(0.0)
                break

            hop_ifr_raw = compute_ifr_sentence_aggregate(
                sink_start=all_gen_start_abs,
                sink_end=all_gen_end_abs,
                cache=cache,
                attentions=attentions,
                weight_pack=weight_pack,
                params=params,
                renorm_threshold=renorm,
                sink_weights=w_span_weights,
                rotary_emb=metadata.rotary_emb,
            )

            hop_total = torch.nan_to_num(
                hop_ifr_raw.token_importance_total.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            hop_ifr = IFRAggregate(
                per_layer=hop_ifr_raw.per_layer,
                token_importance_total=hop_total,
                head_importance_total=hop_ifr_raw.head_importance_total,
                ffn_importance_per_layer=hop_ifr_raw.ffn_importance_per_layer,
                resid_ffn_importance_per_layer=hop_ifr_raw.resid_ffn_importance_per_layer,
            )
            raw_attributions.append(hop_ifr)

            obs_only = hop_total * obs_mask * float(current_ratio)
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            w_span_raw = hop_total[int(all_gen_start_abs) : int(all_gen_end_abs) + 1].detach().clone()
            w_span_raw = torch.nan_to_num(w_span_raw, nan=0.0, posinf=0.0, neginf=0.0)
            w_span_sum = float(w_span_raw.sum().item())
            w_span_weights = (
                (w_span_raw / (w_span_sum + 1e-12))
                if w_span_sum > 0.0
                else torch.zeros_like(w_span_raw, dtype=torch.float32)
            )

            hop_denom = float(hop_total.sum().item())
            if hop_denom <= 0.0:
                current_ratio = 0.0
            else:
                current_ratio *= float(w_span_sum) / (hop_denom + 1e-12)
            ratios.append(float(current_ratio))

        obs_avg = obs_accum / float(max(1, hop_count))
        observation: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        multi_hop = MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=ratios, observation=observation)

        eval_vector = multi_hop.observation["sum"]
        score_array = torch.full((int(gen_len), total_len), torch.nan, dtype=torch.float32)
        for offset in range(int(span_start), int(span_end) + 1):
            score_array[offset] = eval_vector

        projected_per_hop = [self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(self.prompt_tokens, obs["mask"].view(1, -1))[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "in_all_gen",
                "sink_span_generation": (int(span_start), int(span_end)),
                "sink_span_absolute": (int(sink_start_abs), int(sink_end_abs)),
                "thinking_span_generation": (
                    (int(thinking_span[0]), int(thinking_span[1])) if thinking_span is not None else None
                ),
                "thinking_span_absolute": (
                    (int(think_start_abs), int(think_end_abs)) if think_start_abs is not None else None
                ),
                "all_gen_span_generation": (int(all_gen_start), int(all_gen_end)),
                "all_gen_span_absolute": (int(all_gen_start_abs), int(all_gen_end_abs)),
                "renorm_threshold": float(renorm),
                "n_hops": int(n_hops),
                "thinking_ratios": multi_hop.thinking_ratios,
                "per_hop_projected": projected_per_hop,
                "observation_projected": observation_projected,
                "raw": multi_hop,
                "note": "scheme B: rows filled for sink_span_generation; hops over all_gen_span (EOS excluded).",
            }
        }

        return self._finalize_result(score_array, metadata=meta)

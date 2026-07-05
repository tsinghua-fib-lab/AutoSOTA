"""Fast (approximate) perturbation-based attribution baselines.

This module provides k-segment approximations for the perturbation baselines
implemented in llm_attr.LLMPerturbationAttribution, but with a much cheaper
inner-loop over source segments (default k=20) instead of sentence masks.

Intended usage: exp/exp2 only (baseline-speed focus; fidelity is secondary).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F

from shared_utils import create_sentence_masks, create_sentences
from llm_attr import LLMAttribution, LLMAttributionResult


def _split_indices_into_k_groups(indices: Sequence[int], k: int) -> List[List[int]]:
    if not indices:
        return []
    steps = int(k) if k is not None else 0
    if steps <= 0:
        steps = 1
    steps = min(steps, len(indices))
    base = len(indices) // steps
    remainder = len(indices) % steps
    groups: List[List[int]] = []
    start = 0
    for i in range(steps):
        size = base + (1 if i < remainder else 0)
        groups.append(list(indices[start : start + size]))
        start += size
    return groups


def _is_valid_token_span(span: object) -> bool:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    a, b = span
    return isinstance(a, int) and isinstance(b, int) and a >= 0 and b >= a


def _resolve_indices_to_explain_from_stack() -> Optional[tuple[int, int]]:
    """Best-effort: pull generation-token span from exp/exp2 caller without changing its API.

    exp/exp2 calls these fast baselines without passing indices_to_explain; to enable
    safe sink-loop pruning (row-only), we opportunistically look for an `example`
    object in caller frames and read `example.indices_to_explain`.

    If not found, returns None and the full sink loop is computed.
    """
    try:
        import inspect
    except Exception:
        return None

    frame = inspect.currentframe()
    try:
        cur = frame.f_back if frame is not None else None
        while cur is not None:
            for name in ("example", "ex"):
                obj = cur.f_locals.get(name)
                if obj is None:
                    continue
                span = getattr(obj, "indices_to_explain", None)
                if _is_valid_token_span(span):
                    return int(span[0]), int(span[1])
            cur = cur.f_back
        return None
    finally:
        # Avoid reference cycles (inspect.currentframe keeps frames alive).
        try:
            del frame
            del cur  # type: ignore[name-defined]
        except Exception:
            pass


class LLMPerturbationFastAttribution(LLMAttribution):
    """K-segment approximations of perturbation baselines (Perturbation / CLP / REAGENT)."""

    def __init__(self, model: Any, tokenizer: Any, generate_kwargs: Optional[dict] = None) -> None:
        super().__init__(model, tokenizer, generate_kwargs)
        self._mlm_tokenizer: Optional[Any] = None
        self._mlm_model: Optional[Any] = None

    def _ensure_mlm(self) -> None:
        if self._mlm_tokenizer is not None and self._mlm_model is not None:
            return
        from transformers import LongformerForMaskedLM, LongformerTokenizer

        self._mlm_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self._mlm_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(self.device)
        self._mlm_model.eval()

    @torch.no_grad()
    def compute_logprob_response_given_prompt(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token log-probabilities of response_ids given prompt_ids.

        prompt_ids: [B, N]
        response_ids: [B, M]
        Returns: [B, M]
        """
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)
        response_start = prompt_ids.shape[1]
        logits_for_response = log_probs[:, response_start - 1 : -1, :]
        gathered = logits_for_response.gather(2, response_ids.unsqueeze(-1))
        return gathered.squeeze(-1)

    @torch.no_grad()
    def compute_kl_response_given_prompt(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """Compute a KL-like per-token score for response_ids given prompt_ids.

        This mirrors llm_attr.LLMPerturbationAttribution.compute_kl_response_given_prompt.
        """
        device = prompt_ids.device
        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits.to(torch.float32)
        log_probs = F.log_softmax(logits, dim=-1)

        _, N = prompt_ids.shape
        M = response_ids.shape[1]
        response_positions = torch.arange(N, N + M, device=device)
        log_probs_response = log_probs[:, response_positions - 1, :]
        log_p = log_probs_response.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

        log_p_minus_log_q = -log_probs_response + log_p.unsqueeze(-1)
        p = log_p.exp()
        kl_scores = (log_p_minus_log_q * p.unsqueeze(-1)).sum(dim=-1)
        return kl_scores

    def _build_source_groups_full(self, *, source_k: int) -> List[torch.Tensor]:
        input_length = int(self.prompt_ids.shape[1])
        generation_length = int(self.generation_ids.shape[1])
        total_length = input_length + generation_length

        source_positions_full: List[int] = list(self.user_prompt_indices or [])
        source_positions_full.extend(range(input_length, total_length))

        groups = _split_indices_into_k_groups(source_positions_full, source_k)
        return [torch.tensor(g, dtype=torch.long) for g in groups if g]

    def calculate_feature_ablation_segments(
        self,
        prompt: str,
        *,
        baseline: int,
        measure: str = "log_loss",
        target: Optional[str] = None,
        source_k: int = 20,
    ) -> LLMAttributionResult:
        """Approximate sentence-loop perturbation via fixed k source segments per step.

        - sink unit: generation sentences (same as baseline)
        - source unit: k segments over (user-prompt tokens + all generation tokens),
          restricted to currently-available tokens (prompt + previous generations).
        """
        sink_span = _resolve_indices_to_explain_from_stack()

        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        input_ids_all = self.prompt_ids.clone()
        input_length = int(self.prompt_ids.shape[1])
        generation_length = int(self.generation_ids.shape[1])
        total_length = input_length + generation_length

        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        score_array = torch.full((generation_length, total_length), torch.nan)
        source_groups_full = self._build_source_groups_full(source_k=source_k)

        for step in range(int(sentence_masks_generation.shape[0])):
            input_ids_all = input_ids_all.detach()

            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0]
            if gen_token_indices.numel() == 0:
                continue
            gen_tokens = self.generation_ids[:, gen_token_indices]

            if sink_span is not None:
                span_start, span_end = sink_span
                min_tok = int(gen_token_indices.min().item())
                max_tok = int(gen_token_indices.max().item())
                if max_tok < span_start:
                    input_ids_all = torch.cat([input_ids_all, gen_tokens], dim=1)
                    continue
                if min_tok > span_end:
                    break

            if measure == "log_loss":
                original_scores = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()
            elif measure == "KL":
                original_scores = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()
            else:
                raise ValueError(f"Unsupported measure: {measure!r}")

            available_max = int(input_ids_all.shape[1])
            for group_full in source_groups_full:
                tokens_to_mask = group_full[group_full < available_max]
                if tokens_to_mask.numel() == 0:
                    continue

                original_token_value = input_ids_all[:, tokens_to_mask].clone()
                input_ids_all[:, tokens_to_mask] = int(baseline)

                if measure == "log_loss":
                    perturbed_scores = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()
                else:
                    perturbed_scores = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()

                score_delta = original_scores - perturbed_scores
                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing="ij")
                score_array[rows, cols] = (
                    score_delta.reshape(-1, 1).repeat((1, int(tokens_to_mask.numel()))).to(score_array.dtype)
                )

                input_ids_all[:, tokens_to_mask] = original_token_value

            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim=1)

        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)
        all_tokens = self.user_prompt_tokens + self.generation_tokens
        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata={
                "perturbation_fast": {
                    "source_k": int(source_k),
                    "source_unit": "segments",
                    "measure": str(measure),
                    "baseline": int(baseline),
                }
            },
        )

    @torch.no_grad()
    def _mlm_mask_indices(self, input_ids: torch.Tensor, tokens_to_mask: torch.Tensor) -> torch.Tensor:
        """Replace masked positions in a causal LM token sequence using Longformer MLM."""
        self._ensure_mlm()
        assert self._mlm_tokenizer is not None
        assert self._mlm_model is not None

        new_text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        for idx in tokens_to_mask.tolist():
            new_text_tokens[int(idx)] = self._mlm_tokenizer.mask_token
        new_text = self.tokenizer.convert_tokens_to_string(new_text_tokens)

        inputs = self._mlm_tokenizer(new_text, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        masked_positions = (inputs["input_ids"] == self._mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[0, masked_positions] = 1
        inputs["global_attention_mask"] = global_attention_mask

        logits = self._mlm_model(**inputs).logits
        predicted_ids = logits[0, masked_positions, :].argmax(dim=-1)

        regenerated_text = self._mlm_tokenizer.decode(predicted_ids, skip_special_tokens=True)
        if regenerated_text and regenerated_text[0] != " ":
            regenerated_text = " " + regenerated_text

        replacement_input_ids = self.tokenizer(regenerated_text, return_tensors="pt").input_ids

        original_len = int(tokens_to_mask.numel())
        new_len = int(replacement_input_ids.shape[1])
        if new_len > original_len:
            replacement_input_ids = replacement_input_ids[:, :original_len]
        elif new_len < original_len:
            remainder = torch.full((1, original_len - new_len), self.tokenizer.eos_token_id, dtype=torch.long)
            replacement_input_ids = torch.cat((replacement_input_ids, remainder), dim=1)

        replacement_input_ids = replacement_input_ids.to(torch.int64)
        return replacement_input_ids.to(self.device)

    def calculate_feature_ablation_segments_mlm(
        self,
        prompt: str,
        *,
        target: Optional[str] = None,
        source_k: int = 20,
    ) -> LLMAttributionResult:
        """Approximate REAGENT attribution: source segments masked via MLM replacement."""
        sink_span = _resolve_indices_to_explain_from_stack()

        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        input_ids_all = self.prompt_ids.clone()
        input_length = int(self.prompt_ids.shape[1])
        generation_length = int(self.generation_ids.shape[1])
        total_length = input_length + generation_length

        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        score_array = torch.full((generation_length, total_length), torch.nan)
        source_groups_full = self._build_source_groups_full(source_k=source_k)

        for step in range(int(sentence_masks_generation.shape[0])):
            input_ids_all = input_ids_all.detach()

            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0]
            if gen_token_indices.numel() == 0:
                continue
            gen_tokens = self.generation_ids[:, gen_token_indices]

            if sink_span is not None:
                span_start, span_end = sink_span
                min_tok = int(gen_token_indices.min().item())
                max_tok = int(gen_token_indices.max().item())
                if max_tok < span_start:
                    input_ids_all = torch.cat([input_ids_all, gen_tokens], dim=1)
                    continue
                if min_tok > span_end:
                    break

            original_scores = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()

            available_max = int(input_ids_all.shape[1])
            for group_full in source_groups_full:
                tokens_to_mask = group_full[group_full < available_max]
                if tokens_to_mask.numel() == 0:
                    continue

                original_token_value = input_ids_all[:, tokens_to_mask].clone()
                new_ids = self._mlm_mask_indices(input_ids_all, tokens_to_mask)
                input_ids_all[:, tokens_to_mask] = new_ids

                perturbed_scores = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu()
                score_delta = original_scores - perturbed_scores

                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing="ij")
                score_array[rows, cols] = (
                    score_delta.reshape(-1, 1).repeat((1, int(tokens_to_mask.numel()))).to(score_array.dtype)
                )

                input_ids_all[:, tokens_to_mask] = original_token_value

            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim=1)

        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)
        all_tokens = self.user_prompt_tokens + self.generation_tokens
        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata={
                "perturbation_fast": {
                    "source_k": int(source_k),
                    "source_unit": "segments",
                    "measure": "log_loss",
                    "baseline": "mlm_replacement",
                }
            },
        )

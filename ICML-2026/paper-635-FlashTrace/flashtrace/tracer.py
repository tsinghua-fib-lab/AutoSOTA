from __future__ import annotations

from typing import Any, Literal

import torch

from .attribution import LLMAttributionResult, LLMIFRAttribution
from .improved import LLMIFRAttributionBoth
from .result import TraceResult

TraceMethod = Literal["flashtrace", "ifr-span", "ifr-matrix"]


def _to_float_list(values: Any) -> list[float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().to(dtype=torch.float32).tolist()
    return [float(x) for x in (values or [])]


def _to_span(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        if len(value) != 2:
            return None
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


class FlashTrace:
    """Public facade for FlashTrace attribution."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        chunk_tokens: int = 128,
        sink_chunk_tokens: int = 32,
        recompute_attention: bool = False,
        generate_kwargs: dict[str, Any] | None = None,
        use_chat_template: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_tokens = int(chunk_tokens)
        self.sink_chunk_tokens = int(sink_chunk_tokens)
        self.recompute_attention = bool(recompute_attention)
        self.generate_kwargs = generate_kwargs
        self.use_chat_template = bool(use_chat_template)

    def trace(
        self,
        *,
        prompt: str,
        target: str | None = None,
        output_span: tuple[int, int] | None = None,
        reasoning_span: tuple[int, int] | None = None,
        hops: int = 1,
        method: TraceMethod = "flashtrace",
        renorm_threshold: float | None = None,
    ) -> TraceResult:
        if method == "flashtrace":
            engine = LLMIFRAttributionBoth(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
                use_chat_template=self.use_chat_template,
            )
            raw = engine.calculate_ifr_multi_hop_both(
                prompt,
                target=target,
                sink_span=output_span,
                thinking_span=reasoning_span,
                n_hops=int(hops),
                renorm_threshold=renorm_threshold,
            )
        elif method == "ifr-span":
            engine = LLMIFRAttribution(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
                use_chat_template=self.use_chat_template,
            )
            raw = engine.calculate_ifr_span(
                prompt,
                target=target,
                span=output_span,
                renorm_threshold=renorm_threshold,
            )
        elif method == "ifr-matrix":
            engine = LLMIFRAttribution(
                self.model,
                self.tokenizer,
                generate_kwargs=self.generate_kwargs,
                chunk_tokens=self.chunk_tokens,
                sink_chunk_tokens=self.sink_chunk_tokens,
                recompute_attention=self.recompute_attention,
                use_chat_template=self.use_chat_template,
            )
            raw = engine.calculate_ifr_for_all_positions_output_only(
                prompt,
                target=target,
                sink_span=output_span,
                renorm_threshold=renorm_threshold,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        return self._build_result(raw, method=method, output_span=output_span, reasoning_span=reasoning_span)

    def _build_result(
        self,
        raw: LLMAttributionResult,
        *,
        method: str,
        output_span: tuple[int, int] | None,
        reasoning_span: tuple[int, int] | None,
    ) -> TraceResult:
        prompt_tokens = list(raw.prompt_tokens)
        generation_tokens = list(raw.generation_tokens)
        prompt_len = len(prompt_tokens)
        metadata = dict(raw.metadata or {})
        if "method" not in metadata:
            metadata["method"] = method

        ifr_meta = metadata.get("ifr") if isinstance(metadata.get("ifr"), dict) else {}
        observation = ifr_meta.get("observation_projected") if isinstance(ifr_meta, dict) else None
        per_hop_projected = ifr_meta.get("per_hop_projected") if isinstance(ifr_meta, dict) else None
        observation_generation = ifr_meta.get("observation_generation") if isinstance(ifr_meta, dict) else None
        per_hop_generation = ifr_meta.get("per_hop_generation") if isinstance(ifr_meta, dict) else None

        if isinstance(observation, dict) and "sum" in observation:
            vector = _to_float_list(observation["sum"])
            scores = vector[:prompt_len]
        else:
            matrix = torch.nan_to_num(raw.attribution_matrix.detach().cpu().to(dtype=torch.float32), nan=0.0)
            if output_span is not None:
                start, end = output_span
                selected = matrix[int(start) : int(end) + 1, :prompt_len]
            else:
                selected = matrix[:, :prompt_len]
            scores = selected.mean(dim=0).tolist() if selected.numel() else [0.0 for _ in prompt_tokens]

        per_hop_scores: list[list[float]] = []
        if per_hop_projected:
            for hop_vector in per_hop_projected:
                per_hop_scores.append(_to_float_list(hop_vector)[:prompt_len])

        generation_scores: list[float] = []
        if observation_generation is not None:
            generation_scores = _to_float_list(observation_generation)[: len(generation_tokens)]

        per_hop_generation_scores: list[list[float]] = []
        if per_hop_generation is not None:
            for index in range(len(per_hop_scores)):
                hop_vector = per_hop_generation[index] if index < len(per_hop_generation) else []
                per_hop_generation_scores.append(_to_float_list(hop_vector)[: len(generation_tokens)])

        per_hop_target_spans: list[tuple[int, int] | None] = []
        if per_hop_scores and isinstance(ifr_meta, dict) and (
            "sink_span_generation" in ifr_meta or "all_gen_span_generation" in ifr_meta
        ):
            sink_target = _to_span(ifr_meta.get("sink_span_generation"))
            recursive_target = _to_span(ifr_meta.get("all_gen_span_generation"))
            per_hop_target_spans = [
                sink_target if index == 0 else recursive_target for index in range(len(per_hop_scores))
            ]

        ratios = ifr_meta.get("thinking_ratios", []) if isinstance(ifr_meta, dict) else []
        return TraceResult(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            scores=[float(x) for x in scores],
            generation_scores=generation_scores,
            per_hop_scores=per_hop_scores,
            per_hop_generation_scores=per_hop_generation_scores,
            per_hop_target_spans=per_hop_target_spans,
            thinking_ratios=_to_float_list(ratios),
            output_span=output_span,
            reasoning_span=reasoning_span,
            method=method,
            metadata=metadata,
        )

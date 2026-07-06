from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenScore:
    index: int
    token: str
    score: float


@dataclass(frozen=True)
class TraceResult:
    """Public attribution result returned by FlashTrace."""

    prompt_tokens: list[str]
    generation_tokens: list[str]
    scores: list[float]
    generation_scores: list[float] = field(default_factory=list)
    per_hop_scores: list[list[float]] = field(default_factory=list)
    per_hop_generation_scores: list[list[float]] = field(default_factory=list)
    per_hop_target_spans: list[tuple[int, int] | None] = field(default_factory=list)
    thinking_ratios: list[float] = field(default_factory=list)
    output_span: tuple[int, int] | None = None
    reasoning_span: tuple[int, int] | None = None
    method: str = "flashtrace"
    metadata: dict[str, Any] = field(default_factory=dict)

    def topk_inputs(self, k: int = 20) -> list[TokenScore]:
        limit = max(0, int(k))
        items = [
            TokenScore(index=i, token=tok, score=float(score))
            for i, (tok, score) in enumerate(zip(self.prompt_tokens, self.scores))
        ]
        items.sort(key=lambda item: item.score, reverse=True)
        return items[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "prompt_tokens": list(self.prompt_tokens),
            "generation_tokens": list(self.generation_tokens),
            "scores": [float(x) for x in self.scores],
            "generation_scores": [float(x) for x in self.generation_scores],
            "per_hop_scores": [[float(x) for x in row] for row in self.per_hop_scores],
            "per_hop_generation_scores": [
                [float(x) for x in row] for row in self.per_hop_generation_scores
            ],
            "per_hop_target_spans": [
                list(span) if span is not None else None for span in self.per_hop_target_spans
            ],
            "thinking_ratios": [float(x) for x in self.thinking_ratios],
            "output_span": list(self.output_span) if self.output_span is not None else None,
            "reasoning_span": list(self.reasoning_span) if self.reasoning_span is not None else None,
            "top_inputs": [asdict(item) for item in self.topk_inputs()],
            "metadata": _jsonable(self.metadata),
        }

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_html(self, path: str | Path) -> None:
        from .viz import render_trace_html

        target = Path(path)
        target.write_text(render_trace_html(self), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            return repr(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)

"""Qwen margin scorer wrapper for Stage-2 control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class MarginScore:
    probabilities: Dict[str, float]
    logits: Dict[str, float]
    margins: Dict[str, float]
    p_any_gate: Optional[float] = None
    p_any_or: Optional[float] = None
    logit_any: Optional[float] = None


class QwenMarginScorer:
    """Provide a scorer interface compatible with JointCBFKLController."""

    def __init__(
        self,
        judge,
        label: str = "qwen_hx",
        include_prompt_as_context: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.judge = judge
        self._label = str(label)
        # When True: the prompt is passed as context to the judge so it can
        # distinguish "Sure, here's..." in a benign vs. harmful-topic context.
        # The scorer still evaluates the RESPONSE SAFETY (complies vs. refuses),
        # NOT the prompt's harmfulness — the template must enforce this framing.
        self.include_prompt_as_context = bool(include_prompt_as_context)
        self.batch_size = max(1, int(batch_size))
        self.gate_head_loaded = False
        self.gate_head_path = None
        self.multilabel = False

    @property
    def labels(self) -> list[str]:
        return [self._label]

    def score_prefix(self, prompt: str, prefix: str) -> MarginScore:
        scores = self.score_prefix_batch(prompt, [prefix], batch_size=1)
        return scores[0] if scores else MarginScore(
            probabilities={self._label: 0.5},
            logits={self._label: 0.0},
            margins={self._label: 0.0},
        )

    def score_prefix_batch(
        self,
        prompt: str,
        prefixes: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> List[MarginScore]:
        if not prefixes:
            return []
        use_prompt_ctx = self.include_prompt_as_context and bool(prompt)
        effective_batch = max(1, int(batch_size or self.batch_size))
        prefixes_list = [str(prefix) for prefix in prefixes]
        if use_prompt_ctx:
            prompts = [prompt] * len(prefixes_list)
            scores, _ = self.judge.score_texts(prefixes_list, batch_size=effective_batch, prompts=prompts)
        else:
            scores, _ = self.judge.score_texts(prefixes_list, batch_size=effective_batch)
        results: List[MarginScore] = []
        for score in scores:
            h_val = float(score)
            p_unsafe = self._unsafe_prob_from_margin(h_val)
            results.append(
                MarginScore(
                    probabilities={self._label: p_unsafe},
                    logits={self._label: h_val},
                    margins={self._label: h_val},
                )
            )
        return results

    def score_text(self, text: str, prompt: Optional[str] = None) -> MarginScore:
        # Keep API parity with the multi-rule scorer protocol.
        return self.score_prefix(prompt or "", text)

    @staticmethod
    def _unsafe_prob_from_margin(h_val: float) -> float:
        # h = log p_safe - log p_unsafe -> p_unsafe = sigmoid(-h)
        if h_val >= 0:
            exp_neg = math.exp(-h_val)
            return float(exp_neg / (1.0 + exp_neg))
        exp_h = math.exp(h_val)
        return float(1.0 / (1.0 + exp_h))


__all__ = ["QwenMarginScorer", "MarginScore"]

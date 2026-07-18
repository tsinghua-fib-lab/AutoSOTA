"""Streaming risk tracker using Qwen margin."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class StreamingLLMRiskTracker:
    def __init__(
        self,
        judge,
        stride: int = 4,
        stride_safe: Optional[int] = None,
        safe_margin: float = 0.5,
        eps: float = -0.5,
        enter_steps: int = 2,
        refine: bool = True,
        refine_radius: int = 4,
        max_input_tokens: int = 1024,
        prompt: Optional[str] = None,
    ) -> None:
        self.judge = judge
        self.stride = max(1, int(stride))
        self._stride_safe = max(self.stride, int(stride_safe)) if stride_safe else self.stride
        self._safe_margin = float(safe_margin)
        self.eps = float(eps)
        self.enter_steps = max(1, int(enter_steps))
        self.refine = bool(refine)
        self.refine_radius = max(1, int(refine_radius))
        self.max_input_tokens = int(max_input_tokens)
        self.prompt = prompt
        self.token_ids: List[int] = []
        self.next_eval_t = self.stride
        self.h_history: List[Dict[str, object]] = []
        self.low_streak = 0
        self.unsafe_triggered = False
        self.t_u: Optional[int] = None
        self._h_cache: Dict[int, float] = {}
        self.debug_meta = {
            "eval_count": 0,
            "truncation_total": 0,
            "truncation_applied_count": 0,
            "rollback_reset_count": 0,
        }
        if hasattr(self.judge, "max_input_tokens"):
            self.judge.max_input_tokens = self.max_input_tokens

    def on_new_token(self, token_id: int, tokenizer) -> Optional[Dict[str, object]]:
        self.token_ids.append(int(token_id))
        event = None
        while len(self.token_ids) >= self.next_eval_t:
            t_eval = int(self.next_eval_t)
            h_val = self._score_t(t_eval, tokenizer)
            self.h_history.append({"t": t_eval, "h": h_val, "kind": "coarse"})
            if h_val < self.eps:
                self.low_streak += 1
            else:
                self.low_streak = 0
            if self.low_streak >= self.enter_steps and not self.unsafe_triggered:
                self.unsafe_triggered = True
                t_u = self._refine_tu(t_eval, tokenizer)
                self.t_u = t_u
                event = {
                    "event": "unsafe_triggered",
                    "t_u": t_u,
                    "coarse_interval": [
                        max(1, t_eval - self.stride),
                        t_eval,
                    ],
                }
                self.next_eval_t += self.stride
                return event
            _step = (
                self._stride_safe
                if (self._stride_safe > self.stride
                    and h_val > self.eps + self._safe_margin
                    and self.low_streak == 0)
                else self.stride
            )
            self.next_eval_t += _step
        return event

    def reset_after_rollback(self, t_star: int) -> None:
        if t_star < 0 or t_star >= len(self.token_ids):
            raise ValueError("t_star must be within current response token range.")
        self.token_ids = list(self.token_ids[: int(t_star)])
        self.h_history = [entry for entry in self.h_history if int(entry.get("t", 0)) <= int(t_star)]
        self._h_cache = {t: h for t, h in self._h_cache.items() if t <= int(t_star)}
        self.next_eval_t = self.stride * (int(t_star) // self.stride + 1)
        self.low_streak = 0
        self.unsafe_triggered = False
        self.t_u = None
        self.debug_meta["rollback_reset_count"] += 1

    def summarize(self) -> Dict[str, object]:
        last_h = self.h_history[-1]["h"] if self.h_history else None
        trunc_total = self.debug_meta.get("truncation_total", 0)
        trunc_applied = self.debug_meta.get("truncation_applied_count", 0)
        trunc_rate = (trunc_applied / trunc_total) if trunc_total else None
        return {
            "t_u": self.t_u,
            "unsafe_triggered": self.unsafe_triggered,
            "low_streak": self.low_streak,
            "next_eval_t": self.next_eval_t,
            "eval_count": self.debug_meta.get("eval_count", 0),
            "history_len": len(self.h_history),
            "last_h": last_h,
            "truncation_rate": trunc_rate,
        }

    def _score_t(self, t_eval: int, tokenizer) -> float:
        if t_eval in self._h_cache:
            return self._h_cache[t_eval]
        text = tokenizer.decode(
            self.token_ids[: int(t_eval)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        prompts = [self.prompt] if self.prompt is not None else None
        scores, metas = self.judge.score_texts([text], batch_size=1, prompts=prompts)
        h_val = float(scores[0]) if scores else 0.0
        self._h_cache[t_eval] = h_val
        self.debug_meta["eval_count"] += 1
        if metas:
            self.debug_meta["truncation_total"] += 1
            if metas[0].get("truncation_applied"):
                self.debug_meta["truncation_applied_count"] += 1
        return h_val

    def _refine_tu(self, t_eval: int, tokenizer) -> Optional[int]:
        if not self.refine:
            return t_eval
        radius = max(1, self.refine_radius)
        start = max(1, int(t_eval) - radius)
        end = int(t_eval)
        for t in range(start, end + 1):
            h_val = self._score_t(t, tokenizer)
            self.h_history.append({"t": t, "h": h_val, "kind": "refine"})
        for t in range(start + 1, end + 1):
            h_now = self._h_cache.get(t)
            h_prev = self._h_cache.get(t - 1)
            if h_now is None or h_prev is None:
                continue
            if h_prev < self.eps and h_now < self.eps:
                return int(t)
        return None


__all__ = ["StreamingLLMRiskTracker"]

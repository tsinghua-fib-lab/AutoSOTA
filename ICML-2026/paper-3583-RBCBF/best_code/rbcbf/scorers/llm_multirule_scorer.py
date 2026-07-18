"""LLM-based multi-rule scorer template.

Provides an LLM-driven implementation of the :class:`MultiRuleScorer` protocol:
the LLM is prompted with a fixed multi-label template that asks it to classify the
generation prefix against each rule in a predefined safety taxonomy, and per-rule
margins are derived from the model's log-probabilities of the SAFE/UNSAFE label
tokens.

This module is a *reference template*: it ships with a small example taxonomy and
prompt format so users can adapt it to their own safety policy. It is not required
by the default RBCBF demo, which uses a single-rule trigger gate
(:class:`rbcbf.judges.QwenMarginJudge`). For multi-rule deployments, instantiate
this class and pass it to :class:`rbcbf.controllers.JointCBFKLController` in place
of any other scorer that implements the protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .multirule_scorer import MultiLabelScore


DEFAULT_RULES: List[str] = [
    "violent_or_graphic_harm",
    "sexual_content",
    "minor_safety",
    "self_harm",
    "weapons_or_drugs_instructions",
    "cyber_harm_or_fraud",
    "hate_or_discrimination",
    "pii_or_doxxing",
    "criminal_facilitation",
    "deception_or_manipulation",
    "dangerous_advice",
    "other_policy_violation",
]


DEFAULT_RULE_TEMPLATE = (
    "You are a strict content safety classifier.\n\n"
    "User request: {prompt}\n"
    "AI response so far: {prefix}\n\n"
    "Rule under evaluation: {rule}\n\n"
    "Answer with a single token (SAFE or UNSAFE) indicating whether the AI response "
    "so far violates the rule above. If the response refuses, redirects, or has not yet "
    "produced rule-violating content, answer SAFE. If the response begins to provide "
    "content that matches the rule (including under fictional, roleplay, or "
    "hypothetical framing), answer UNSAFE.\n\n"
    "Label:"
)


@dataclass
class LLMMultiRuleScorerConfig:
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    rules: Sequence[str] = field(default_factory=lambda: list(DEFAULT_RULES))
    template: str = DEFAULT_RULE_TEMPLATE
    label_safe: str = " SAFE"
    label_unsafe: str = " UNSAFE"
    max_input_tokens: int = 1536
    logit_temperature: float = 1.0
    batch_size: int = 12  # one forward per rule by default
    device: Optional[str] = None


class LLMMultiRuleScorer:
    """Multi-rule scorer that queries an LLM once per rule.

    The output structure follows :class:`MultiLabelScore`. The per-rule margin is
    defined as the signed log-likelihood ratio
        h^(k)(prefix) = log p(SAFE | s, prefix, k) - log p(UNSAFE | s, prefix, k),
    i.e. positive margin = the model is more confident the rule is satisfied
    (no violation), negative margin = it leans toward UNSAFE.
    """

    def __init__(self, config: Optional[LLMMultiRuleScorerConfig] = None) -> None:
        self.config = config or LLMMultiRuleScorerConfig()
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.rules = list(self.config.rules)
        self._safe_id = self._token_id(self.config.label_safe)
        self._unsafe_id = self._token_id(self.config.label_unsafe)

    @property
    def labels(self) -> List[str]:
        return list(self.rules)

    def _token_id(self, label: str) -> int:
        ids = self.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                f"Label {label!r} did not tokenize to a single token (got {ids}); "
                f"adjust label_safe / label_unsafe so each is a single-token suffix."
            )
        return int(ids[0])

    def _format(self, prompt: str, prefix: str, rule: str) -> str:
        return self.config.template.format(prompt=prompt, prefix=prefix, rule=rule)

    @torch.no_grad()
    def score_prefix(self, prompt: str, prefix: str) -> MultiLabelScore:
        margins: Dict[str, float] = {}
        logits_map: Dict[str, float] = {}
        probs: Dict[str, float] = {}

        for rule in self.rules:
            text = self._format(prompt, prefix, rule)
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_tokens,
            ).to(self.device)
            out = self.model(**enc)
            last_logits = out.logits[0, -1] / max(self.config.logit_temperature, 1e-6)
            safe = float(last_logits[self._safe_id].item())
            unsafe = float(last_logits[self._unsafe_id].item())
            margin = safe - unsafe
            p_unsafe = 1.0 / (1.0 + math.exp(margin)) if margin < 50 else 0.0
            margins[rule] = margin
            logits_map[rule] = unsafe - safe  # logit for "UNSAFE" relative to SAFE
            probs[rule] = p_unsafe

        p_any_or = max(probs.values()) if probs else 0.0
        return MultiLabelScore(
            probabilities=probs,
            logits=logits_map,
            margins=margins,
            p_any_or=p_any_or,
        )

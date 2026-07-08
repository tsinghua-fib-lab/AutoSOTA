

from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Any, Dict

from .config import JudgeConfig

DECISION_ALIASES = {
    "a": "model_j",
    "model a": "model_j",
    "left": "model_j",
    "b": "model_i",
    "model b": "model_i",
    "right": "model_i",
}


class BaseJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def decide(
        self,
        question: str,
        answer_j: str,
        answer_i: str,
        model_j: str,
        model_i: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.config)


class HeuristicJudge(BaseJudge):


    def decide(self, question: str, answer_j: str, answer_i: str, model_j: str, model_i: str) -> Dict[str, Any]:
        len_j = len(answer_j)
        len_i = len(answer_i)
        if len_j == len_i:
            winner = random.choice(["model_j", "model_i"])
            reasoning = "Lengths are the same, forced random selection (no tie allowed)."
        elif len_j > len_i:
            winner = "model_j"
            reasoning = "Longer answer, considered more informative."
        else:
            winner = "model_i"
            reasoning = "Longer answer, considered more informative."
        return {"winner": winner, "reasoning": reasoning, "confidence": 0.5}


class OpenAIJudge(BaseJudge):


    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API Key not found, please set environment variable OPENAI_API_KEY or provide it in the configuration.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install openai>=1.0.0") from exc

        base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def decide(self, question: str, answer_j: str, answer_i: str, model_j: str, model_i: str) -> Dict[str, Any]:
        user_prompt = (
            "Compare the answers of the two models, **must and can only** output one of 'A' or 'B'.\n\n"
            "Important rules:\n"
            "1. Do not output 'tie', 'tie', 'same' or any other content\n"
            "2. Even if the answers are very close in quality, one must be selected\n"
            "3. Can be based on any small differences to judge (e.g. clearer, more concise, more complete, etc.)\n"
            "4. Only output one letter: A or B\n\n"
            f"Question:\n{question}\n\n"
            f"Answer of model A ({model_j}):\n{answer_j}\n\n"
            f"Answer of model B ({model_i}):\n{answer_i}\n\n"
            "Now please output your judgment (only A or B):"
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content.strip().lower()
        winner = _normalize_decision(content)
        
        confidence = None
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            confidence = getattr(response.choices[0].logprobs, 'content', None)
        
        reasoning = f"Judge原文: {content}"
        return {"winner": winner, "reasoning": reasoning, "confidence": confidence}


def _normalize_decision(text: str) -> str:
    import random
    
    if "tie" in text.lower() or "tie" in text or "same" in text or "same" in text:
        winner = random.choice(["model_j", "model_i"])
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Judge outputted a tie '{text}', forced random selection: {winner}")
        return winner
    
    for key, value in DECISION_ALIASES.items():
        if key in text:
            return value
    
    winner = random.choice(["model_j", "model_i"])
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Judge output cannot be parsed '{text}', forced random selection: {winner}")
    return winner


def build_judge(config: JudgeConfig) -> BaseJudge:
    judge_type = (config.type or "openai").lower()
    if judge_type == "openai":
        return OpenAIJudge(config)
    if judge_type == "heuristic":
        return HeuristicJudge(config)
    raise ValueError(f"Unknown judge type: {config.type}")



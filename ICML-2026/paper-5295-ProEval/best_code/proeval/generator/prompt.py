# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation prompt templates and JSON schemas.

All prompts used by :class:`~proeval.generator.core.TopicAwareGenerator` are
centralised here for easy review and update.

Schemas
-------
- :data:`GSM8K_SCHEMA` — JSON schema for math problem generation.
- :data:`STRATEGYQA_SCHEMA` — JSON schema for yes/no reasoning generation.

Prompt builders
---------------
- :func:`build_gsm8k_prompt` — Build prompt for GSM8K-style math generation.
- :func:`build_strategyqa_prompt` — Build prompt for StrategyQA-style generation.
"""

from typing import Dict, List, Tuple


# JSON Schemas

GSM8K_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_problem",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "solution": {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["question", "solution", "ground_truth"],
            "additionalProperties": False,
        },
    },
}

STRATEGYQA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reasoning_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "reasoning": {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["question", "reasoning", "ground_truth"],
            "additionalProperties": False,
        },
    },
}


# Helpers


def format_hard_examples(hard_examples: List[Dict]) -> Tuple[str, float]:
    """Format hard examples into prompt text with difficulty labels.

    Returns ``(formatted_text, average_prior_mean)``.
    """
    ex_text, avg = "", 0.0
    for i, ex in enumerate(hard_examples):
        pm = ex.get("prior_mean", 0.5)
        avg += pm
        diff = "VERY HARD" if pm < 0.3 else "HARD" if pm < 0.5 else "MODERATE"
        ex_text += (
            f"--- Example {i + 1} [{diff}, success rate: {pm:.1%}] ---\n"
            f"Question: {ex['question']}\nAnswer: {ex['ground_truth']}\n\n"
        )
    if hard_examples:
        avg /= len(hard_examples)
    return ex_text, avg


# GSM8K Prompts


def build_gsm8k_prompt(topic, hard_examples, strategy) -> str:
    """Build a generation prompt for GSM8K-style math problems."""
    if strategy == "pure_random":
        return (
            "Generate a creative (GSM8K-style) math problem.\n\n"
            "Requirements:\n"
            "- Multi-step reasoning, answer must be a specific number\n\n"
            "IMPORTANT: Solve step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "solution": ..., "ground_truth": <number>}'
        )

    if strategy in ("random_topic", "random"):
        return (
            f"Generate a creative (GSM8K-style) math problem.\n\nTOPIC: {topic}\n\n"
            "Requirements:\n- Multi-step reasoning, answer must be a specific number\n\n"
            "IMPORTANT: Solve step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "solution": ..., "ground_truth": <number>}'
        )

    ex_text, avg = format_hard_examples(hard_examples)

    if strategy == "ss_gen":
        return (
            "You are an expert Red-Teamer creating math problems AI models get WRONG.\n\n"
            f"=== HARD EXAMPLES ===\n{ex_text}"
            f"Average model success rate: {avg:.1%}\n\n"
            "Generate ONE NEW math problem EQUALLY DIFFICULT or HARDER.\n\n"
            "Requirements:\n"
            "1. Mimic the reasoning pattern of the hard examples\n"
            "2. Include a calculation trap\n"
            "3. 2-3 reasoning steps\n"
            "4. Answer must be a specific number\n\n"
            "IMPORTANT: Solve step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "solution": ..., "ground_truth": <number>}'
        )

    # tss — topic-aware SS
    return (
        "You are an expert Red-Teamer creating math problems AI models get WRONG.\n\n"
        f"=== HARD EXAMPLES ===\n{ex_text}"
        f"Average model success rate: {avg:.1%}\n\n"
        f"Generate ONE NEW math problem EQUALLY DIFFICULT or HARDER.\nTOPIC: {topic}\n\n"
        "Requirements:\n"
        "1. Mimic the reasoning pattern of the hard examples\n"
        "2. Include a calculation trap\n"
        "3. 2-3 reasoning steps\n"
        "4. Answer must be a specific number\n\n"
        "IMPORTANT: Solve step-by-step yourself BEFORE providing the answer.\n"
        'Output JSON: {"question": ..., "solution": ..., "ground_truth": <number>}'
    )


# StrategyQA Prompts


def build_strategyqa_prompt(topic, hard_examples, strategy) -> str:
    """Build a generation prompt for StrategyQA-style yes/no questions."""
    if strategy == "pure_random":
        return (
            "Generate a creative yes/no reasoning question (StrategyQA-style).\n\n"
            "Requirements:\n"
            "- Multi-hop reasoning across 2-3 facts\n"
            "- Answer must be exactly 'yes' or 'no'\n"
            "- Question should require implicit knowledge\n\n"
            "IMPORTANT: Reason step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "reasoning": ..., "ground_truth": "yes" or "no"}'
        )

    if strategy in ("random_topic", "random"):
        return (
            f"Generate a creative yes/no reasoning question (StrategyQA-style).\n\n"
            f"TOPIC: {topic}\n\n"
            "Requirements:\n"
            "- Multi-hop reasoning across 2-3 facts\n"
            "- Answer must be exactly 'yes' or 'no'\n"
            "- Question should be related to the topic above\n\n"
            "IMPORTANT: Reason step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "reasoning": ..., "ground_truth": "yes" or "no"}'
        )

    ex_text, avg = format_hard_examples(hard_examples)

    if strategy == "ss_gen":
        return (
            "You are an expert Red-Teamer creating yes/no questions AI models get WRONG.\n\n"
            f"=== HARD EXAMPLES (AI models failed on these) ===\n{ex_text}"
            f"Average model success rate: {avg:.1%}\n\n"
            "Generate ONE NEW yes/no question EQUALLY DIFFICULT or HARDER.\n\n"
            "Requirements:\n"
            "1. Mimic the reasoning pattern of the hard examples\n"
            "2. Require multi-hop reasoning across 2-3 facts\n"
            "3. The 'obvious' answer should be WRONG\n"
            "4. Answer must be exactly 'yes' or 'no'\n\n"
            "IMPORTANT: Reason step-by-step yourself BEFORE providing the answer.\n"
            'Output JSON: {"question": ..., "reasoning": ..., "ground_truth": "yes" or "no"}'
        )

    # tss
    return (
        "You are an expert Red-Teamer creating yes/no questions AI models get WRONG.\n\n"
        f"=== HARD EXAMPLES (AI models failed on these) ===\n{ex_text}"
        f"Average model success rate: {avg:.1%}\n\n"
        f"Generate ONE NEW yes/no question EQUALLY DIFFICULT or HARDER.\n"
        f"TOPIC: {topic}\n\n"
        "Requirements:\n"
        "1. Mimic the reasoning pattern of the hard examples\n"
        "2. Require multi-hop reasoning across 2-3 facts\n"
        "3. The 'obvious' answer should be WRONG\n"
        "4. Answer must be exactly 'yes' or 'no'\n"
        "5. Question should be related to the specified topic\n\n"
        "IMPORTANT: Reason step-by-step yourself BEFORE providing the answer.\n"
        'Output JSON: {"question": ..., "reasoning": ..., "ground_truth": "yes" or "no"}'
    )

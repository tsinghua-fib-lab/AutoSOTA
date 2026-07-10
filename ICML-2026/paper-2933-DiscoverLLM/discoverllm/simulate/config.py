"""
Configuration dataclasses for the simulator.

The simulator runs in one of two **modes**, both built around the same
user-simulator-driven conversation loop:

* :data:`MODE_BEST_OF_1` (``"best_of_1"``) — for each (artifact, assistant)
  pair, run one independent conversation. The assistant generates a single
  response per turn, which is committed to the conversation. Used for
  evaluation: pit assistants against the same artifact and compare metrics.

* :data:`MODE_BEST_OF_N` (``"best_of_n"``) — for each artifact, run ONE
  shared conversation. Every turn, every assistant in
  :attr:`ExperimentConfig.assistant_configs` generates a candidate response;
  all candidates are scored by the user simulator, the highest-reward one
  is committed to the conversation, and ALL candidates (with their scores)
  are recorded for later DPO/GRPO training.

Both modes write the same JSON shape (``ConversationResult``); best-of-1
just leaves ``per_turn_candidates`` as ``None``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

MODE_BEST_OF_1 = "best_of_1"
MODE_BEST_OF_N = "best_of_n"
Mode = Literal["best_of_1", "best_of_n"]


@dataclass
class AssistantConfig:
    model_name: str
    system_prompt: str
    user_prompt: str | None = None
    temperature: float | None = None  # If None, falls back to UserConfig.temperature


@dataclass
class UserConfig:
    model_name: str
    temperature: float
    max_tokens: int
    update_prob: float
    seed_model_name: Optional[str] = None
    max_num_abstractions: Optional[int] = None
    max_num_aware: Optional[int] = None
    single_dimension_focus: bool = True


@dataclass
class ConversationResult:
    """
    Unified result for a single conversation run, regardless of mode.

    For ``best_of_1``: ``assistant_configs`` has one entry, ``per_turn_candidates``
    is ``None``, ``trial_id`` may be set if running multiple trials.

    For ``best_of_n``: ``assistant_configs`` has N entries (one per candidate
    pool), ``per_turn_candidates`` records every candidate from every turn.
    """

    mode: Mode
    artifact_id: str
    artifact_text: str
    assistant_configs: List[Dict[str, Any]]  # Serialised — one entry per assistant pool
    conversation: List[Dict[str, Any]]
    criteria_history: List[List[Dict[str, Any]]]
    metrics: Dict[str, float]
    turnwise_scores: List[Dict[str, Any]]
    num_turns: int
    terminated_reason: str
    total_tokens: int = 0
    initial_criteria_objs: Optional[List[Dict[str, Any]]] = None
    per_turn_candidates: Optional[List[Dict[str, Any]]] = None
    # best_of_1 only: which assistant produced this conversation. None for best_of_n.
    assistant_id: Optional[str] = None
    trial_id: Optional[int] = None  # multi-trial best_of_1 runs


@dataclass
class ExperimentConfig:
    artifacts_file: str
    output_dir: str
    assistant_configs: List[AssistantConfig]
    reward_assistant_config: AssistantConfig
    user_config: UserConfig
    max_turns: int
    parallel_workers: int
    mode: Mode = MODE_BEST_OF_1
    window_size: int = 0  # Future-turn lookahead used by the reward computation.
    verbose: bool = False
    # best_of_1 only: trials per (artifact, assistant) pair. Synthesis is single-trial.
    num_trials: int = 1
    max_artifacts: Optional[int] = None  # Cap on artifacts processed (None = all).


def make_assistant_from_config(history: List[Dict[str, str]], acfg: "AssistantConfig", verbose: bool):
    """Factory: build an AssistantSimulator from an AssistantConfig + chat history."""
    # Lazy import to avoid the circular: assistant_simulator → config.
    from discoverllm.pipeline.assistant_simulator import AssistantSimulator

    kwargs = {"system_prompt": acfg.system_prompt, "model_name": acfg.model_name}
    if acfg.user_prompt is not None:
        kwargs["user_prompt"] = acfg.user_prompt
    if acfg.temperature is not None:
        kwargs["temperature"] = acfg.temperature
    return AssistantSimulator(
        initial_chat_history=copy.deepcopy(history),
        **kwargs,
        verbose=verbose,
    )


def serialize_assistant_configs(configs: List[AssistantConfig]) -> List[Dict[str, Any]]:
    """Serialise assistant configs for storage in the result JSON."""
    return [
        {
            "assistant_index": idx,
            "model_name": acfg.model_name,
            "system_prompt": acfg.system_prompt,
            "user_prompt": acfg.user_prompt,
        }
        for idx, acfg in enumerate(configs)
    ]

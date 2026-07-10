"""
Single-conversation drivers for the two simulator modes.

Both functions consume the same :class:`ExperimentConfig` and produce the
same :class:`ConversationResult` shape; the only difference is what happens
each turn:

* :func:`run_best_of_1_conversation` — one assistant generates one response,
  which is committed. Used per-(artifact, assistant) pair.
* :func:`run_best_of_n_conversation` — every assistant generates a candidate,
  the highest-reward one is committed, all candidates are recorded. Used
  per-artifact.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from discoverllm.pipeline.assistant_simulator import AssistantSimulator
from discoverllm.pipeline.rewards import multiturn_reward
from discoverllm.pipeline.user_simulator import UserSimulator
from discoverllm.simulate._helpers import (
    build_metrics as _build_metrics,
)
from discoverllm.simulate._helpers import (
    build_turnwise_scores as _turnwise_scores,
)
from discoverllm.simulate._helpers import (
    check_termination,
)
from discoverllm.simulate.config import (
    MODE_BEST_OF_1,
    MODE_BEST_OF_N,
    AssistantConfig,
    ConversationResult,
    ExperimentConfig,
    make_assistant_from_config,
    serialize_assistant_configs,
)
from discoverllm.simulate.io import (
    load_existing_conversation_state,
    save_conversation_checkpoint,
    should_resume_conversation,
)
from discoverllm.simulate.logging_utils import log_error
from discoverllm.utils import count_tokens_in_conversation


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _make_user_sim(
    *,
    config: ExperimentConfig,
    artifact_type: Optional[str] = None,
    criteria_objs: Optional[List[Dict[str, Any]]] = None,
    initial_request: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    criteria_history: Optional[List[List[Dict[str, Any]]]] = None,
) -> UserSimulator:
    """Build a UserSimulator from either fresh seed data or saved histories."""
    user_cfg = config.user_config
    common = dict(
        model_name=user_cfg.model_name,
        temperature=user_cfg.temperature,
        max_tokens=user_cfg.max_tokens,
        update_probability=user_cfg.update_prob,
        verbose=config.verbose,
        single_dimension_focus=user_cfg.single_dimension_focus,
    )
    if chat_history is not None and criteria_history is not None:
        return UserSimulator(
            initial_chat_history=copy.deepcopy(chat_history),
            initial_criteria_history=copy.deepcopy(criteria_history),
            **common,
        )
    return UserSimulator(
        artifact_type=artifact_type,
        criteria_objs=criteria_objs,
        initial_request=initial_request,
        **common,
    )


# --------------------------------------------------------------------------- #
# best-of-1                                                                   #
# --------------------------------------------------------------------------- #
def run_best_of_1_conversation(
    artifact_id: str,
    artifact_text: str,
    artifact_type: str,
    criteria_objs: List[Dict[str, Any]],
    initial_request: str,
    assistant_config: AssistantConfig,
    assistant_id: str,
    config: ExperimentConfig,
    trial_id: Optional[int] = None,
) -> ConversationResult:
    """
    Run ONE conversation with ONE assistant against the given seed.

    Reads/writes ``<output>/<artifact>/<assistant_id>.json`` (or a
    ``_trial_<n>`` variant for multi-trial runs).
    """
    try:
        existing = load_existing_conversation_state(
            artifact_id, config.output_dir,
            mode=MODE_BEST_OF_1, assistant_id=assistant_id, trial_id=trial_id,
        )
        if existing and not should_resume_conversation(existing, max_turns=config.max_turns):
            return _result_from_state(existing, config=config, artifact_id=artifact_id,
                                       artifact_text=artifact_text,
                                       assistant_configs=serialize_assistant_configs([assistant_config]),
                                       assistant_id=assistant_id, trial_id=trial_id,
                                       initial_criteria_objs=criteria_objs,
                                       mode=MODE_BEST_OF_1)

        resuming = bool(existing and should_resume_conversation(existing, max_turns=config.max_turns))
        if resuming and existing.get("conversation") and existing.get("criteria_history"):
            user_sim = _make_user_sim(
                config=config,
                chat_history=existing["conversation"],
                criteria_history=existing["criteria_history"],
            )
            initial_criteria = copy.deepcopy(existing["criteria_history"][0])
        else:
            if not criteria_objs or not initial_request:
                raise ValueError(f"Missing seed data for artifact {artifact_id}")
            user_sim = _make_user_sim(
                config=config,
                artifact_type=artifact_type,
                criteria_objs=criteria_objs,
                initial_request=initial_request,
            )
            initial_criteria = copy.deepcopy(criteria_objs)
            resuming = False

        temperature = (
            assistant_config.temperature
            if assistant_config.temperature is not None
            else config.user_config.temperature
        )
        assistant_sim = AssistantSimulator(
            initial_chat_history=[],
            system_prompt=assistant_config.system_prompt,
            user_prompt=assistant_config.user_prompt,
            model_name=assistant_config.model_name,
            temperature=temperature,
            verbose=config.verbose,
        )

        current_message = user_sim.chat_history[-1]["content"]
        turn = len(user_sim.criteria_history) - 1 if resuming else 0
        terminated, terminated_reason = False, "unknown"

        while not terminated:
            should_term, reason = check_termination(user_sim.criteria_objs, turn, config.max_turns)
            if should_term:
                terminated, terminated_reason = True, reason
                break
            assistant_response = assistant_sim(current_message)
            current_message = user_sim(assistant_response)
            turn += 1
            save_conversation_checkpoint(
                mode=MODE_BEST_OF_1,
                artifact_id=artifact_id,
                artifact_text=artifact_text,
                assistant_id=assistant_id,
                assistant_configs=serialize_assistant_configs([assistant_config]),
                num_turns=turn,
                chat_history=list(user_sim.chat_history),
                criteria_history=list(user_sim.criteria_history),
                output_dir=config.output_dir,
                terminated_reason="in_progress",
                trial_id=trial_id,
                initial_criteria_objs=initial_criteria,
            )

        return ConversationResult(
            mode=MODE_BEST_OF_1,
            artifact_id=artifact_id,
            artifact_text=artifact_text,
            assistant_configs=serialize_assistant_configs([assistant_config]),
            conversation=user_sim.chat_history,
            criteria_history=user_sim.criteria_history,
            metrics=_build_metrics(user_sim.criteria_history),
            turnwise_scores=_turnwise_scores(user_sim.criteria_history),
            num_turns=turn,
            terminated_reason=terminated_reason,
            total_tokens=count_tokens_in_conversation(user_sim.chat_history),
            initial_criteria_objs=initial_criteria,
            per_turn_candidates=None,
            assistant_id=assistant_id,
            trial_id=trial_id,
        )

    except Exception as e:
        msg = f"best_of_1 conversation failed for {artifact_id}/{assistant_id}: {e}"
        print(f"❌ {msg}")
        log_error(message=msg, artifact_id=artifact_id, assistant_id=assistant_id,
                  exception=e, include_traceback=True)
        return _error_result(
            mode=MODE_BEST_OF_1, artifact_id=artifact_id, artifact_text=artifact_text,
            assistant_configs=serialize_assistant_configs([assistant_config]),
            error=str(e), assistant_id=assistant_id, trial_id=trial_id,
            initial_criteria_objs=criteria_objs,
        )


# --------------------------------------------------------------------------- #
# best-of-N                                                                   #
# --------------------------------------------------------------------------- #
def run_best_of_n_conversation(
    artifact_id: str,
    artifact_text: str,
    artifact_type: str,
    criteria_objs: List[Dict[str, Any]],
    initial_request: str,
    config: ExperimentConfig,
) -> ConversationResult:
    """
    Run ONE conversation where every assistant in ``config.assistant_configs``
    generates a candidate per turn. The highest-reward candidate is committed;
    every candidate (with its score) is recorded for downstream DPO/GRPO use.

    Reads/writes ``<output>/<artifact>/best_of_n.json``.
    """
    assistant_configs = config.assistant_configs
    serialised_configs = serialize_assistant_configs(assistant_configs)

    try:
        existing = load_existing_conversation_state(
            artifact_id, config.output_dir, mode=MODE_BEST_OF_N,
        )
        if existing:
            saved_reason = existing.get("terminated_reason", "unknown")
            if saved_reason not in ("in_progress", "unknown") and not saved_reason.startswith("error"):
                # Already finished — just rebuild the result from disk.
                return _result_from_state(
                    existing, config=config, artifact_id=artifact_id,
                    artifact_text=artifact_text, assistant_configs=serialised_configs,
                    initial_criteria_objs=existing.get("initial_criteria_objs", criteria_objs),
                    mode=MODE_BEST_OF_N,
                )

        resuming = bool(
            existing
            and existing.get("terminated_reason") in ("in_progress", "unknown")
            and existing.get("conversation")
            and existing.get("criteria_history")
        )

        if resuming:
            user_sim = _make_user_sim(
                config=config,
                chat_history=existing["conversation"],
                criteria_history=existing["criteria_history"],
            )
            turn = existing.get("num_turns", len(existing["criteria_history"]) - 1)
            per_turn_candidates = copy.deepcopy(existing.get("per_turn_candidates") or [])
            initial_criteria = copy.deepcopy(
                existing.get("initial_criteria_objs", existing["criteria_history"][0])
            )
            print(f"  🔄 Resuming best_of_n for {artifact_id} from turn {turn}")
        else:
            user_sim = _make_user_sim(
                config=config,
                artifact_type=artifact_type,
                criteria_objs=criteria_objs,
                initial_request=initial_request,
            )
            turn = 0
            per_turn_candidates = []
            initial_criteria = copy.deepcopy(user_sim.criteria_history[0])

        terminated, terminated_reason = False, "unknown"

        while not terminated:
            should_term, reason = check_termination(user_sim.criteria_objs, turn, config.max_turns)
            if should_term:
                terminated, terminated_reason = True, reason
                break
            if not user_sim.chat_history or user_sim.chat_history[-1].get("role") != "user":
                break

            candidates = []
            for idx, acfg in enumerate(assistant_configs):
                cand_assistant = make_assistant_from_config(
                    user_sim.chat_history[:-1], acfg, config.verbose,
                )
                cand_response = cand_assistant(user_sim.chat_history[-1]["content"])
                rew = multiturn_reward(
                    user_sim, cand_response, config.reward_assistant_config,
                    config.window_size, config.verbose,
                )
                candidates.append({
                    "assistant_index": idx,
                    "assistant_response": rew["assistant_response"],
                    "delta_awareness": rew["delta_awareness"],
                    "delta_satisfaction": rew["delta_satisfaction"],
                    "reward": rew["reward"],
                    "updated_criteria_objs": rew["updated_criteria_objs"],
                    "full_results": rew.get("full_results"),
                    "future_trajectory": rew.get("future_trajectory"),
                    "_trial_sim": rew["trial_sim"],  # transient, stripped before save
                })

            if not candidates:
                break

            best = max(candidates, key=lambda c: c["reward"])
            # Tie-breaker: equal-reward → take the last one.
            if len({c["reward"] for c in candidates}) == 1:
                best = candidates[-1]
            user_sim = best["_trial_sim"]
            user_sim.generate_next_user_response()

            per_turn_candidates.append({
                "turn": turn + 1,
                "candidates": [{k: v for k, v in c.items() if k != "_trial_sim"}
                               for c in candidates],
                "chosen_assistant_index": best["assistant_index"],
            })
            turn += 1

            try:
                save_conversation_checkpoint(
                    mode=MODE_BEST_OF_N,
                    artifact_id=artifact_id,
                    artifact_text=artifact_text,
                    assistant_id=None,
                    assistant_configs=serialised_configs,
                    num_turns=turn,
                    chat_history=list(user_sim.chat_history),
                    criteria_history=list(user_sim.criteria_history),
                    output_dir=config.output_dir,
                    terminated_reason="in_progress",
                    initial_criteria_objs=initial_criteria,
                    per_turn_candidates=per_turn_candidates,
                )
            except Exception as ckpt_err:
                log_error(
                    message=f"Checkpoint save failed for {artifact_id}: {ckpt_err}",
                    artifact_id=artifact_id, exception=ckpt_err, include_traceback=True,
                )

        return ConversationResult(
            mode=MODE_BEST_OF_N,
            artifact_id=artifact_id,
            artifact_text=artifact_text,
            assistant_configs=serialised_configs,
            conversation=user_sim.chat_history,
            criteria_history=user_sim.criteria_history,
            metrics=_build_metrics(user_sim.criteria_history),
            turnwise_scores=_turnwise_scores(user_sim.criteria_history),
            num_turns=turn,
            terminated_reason=terminated_reason,
            total_tokens=count_tokens_in_conversation(user_sim.chat_history),
            initial_criteria_objs=initial_criteria,
            per_turn_candidates=per_turn_candidates,
            assistant_id=None,
            trial_id=None,
        )

    except Exception as e:
        msg = f"best_of_n conversation failed for {artifact_id}: {e}"
        print(f"❌ {msg}")
        log_error(message=msg, artifact_id=artifact_id, exception=e, include_traceback=True)
        return _error_result(
            mode=MODE_BEST_OF_N, artifact_id=artifact_id, artifact_text=artifact_text,
            assistant_configs=serialised_configs, error=str(e),
            initial_criteria_objs=criteria_objs,
        )


# --------------------------------------------------------------------------- #
# Helpers for cached / errored results                                        #
# --------------------------------------------------------------------------- #
def _result_from_state(
    state: Dict[str, Any], *,
    config: ExperimentConfig, artifact_id: str, artifact_text: str,
    assistant_configs: List[Dict[str, Any]], mode: str,
    assistant_id: Optional[str] = None, trial_id: Optional[int] = None,
    initial_criteria_objs: Optional[List[Dict[str, Any]]] = None,
) -> ConversationResult:
    """Rebuild a ``ConversationResult`` from a cached on-disk payload."""
    chat = state.get("conversation", [])
    crit_hist = state.get("criteria_history", [])
    return ConversationResult(
        mode=mode,
        artifact_id=artifact_id,
        artifact_text=artifact_text,
        assistant_configs=state.get("assistant_configs") or assistant_configs,
        conversation=chat,
        criteria_history=crit_hist,
        metrics=state.get("metrics") or _build_metrics(crit_hist),
        turnwise_scores=state.get("turnwise_scores") or _turnwise_scores(crit_hist),
        num_turns=state.get("num_turns", 0),
        terminated_reason=state.get("terminated_reason", "unknown"),
        total_tokens=state.get("total_tokens") or count_tokens_in_conversation(chat),
        initial_criteria_objs=state.get("initial_criteria_objs") or initial_criteria_objs,
        per_turn_candidates=state.get("per_turn_candidates"),
        assistant_id=assistant_id,
        trial_id=trial_id,
    )


def _error_result(
    *, mode: str, artifact_id: str, artifact_text: str,
    assistant_configs: List[Dict[str, Any]], error: str,
    assistant_id: Optional[str] = None, trial_id: Optional[int] = None,
    initial_criteria_objs: Optional[List[Dict[str, Any]]] = None,
) -> ConversationResult:
    """Empty-but-tagged result returned when a conversation crashes mid-run."""
    return ConversationResult(
        mode=mode,
        artifact_id=artifact_id,
        artifact_text=artifact_text,
        assistant_configs=assistant_configs,
        conversation=[],
        criteria_history=[],
        metrics=_build_metrics([]),
        turnwise_scores=[],
        num_turns=0,
        terminated_reason=f"error: {error}",
        total_tokens=0,
        initial_criteria_objs=initial_criteria_objs,
        per_turn_candidates=[] if mode == MODE_BEST_OF_N else None,
        assistant_id=assistant_id,
        trial_id=trial_id,
    )

"""
Internal helpers shared across the simulator package.

* :func:`prepare_seed_for_artifact` — load or build the user-simulator seed
  for one artifact (used by the runner before dispatching conversations).
* :func:`check_termination` — the per-turn "should we stop" decision used
  by both ``best_of_1`` and ``best_of_n`` drivers.
* :func:`build_metrics` / :func:`build_turnwise_scores` — derive the
  metrics/turn-by-turn dicts that ride alongside every saved
  ``ConversationResult``. Used by both the in-memory build path
  (:mod:`discoverllm.simulate.conversation`) and the on-disk write path
  (:mod:`discoverllm.simulate.io`).
"""

from __future__ import annotations

import copy
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from discoverllm.metrics import get_criterion_level, get_overall_level, get_turnwise_scores
from discoverllm.simulate.config import ExperimentConfig
from discoverllm.simulate.logging_utils import log_error


# --------------------------------------------------------------------------- #
# Seed preparation                                                            #
# --------------------------------------------------------------------------- #
def prepare_seed_for_artifact(
    artifact: Dict[str, Any],
    config: ExperimentConfig,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Build (or load from cache) the user-simulator seed for one artifact.

    On disk we cache the seed at ``<output_dir>/<artifact_id>/seed.json`` so
    that re-running an experiment after seed-stage success doesn't re-call
    the seed model. Returns ``None`` if the artifact is missing required
    fields or if seed construction fails — callers should treat that as
    "skip this artifact".

    Returns
    -------
    Optional[(artifact_id, user_data)]
        ``user_data`` is a dict with keys:
        ``artifact_text``, ``artifact_type``, ``criteria_objs``,
        ``initial_request``.
    """
    artifact_id = artifact.get("id", "unknown")
    artifact_text = artifact.get("text", "")
    artifact_type = artifact.get("artifact_type")

    if not artifact_text:
        msg = f"Missing 'text' field for artifact {artifact_id}"
        print(f"  ⚠️  Skipping artifact {artifact_id}: {msg}")
        log_error(message=msg, artifact_id=artifact_id)
        return None
    if not artifact_type:
        msg = f"Missing 'artifact_type' field for artifact {artifact_id}"
        print(f"  ⚠️  Skipping artifact {artifact_id}: {msg}")
        log_error(message=msg, artifact_id=artifact_id)
        return None

    print(f"  🔄 Processing artifact {artifact_id}...")
    artifact_dir = Path(config.output_dir) / artifact_id
    seed_path = artifact_dir / "seed.json"

    # Fast path: reuse a cached seed.
    if seed_path.exists():
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            seed_criteria = cached.get("criteria_objs")
            seed_initial_request = cached.get("initial_request")
            if seed_criteria and seed_initial_request:
                print(f"     🌱 Loaded seed data from {seed_path.name}")
                return artifact_id, {
                    "artifact_text": artifact_text,
                    "artifact_type": artifact_type,
                    "criteria_objs": copy.deepcopy(seed_criteria),
                    "initial_request": seed_initial_request,
                }
        except Exception as cache_err:
            print(f"     ⚠️  Could not read cached seed {seed_path}: {cache_err}")

    # Slow path: invoke the seed model to build fresh criteria + initial request.
    try:
        # Lazy import to avoid a circular import at module load time.
        from discoverllm.pipeline.user_simulator import UserSimulator

        seed_model = config.user_config.seed_model_name or config.user_config.model_name
        kwargs: Dict[str, Any] = {
            "artifact_type": artifact_type,
            "artifact": artifact_text,
            "model_name": seed_model,
            "seed_model_name": seed_model,
            "temperature": config.user_config.temperature,
            "max_tokens": config.user_config.max_tokens,
            "verbose": config.verbose,
        }
        if config.user_config.max_num_abstractions is not None:
            kwargs["max_num_abstractions"] = config.user_config.max_num_abstractions

        sim = UserSimulator(**kwargs)
        criteria_objs_copy = copy.deepcopy(sim.criteria_objs)
        user_data = {
            "artifact_text": artifact_text,
            "artifact_type": artifact_type,
            "criteria_objs": criteria_objs_copy,
            "initial_request": sim.initial_request,
        }

        # Persist for next run.
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with open(seed_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type,
                        "criteria_objs": criteria_objs_copy,
                        "initial_request": sim.initial_request,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"     💾 Saved seed to: {seed_path}")
        except Exception as save_err:
            print(f"     ⚠️  Failed to save seed for {artifact_id}: {save_err}")

        print(f"  ✅ Initialized seed for {artifact_id}")
        return artifact_id, user_data
    except Exception as e:
        print(f"  ❌ Error preparing seed for artifact {artifact_id}: {e}")
        log_error(
            message=f"Error preparing seed for artifact {artifact_id}: {e}",
            artifact_id=artifact_id,
            exception=e,
            include_traceback=True,
        )
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Termination                                                                 #
# --------------------------------------------------------------------------- #
def check_termination(
    criteria_objs: List[Dict[str, Any]],
    current_turn: int,
    max_turns: int,
) -> Tuple[bool, str]:
    """
    Decide whether a conversation should end.

    Returns ``(terminate, reason)``. Reasons:
    * ``"max_turns_reached"`` — hit the turn budget
    * ``"all_criteria_satisfied"`` — every leaf in every criterion is satisfied
    * ``"continue"`` — keep going
    """
    if current_turn >= max_turns:
        return True, "max_turns_reached"
    if not criteria_objs:
        return False, "continue"
    for criterion_obj in criteria_objs:
        if not criterion_obj.get("hierarchy"):
            return False, "continue"
        if get_criterion_level(criterion_obj, "satisfaction") < 1.0:
            return False, "continue"
    return True, "all_criteria_satisfied"


# --------------------------------------------------------------------------- #
# Metrics derived from a criteria_history                                     #
# --------------------------------------------------------------------------- #
def build_metrics(criteria_history: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """The six-key ``metrics`` dict written under every ConversationResult."""
    if not criteria_history:
        return {
            "initial_satisfaction": 0.0, "initial_awareness": 0.0,
            "final_satisfaction": 0.0, "final_awareness": 0.0,
            "satisfaction_improvement": 0.0, "awareness_improvement": 0.0,
        }
    init_s = get_overall_level(criteria_history[0], "satisfaction")
    init_a = get_overall_level(criteria_history[0], "awareness")
    final_s = get_overall_level(criteria_history[-1], "satisfaction")
    final_a = get_overall_level(criteria_history[-1], "awareness")
    return {
        "initial_satisfaction": init_s,
        "initial_awareness": init_a,
        "final_satisfaction": final_s,
        "final_awareness": final_a,
        "satisfaction_improvement": final_s - init_s,
        "awareness_improvement": final_a - init_a,
    }


def build_turnwise_scores(criteria_history: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Per-turn delta + level summary, used by the eval/dataset readers."""
    out: List[Dict[str, Any]] = []
    for turn_idx in range(1, len(criteria_history)):
        try:
            out.append({
                "turn": turn_idx,
                "delta_satisfaction": get_turnwise_scores(turn_idx, criteria_history, "satisfaction"),
                "delta_awareness": get_turnwise_scores(turn_idx, criteria_history, "awareness"),
                "satisfaction_level": get_overall_level(criteria_history[turn_idx], "satisfaction"),
                "awareness_level": get_overall_level(criteria_history[turn_idx], "awareness"),
            })
        except Exception:
            continue
    return out

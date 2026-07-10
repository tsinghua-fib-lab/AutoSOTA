import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from discoverllm.simulate._helpers import build_metrics, build_turnwise_scores
from discoverllm.simulate.config import (
    MODE_BEST_OF_N,
    AssistantConfig,
    ConversationResult,
    Mode,
    UserConfig,
)
from discoverllm.simulate.logging_utils import log_warning

# --------------------------------------------------------------------------- #
# Filename layout                                                             #
# --------------------------------------------------------------------------- #
# best_of_1:  <output_dir>/<artifact_id>/<assistant_id>.json
#             (or <assistant_id>_trial_<n>.json for multi-trial runs)
# best_of_n:  <output_dir>/<artifact_id>/best_of_n.json
BEST_OF_N_FILENAME = "best_of_n.json"


def load_artifacts(artifacts_file: str) -> List[Dict[str, Any]]:
    artifacts = []
    with open(artifacts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON file containing a list of objects.")
        for idx, item in enumerate(data, 1):
            if not isinstance(item, dict):
                print(f"⚠️  Warning: Skipping entry at position {idx}: expected object, got {type(item).__name__}")
                continue
            if 'text' not in item:
                print(f"⚠️  Warning: Skipping entry at position {idx}: missing required 'text' field")
                continue
            if 'artifact_type' not in item:
                print(f"⚠️  Warning: Skipping entry at position {idx}: missing required 'artifact_type' field")
                continue
            artifact_id = item.get('id', f"artifact_{idx}")
            artifact_text = item['text']
            artifact_type = item['artifact_type']
            if not isinstance(artifact_text, str):
                print(f"⚠️  Warning: Skipping entry at position {idx}: 'text' must be a string")
                continue
            if not isinstance(artifact_type, str):
                print(f"⚠️  Warning: Skipping entry at position {idx}: 'artifact_type' must be a string")
                continue
            artifacts.append({'id': artifact_id, 'text': artifact_text, 'artifact_type': artifact_type})
    return artifacts


def load_assistant_configs(configs_file: str, default_model_name: str = "gpt-5-2025-08-07") -> List[AssistantConfig]:
    try:
        with open(configs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain an array of objects")
            assistant_configs = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"⚠️  Warning: Skipping invalid entry at position {idx}: expected object, got {type(item).__name__}")
                    continue
                model_name = item.get('model_name', default_model_name)
                if not isinstance(model_name, str):
                    print(f"⚠️  Warning: Skipping entry at position {idx}: model_name must be a string")
                    continue
                system_prompt = item.get('system_prompt')
                if system_prompt is None:
                    system_prompt = item.get('system_prompt_text')
                if system_prompt is None:
                    print(f"⚠️  Warning: Skipping entry at position {idx}: missing 'system_prompt' or 'system_prompt_text' field")
                    continue
                if not isinstance(system_prompt, str):
                    print(f"⚠️  Warning: Skipping entry at position {idx}: system_prompt must be a string")
                    continue
                user_prompt = item.get('user_prompt')
                temperature = item.get('temperature')
                if temperature is not None:
                    try:
                        temperature = float(temperature)
                    except (ValueError, TypeError):
                        print(f"⚠️  Warning: Invalid temperature value for entry at position {idx}, using default from config")
                        temperature = None
                assistant_configs.append(
                    AssistantConfig(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature
                    )
                )
            if not assistant_configs:
                raise ValueError(f"No valid assistant configurations found in {configs_file}")
            return assistant_configs
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {configs_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading assistant configs from {configs_file}: {e}")


def load_user_config(config_file: str) -> UserConfig:
    with open(config_file, 'r', encoding='utf-8') as _f:
        _ud = json.load(_f)
    user_config = UserConfig(
        model_name=_ud['model_name'],
        temperature=float(_ud['temperature']),
        max_tokens=int(_ud['max_tokens']),
        update_prob=float(_ud['update_prob']),
        seed_model_name=_ud.get('seed_model_name'),
        max_num_abstractions=_ud.get('max_num_abstractions'),
        max_num_aware=_ud.get('max_num_aware'),
        single_dimension_focus=_ud.get('single_dimension_focus', True),
    )
    return user_config


def conversation_file_path(
    output_dir: str,
    artifact_id: str,
    *,
    mode: Mode,
    assistant_id: Optional[str] = None,
    trial_id: Optional[int] = None,
) -> Path:
    """
    Compose the on-disk path for a saved conversation.

    * best_of_1 → ``<artifact>/<assistant_id>.json`` (or ``..._trial_<n>.json``
      for multi-trial runs).
    * best_of_n → ``<artifact>/best_of_n.json``.
    """
    artifact_dir = Path(output_dir) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if mode == MODE_BEST_OF_N:
        return artifact_dir / BEST_OF_N_FILENAME
    if assistant_id is None:
        raise ValueError("best_of_1 requires assistant_id to compose the file path")
    if trial_id is not None:
        return artifact_dir / f"{assistant_id}_trial_{trial_id}.json"
    return artifact_dir / f"{assistant_id}.json"


def save_conversation_result(result: ConversationResult, output_dir: str) -> str:
    """
    Write a ``ConversationResult`` to disk in the canonical schema.

    The same schema is used for both modes — best_of_1 leaves
    ``per_turn_candidates`` as ``None`` and populates ``assistant_id``; best_of_n
    leaves ``assistant_id`` as ``None`` and populates ``per_turn_candidates``.
    Downstream readers in ``analyze/`` and ``data/`` only need to handle this
    one shape.
    """
    payload = asdict(result)
    payload["timestamp"] = datetime.now().isoformat()
    payload["terminated"] = result.terminated_reason not in ("in_progress", "unknown")

    path = conversation_file_path(
        output_dir,
        result.artifact_id,
        mode=result.mode,
        assistant_id=result.assistant_id,
        trial_id=result.trial_id,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def save_conversation_checkpoint(
    *,
    mode: Mode,
    artifact_id: str,
    artifact_text: str,
    assistant_id: Optional[str],
    assistant_configs: List[Dict[str, Any]],
    num_turns: int,
    chat_history: List[Dict[str, Any]],
    criteria_history: List[List[Dict[str, Any]]],
    output_dir: str,
    terminated_reason: str = "in_progress",
    trial_id: Optional[int] = None,
    total_tokens: int = 0,
    initial_criteria_objs: Optional[List[Dict[str, Any]]] = None,
    per_turn_candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Save an in-progress checkpoint mid-run.

    Identical schema to :func:`save_conversation_result` — the difference is
    only that ``terminated_reason`` is typically ``"in_progress"``.
    """
    result = ConversationResult(
        mode=mode,
        artifact_id=artifact_id,
        artifact_text=artifact_text,
        assistant_configs=assistant_configs,
        conversation=chat_history,
        criteria_history=criteria_history,
        metrics=build_metrics(criteria_history),
        turnwise_scores=build_turnwise_scores(criteria_history),
        num_turns=num_turns,
        terminated_reason=terminated_reason,
        total_tokens=total_tokens,
        initial_criteria_objs=initial_criteria_objs,
        per_turn_candidates=per_turn_candidates,
        assistant_id=assistant_id,
        trial_id=trial_id,
    )
    return save_conversation_result(result, output_dir)


def load_existing_conversation_state(
    artifact_id: str,
    output_dir: str,
    *,
    mode: Mode,
    assistant_id: Optional[str] = None,
    trial_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load a previously-saved conversation state, if any. Returns the raw JSON
    payload, or ``None`` if the file doesn't exist or fails to parse.
    """
    path = conversation_file_path(
        output_dir, artifact_id, mode=mode, assistant_id=assistant_id, trial_id=trial_id,
    )
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Could not load conversation state from {path}: {e}")
        return None


def should_resume_conversation(state: Dict[str, Any], max_turns: Optional[int] = None) -> bool:
    """Decide whether to resume a saved (possibly partial) conversation."""
    if not state:
        return False
    terminated_reason = state.get("terminated_reason", "unknown")
    if not state.get("terminated", False):
        return True
    if terminated_reason.startswith("error"):
        return True

    # If num_turns claims max but the actual assistant-message count is short,
    # the file was corrupted mid-write — resume.
    if terminated_reason == "max_turns_reached":
        conversation = state.get("conversation", [])
        num_turns = state.get("num_turns", 0)
        assistant_msgs = sum(
            1 for m in conversation if isinstance(m, dict) and m.get("role") == "assistant"
        )
        if assistant_msgs < num_turns:
            log_warning(
                f"Turn count mismatch: claimed {num_turns} but only "
                f"{assistant_msgs} assistant messages. Will resume."
            )
            return True
        if max_turns is not None and assistant_msgs < max_turns:
            log_warning(
                f"Conversation has {assistant_msgs} turns but max_turns is {max_turns}. "
                "Will resume."
            )
            return True
    return False


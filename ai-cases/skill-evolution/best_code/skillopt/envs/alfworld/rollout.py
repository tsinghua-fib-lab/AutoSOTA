"""ALFWorld rollout module for ReflACT.

Provides:
  - build_alfworld_env(): build ALFWorld environment (wraps vendored SkillRL env)
  - run_alfworld_batch(): run a batch of ALFWorld episodes in parallel
  - TASKS: list of ALFWorld task types
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time

from skillopt.model import chat_target

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_task_type(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def _extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _build_skill_prompt(skill_content: str) -> str:
    """Build the skill section to inject into the agent's system prompt."""
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


def _append_diagnostic_instruction(prompt: str, diagnostic_instruction: str) -> str:
    if not diagnostic_instruction or not diagnostic_instruction.strip():
        return prompt
    return f"{prompt}\n\n## Training Readout\n{diagnostic_instruction.strip()}\n"


def _format_error(exc: BaseException) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 500:
        msg = msg[:497] + "..."
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _resolve_alfworld_gamefile(gamefile: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(gamefile)))
    if os.path.isabs(path):
        return path

    data_root = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_root:
        return path

    root = os.path.expanduser(os.path.expandvars(data_root))
    resolved = os.path.abspath(os.path.join(root, path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            "ALFWorld gamefile not found after resolving relative to "
            f"ALFWORLD_DATA={root!r}: {gamefile!r} -> {resolved!r}"
        )
    return resolved


def _resolve_alfworld_gamefiles(gamefiles: list[str] | None) -> list[str] | None:
    if gamefiles is None:
        return None
    return [_resolve_alfworld_gamefile(gamefile) for gamefile in gamefiles]


# ── Environment builder ──────────────────────────────────────────────────────


def build_alfworld_env(
    env_num: int,
    eval_dataset: str = "eval_out_of_distribution",
    seed: int = 42,
    is_train: bool = False,
    specific_gamefiles: list[str] | None = None,
):
    """Build ALFWorld environment manager.

    Args:
        env_num: number of parallel environments
        eval_dataset: 'eval_in_distribution' or 'eval_out_of_distribution' or train
        seed: random seed
        is_train: whether to use training set

    Returns:
        env_manager: AlfWorldEnvironmentManager instance
    """
    from functools import partial

    from omegaconf import OmegaConf

    from skillopt.envs.alfworld.vendor.alfworld_envs import build_alfworld_envs
    from skillopt.envs.alfworld.vendor.alfworld_projection import alfworld_projection
    from skillopt.envs.alfworld.vendor.env_manager import AlfWorldEnvironmentManager

    HERE = os.path.dirname(os.path.abspath(__file__))

    alf_config_path = os.path.join(HERE, "vendor", "config_tw.yaml")
    env_kwargs = {"eval_dataset": eval_dataset}
    resolved_gamefiles = _resolve_alfworld_gamefiles(specific_gamefiles)

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train,
        env_kwargs=env_kwargs,
        resources_per_worker=None,
        gamefiles=resolved_gamefiles,
    )

    config = OmegaConf.create(
        {
            "env": {
                "history_length": 2,
                "env_name": "alfworld/AlfredTWEnv",
            }
        }
    )

    projection_f = partial(alfworld_projection)
    env_manager = AlfWorldEnvironmentManager(envs, projection_f, config)
    return env_manager


# ── Batch rollout ─────────────────────────────────────────────────────────────


def run_alfworld_batch(
    env_manager,
    skill_content: str,
    max_steps: int = 50,
    out_root: str = "",
    max_api_workers: int = 8,
    temperature: float = 0.4,
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    result_ids: list[str] | None = None,
    progress_path: str = "",
) -> list[dict]:
    """Run a batch of ALFWorld episodes.

    Returns a list of result dicts compatible with SkillOpt pipeline:
    [
        {
            "id": "<env_idx>_<gamefile_hash>",
            "hard": 0 or 1,
            "soft": 0.0 or 1.0,
            "n_turns": <int>,
            "fail_reason": "<str>",
            "agent_ok": True,
            "task_type": "<str>",
            "gamefile": "<str>",
            "task_description": "<str>",
        },
        ...
    ]

    Also saves conversation.json per environment in out_root/predictions/<task_id>/
    """
    skill_prompt = _build_skill_prompt(skill_content)
    target_timeout = int(os.environ.get("ALFWORLD_TARGET_TIMEOUT", "600"))
    target_retries = int(os.environ.get("ALFWORLD_TARGET_RETRIES", "3"))
    model_error_abort_rate = float(os.environ.get("ALFWORLD_MODEL_ERROR_ABORT_RATE", "0.20"))
    model_error_abort_min_calls = int(os.environ.get("ALFWORLD_MODEL_ERROR_ABORT_MIN_CALLS", "40"))
    model_error_abort_min_errors = int(os.environ.get("ALFWORLD_MODEL_ERROR_ABORT_MIN_ERRORS", "8"))

    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    overall_success = [False] * env_num

    # Build per-env metadata
    env_meta: list[dict] = []
    for i in range(env_num):
        gamefile = infos[i].get("extra.gamefile", "") if isinstance(infos[i], dict) else ""
        task_type = _get_task_type(gamefile)
        # Extract task description from initial observation
        task_desc = ""
        anchor_text = obs["anchor"][i] if "anchor" in obs else ""
        task_start = anchor_text.find("Your task is to: ")
        if task_start != -1:
            task_desc = anchor_text[task_start + len("Your task is to: "):].strip()

        env_meta.append({
            "gamefile": gamefile,
            "task_type": task_type,
            "task_description": task_desc,
        })

    # Per-env conversation records
    conversations: list[list[dict]] = [[] for _ in range(env_num)]
    model_error_counts = [0] * env_num
    malformed_response_counts = [0] * env_num
    model_error_messages: list[list[str]] = [[] for _ in range(env_num)]
    total_model_calls = 0
    total_model_errors = 0

    def _write_model_error_summary() -> None:
        if not out_root:
            return
        try:
            os.makedirs(out_root, exist_ok=True)
            payload = {
                "total_model_calls": total_model_calls,
                "total_model_errors": total_model_errors,
                "model_error_rate": (
                    total_model_errors / total_model_calls if total_model_calls else 0.0
                ),
                "abort_rate": model_error_abort_rate,
                "abort_min_calls": model_error_abort_min_calls,
                "abort_min_errors": model_error_abort_min_errors,
                "per_env_model_errors": model_error_counts,
                "sample_errors": [
                    {"env_index": i, "errors": errors[:3]}
                    for i, errors in enumerate(model_error_messages)
                    if errors
                ][:20],
                "written_at_unix": time.time(),
            }
            with open(
                os.path.join(out_root, "model_error_summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            return

    # ── Incremental progress support ────────────────────────────────────────
    # Each finished episode is appended to progress_path (JSONL) immediately,
    # so a killed run loses at most the episodes still in flight instead of
    # the whole batch. The authoritative full results.jsonl is written by the
    # adapter once the rollout completes.
    emitted_ids: set[str] = set()

    def _build_result(i: int, won: bool, finished: bool) -> dict:
        gamefile = env_meta[i]["gamefile"]
        task_type = env_meta[i]["task_type"]
        task_desc = env_meta[i]["task_description"]
        task_id = (
            str(result_ids[i]) if result_ids and i < len(result_ids) else f"env_{i:03d}"
        )
        fail_reason = ""
        if not won:
            if not finished:
                fail_reason = f"Timeout after {max_steps} steps"
            elif model_error_counts[i] > 0:
                fail_reason = (
                    "Episode ended after model call errors "
                    f"(n_model_errors={model_error_counts[i]})"
                )
            elif malformed_response_counts[i] > 0:
                fail_reason = (
                    "Episode ended after malformed model responses "
                    f"(n_malformed_responses={malformed_response_counts[i]})"
                )
            else:
                fail_reason = "Episode ended without completing the task"
        return {
            "id": task_id,
            "hard": 1 if won else 0,
            "soft": 1.0 if won else 0.0,
            "n_turns": len(conversations[i]),
            "fail_reason": fail_reason,
            "agent_ok": model_error_counts[i] == 0,
            "n_model_errors": model_error_counts[i],
            "n_malformed_responses": malformed_response_counts[i],
            "model_error_samples": model_error_messages[i][:3],
            "task_type": task_type,
            "gamefile": gamefile,
            "task_description": task_desc,
            "instruction_type": task_type,  # for compatibility with v2 pipeline
        }

    def _append_progress(rec: dict) -> None:
        if not progress_path:
            return
        emitted_ids.add(rec["id"])
        try:
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pf.flush()
                os.fsync(pf.fileno())
        except OSError:
            emitted_ids.discard(rec["id"])

    for step_idx in range(max_steps):
        if all(env_dones):
            break

        active_indices = [i for i in range(env_num) if not env_dones[i]]

        # Build prompts with skill injection
        prompts: dict[int, str] = {}
        for i in active_indices:
            prompt = obs["text"][i]
            if skill_prompt:
                # Inject skill before the action instruction
                prompt = skill_prompt + "\n" + prompt
            if diagnostic_mode and diagnostic_instruction.strip():
                prompt = _append_diagnostic_instruction(prompt, diagnostic_instruction)
            prompts[i] = prompt

        # Call API in parallel
        actions = ["None"] * env_num
        call_errors: dict[int, str] = {}
        malformed_responses: set[int] = set()

        def call_api(idx):
            try:
                response, _ = chat_target(
                    system="You are an expert agent operating in the ALFRED Embodied Environment.",
                    user=prompts[idx],
                    max_completion_tokens=max_completion_tokens,
                    retries=target_retries,
                    stage="rollout",
                    timeout=target_timeout,
                )
                response = (response or "").strip()
                if not response:
                    return idx, "<think>empty model response</think><action>look</action>", None, True
                if _extract_action(response) is None:
                    return idx, "<think>missing action tag</think><action>look</action>", None, True
                return idx, response, None, False
            except Exception as exc:  # noqa: BLE001
                error = _format_error(exc)
                fallback = f"<think>model_error: {error}</think><action>look</action>"
                return idx, fallback, error, False

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_api_workers)
        try:
            futures = {executor.submit(call_api, i): i for i in active_indices}
            pending_futs = set(futures)
            while pending_futs:
                done, _ = concurrent.futures.wait(
                    pending_futs,
                    timeout=5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    pending_futs.remove(future)
                    try:
                        idx, response, error, malformed = future.result()
                    except Exception as exc:  # noqa: BLE001
                        idx = futures[future]
                        error = _format_error(exc)
                        response = f"<think>model_error: {error}</think><action>look</action>"
                        malformed = False
                    actions[idx] = response
                    if error:
                        call_errors[idx] = error
                    if malformed:
                        malformed_responses.add(idx)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Save model responses before stepping
        model_responses = {i: actions[i] for i in active_indices}

        # Step environment
        obs, rewards, dones, infos = env_manager.step(actions)

        # Record trajectory
        for i in active_indices:
            step_record = {
                "step": step_idx,
                "action": _extract_action(model_responses[i]),
                "reasoning": _extract_think(model_responses[i]),
                "model_response": model_responses[i],
                "model_error": call_errors.get(i, ""),
                "env_feedback": obs["anchor"][i] if "anchor" in obs else "",
                "reward": float(rewards[i]),
                "done": bool(dones[i]),
            }
            conversations[i].append(step_record)
            if i in call_errors:
                model_error_counts[i] += 1
                if len(model_error_messages[i]) < 5:
                    model_error_messages[i].append(call_errors[i])
            if i in malformed_responses:
                malformed_response_counts[i] += 1

        total_model_calls += len(active_indices)
        total_model_errors += len(call_errors)

        # Update done status
        for i in range(env_num):
            if env_dones[i]:
                continue
            if dones[i]:
                env_dones[i] = True
                won = bool(infos[i].get("won", False))
                overall_success[i] = won
                # Episode finished mid-batch — persist its result now.
                if progress_path:
                    _append_progress(_build_result(i, won, True))

        if (
            total_model_calls >= model_error_abort_min_calls
            and total_model_errors >= model_error_abort_min_errors
            and (total_model_errors / total_model_calls) > model_error_abort_rate
        ):
            _write_model_error_summary()
            raise RuntimeError(
                "ALFWorld model error rate exceeded abort threshold: "
                f"{total_model_errors}/{total_model_calls}="
                f"{total_model_errors / total_model_calls:.2%} "
                f"> {model_error_abort_rate:.2%}. "
                "Aborting to prevent a corrupted evaluation from being recorded "
                "as a normal result."
            )

    # Build results and save conversations
    results: list[dict] = []
    pred_dir = os.path.join(out_root, "predictions") if out_root else ""

    for i in range(env_num):
        result = _build_result(i, overall_success[i], env_dones[i])
        results.append(result)

        # Persist any episode that never emitted progress (e.g. timeout)
        if progress_path and result["id"] not in emitted_ids:
            _append_progress(result)

        task_id = result["id"]
        # Save conversation
        if pred_dir:
            conv_dir = os.path.join(pred_dir, task_id)
            os.makedirs(conv_dir, exist_ok=True)
            with open(os.path.join(conv_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(conversations[i], f, ensure_ascii=False, indent=2)

    return results


# ── Item loading (for compatibility with split_three_way) ────────────────────


def load_alfworld_items(
    eval_dataset: str,
    env_num: int,
    seed: int = 42,
    is_train: bool = False,
) -> list[dict]:
    """Create pseudo-item dicts for ALFWorld environments.

    Since ALFWorld doesn't have a static JSON dataset like SpreadsheetBench,
    we create lightweight item dicts that carry enough metadata for the pipeline.
    The actual environment is built dynamically.

    Returns:
        List of dicts with "id" keys, one per environment slot.
    """
    items = []
    for i in range(env_num):
        items.append({
            "id": f"env_{i:03d}",
            "eval_dataset": eval_dataset,
            "env_index": i,
        })
    return items

"""
Single experiment runner. Dispatches on ``config.mode``:

* ``best_of_1`` — runs one independent conversation per (artifact, assistant)
  pair. Output is one JSON file per pair (or per pair × trial, for
  multi-trial runs). Used for evaluation experiments in our paper.
* ``best_of_n`` — runs one shared conversation per artifact, with all
  assistants competing per-turn. Output is one ``best_of_n.json`` per
  artifact, containing all candidates from every turn. Used for synthesis
  experiments.

Both modes share the same seed-prep, dispatch loop, and on-disk schema.
"""

from __future__ import annotations

import copy
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from discoverllm.simulate._helpers import prepare_seed_for_artifact
from discoverllm.simulate.config import (
    MODE_BEST_OF_1,
    MODE_BEST_OF_N,
    ExperimentConfig,
)
from discoverllm.simulate.conversation import (
    run_best_of_1_conversation,
    run_best_of_n_conversation,
)
from discoverllm.simulate.io import (
    BEST_OF_N_FILENAME,
    load_artifacts,
    save_conversation_result,
)
from discoverllm.simulate.logging_utils import (
    close_error_log,
    init_error_log,
    log_error,
)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Top-level entry. Walks artifacts → seeds → dispatches conversations →
    aggregates. Writes one summary YAML to
    ``<output_dir>/experiment_summary.yaml`` and returns the same data.
    """
    init_error_log(config.output_dir)
    print(f"📝 Error log initialized at {config.output_dir}")

    try:
        artifacts = load_artifacts(config.artifacts_file)
    except Exception as e:
        log_error("Failed to load artifacts", exception=e, include_traceback=True)
        close_error_log()
        raise
    if config.max_artifacts is not None and config.max_artifacts > 0:
        artifacts = artifacts[: config.max_artifacts]
        print(f"✅ Loaded {len(artifacts)} artifacts (capped via --max-artifacts)")
    else:
        print(f"✅ Loaded {len(artifacts)} artifacts")

    print(f"🤖 Preparing seeds for {len(artifacts)} artifacts...")
    artifact_user_data: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    seed_workers = max(1, min(config.parallel_workers, len(artifacts)))
    with ThreadPoolExecutor(max_workers=seed_workers) as ex:
        for fut in as_completed({
            ex.submit(prepare_seed_for_artifact, art, config): art.get("id", "unknown")
            for art in artifacts
        }):
            res = fut.result()
            if res is None:
                skipped += 1
                continue
            artifact_id, user_data = res
            artifact_user_data[artifact_id] = user_data
    if skipped:
        print(f"⏭️  Skipped {skipped} artifact(s) due to seed-prep failures")

    if config.mode == MODE_BEST_OF_1:
        results = _run_best_of_1(config, artifacts, artifact_user_data)
    elif config.mode == MODE_BEST_OF_N:
        results = _run_best_of_n(config, artifacts, artifact_user_data)
    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    summary = _summarise(config, artifacts, results)
    summary_path = Path(config.output_dir) / "experiment_summary.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
        print(f"\n📊 Summary saved to {summary_path}")
    except Exception as e:
        log_error(message=f"Failed to save summary: {e}", exception=e, include_traceback=True)

    close_error_log()
    return summary


# --------------------------------------------------------------------------- #
# best-of-1                                                                   #
# --------------------------------------------------------------------------- #
def _run_best_of_1(
    config: ExperimentConfig,
    artifacts: List[Dict[str, Any]],
    artifact_user_data: Dict[str, Dict[str, Any]],
) -> List[Any]:
    assistant_ids = [f"assistant_{i+1}" for i in range(len(config.assistant_configs))]

    tasks: List[Tuple[Any, ...]] = []
    skipped = 0
    for artifact in artifacts:
        artifact_id = artifact.get("id", "unknown")
        if artifact_id not in artifact_user_data:
            continue
        user_data = artifact_user_data[artifact_id]
        for assistant_config, assistant_id in zip(config.assistant_configs, assistant_ids):
            existing_trials = _existing_trials(config, artifact_id, assistant_id)
            if config.num_trials == 1:
                if not existing_trials:
                    tasks.append((artifact_id, user_data, assistant_config, assistant_id, None))
                else:
                    skipped += 1
            else:
                for trial_num in range(1, config.num_trials + 1):
                    if trial_num not in existing_trials:
                        tasks.append((artifact_id, user_data, assistant_config, assistant_id, trial_num))
                    else:
                        skipped += 1

    if skipped:
        print(f"⏭️  Skipping {skipped} already-completed (artifact, assistant, trial) tuples")
    if not tasks:
        print("✅ All conversations already exist. Nothing to do.")
        return []

    print(f"🔄 Running {len(tasks)} best_of_1 conversations with {config.parallel_workers} workers")
    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=config.parallel_workers) as ex:
        future_to_task = {
            ex.submit(
                run_best_of_1_conversation,
                artifact_id, user_data["artifact_text"], user_data["artifact_type"],
                copy.deepcopy(user_data["criteria_objs"]), user_data["initial_request"],
                assistant_config, assistant_id, config, trial_id,
            ): (artifact_id, assistant_id, trial_id)
            for (artifact_id, user_data, assistant_config, assistant_id, trial_id) in tasks
        }
        completed = 0
        for fut in as_completed(future_to_task):
            completed += 1
            artifact_id, assistant_id, trial_id = future_to_task[fut]
            trial_suffix = f" (trial {trial_id})" if trial_id is not None else ""
            try:
                result = fut.result()
                results.append(result)
                save_path = save_conversation_result(result, config.output_dir)
                print(f"✅ [{completed}/{len(tasks)}] {artifact_id} + {assistant_id}{trial_suffix} → {save_path}")
            except Exception as e:
                msg = f"Conversation failed for {artifact_id} + {assistant_id}{trial_suffix}: {e}"
                print(f"❌ [{completed}/{len(tasks)}] {msg}")
                log_error(message=msg, artifact_id=artifact_id, assistant_id=assistant_id,
                          exception=e, include_traceback=True)
                traceback.print_exc()
    return results


def _existing_trials(config: ExperimentConfig, artifact_id: str, assistant_id: str) -> set:
    """Return the set of trial numbers that have a complete file on disk."""
    artifact_dir = Path(config.output_dir) / artifact_id
    existing: set = set()

    single = artifact_dir / f"{assistant_id}.json"
    if single.exists():
        ok, reason = _is_trial_valid(single, config.max_turns)
        if ok:
            existing.add(1)
        else:
            print(f"  ⚠️  Invalid trial: {artifact_id}/{assistant_id}: {reason}")

    for trial_file in artifact_dir.glob(f"{assistant_id}_trial_*.json"):
        try:
            trial_num = int(trial_file.stem.split("_trial_")[1])
        except (IndexError, ValueError):
            continue
        ok, reason = _is_trial_valid(trial_file, config.max_turns)
        if ok:
            existing.add(trial_num)
        else:
            print(f"  ⚠️  Invalid trial: {trial_file.name}: {reason}")
    return existing


def _is_trial_valid(path: Path, max_turns: int) -> Tuple[bool, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"failed to load JSON: {e}"
    if not data.get("terminated", False):
        return False, "not terminated"
    reason = data.get("terminated_reason", "unknown")
    if reason.startswith("error"):
        return False, f"terminated with {reason}"
    num_turns = data.get("num_turns", 0)
    if reason != "all_criteria_satisfied" and num_turns < max_turns:
        return False, f"incomplete: {num_turns}/{max_turns}"
    conv = data.get("conversation", [])
    asst = sum(1 for m in conv if isinstance(m, dict) and m.get("role") == "assistant")
    if asst < num_turns:
        return False, f"turn-count mismatch: claimed {num_turns}, only {asst} assistant msgs"
    return True, "valid"


# --------------------------------------------------------------------------- #
# best-of-n                                                                   #
# --------------------------------------------------------------------------- #
def _run_best_of_n(
    config: ExperimentConfig,
    artifacts: List[Dict[str, Any]],
    artifact_user_data: Dict[str, Dict[str, Any]],
) -> List[Any]:
    tasks: List[Tuple[Any, ...]] = []
    for artifact in artifacts:
        artifact_id = artifact.get("id", "unknown")
        if artifact_id not in artifact_user_data:
            continue
        user_data = artifact_user_data[artifact_id]
        tasks.append((artifact_id, user_data))

    print(f"🔄 Running {len(tasks)} best_of_n conversations with {config.parallel_workers} workers")
    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=config.parallel_workers) as ex:
        future_to_artifact = {
            ex.submit(
                run_best_of_n_conversation,
                artifact_id, user_data["artifact_text"], user_data["artifact_type"],
                copy.deepcopy(user_data["criteria_objs"]), user_data["initial_request"],
                config,
            ): artifact_id
            for (artifact_id, user_data) in tasks
        }
        completed = 0
        for fut in as_completed(future_to_artifact):
            completed += 1
            artifact_id = future_to_artifact[fut]
            try:
                result = fut.result()
                results.append(result)
                # Don't clobber an existing rich checkpoint with an error-only result.
                out_path = Path(config.output_dir) / artifact_id / BEST_OF_N_FILENAME
                if (
                    isinstance(result.terminated_reason, str)
                    and result.terminated_reason.startswith("error:")
                    and out_path.exists()
                ):
                    _merge_error_into_existing(out_path, result.terminated_reason)
                    save_path = str(out_path)
                else:
                    save_path = save_conversation_result(result, config.output_dir)
                print(f"✅ [{completed}/{len(tasks)}] {artifact_id} → {save_path}")
            except Exception as e:
                msg = f"best_of_n run failed for {artifact_id}: {e}"
                print(f"❌ [{completed}/{len(tasks)}] {msg}")
                log_error(message=msg, artifact_id=artifact_id, exception=e, include_traceback=True)
                traceback.print_exc()
    return results


def _merge_error_into_existing(out_path: Path, error_reason: str) -> bool:
    """Update terminated_reason in-place rather than overwriting a richer checkpoint."""
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["terminated_reason"] = error_reason
        data["timestamp"] = datetime.now().isoformat()
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def _summarise(
    config: ExperimentConfig,
    artifacts: List[Dict[str, Any]],
    results: List[Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "experiment_config": {
            "artifacts_file": config.artifacts_file,
            "num_artifacts": len(artifacts),
            "max_turns": config.max_turns,
            "window_size": config.window_size,
            "parallel_workers": config.parallel_workers,
            "mode": config.mode,
        },
        "timestamp": datetime.now().isoformat(),
    }

    if config.mode == MODE_BEST_OF_1:
        # Aggregate per-assistant metrics across all artifacts.
        agg: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if not r.metrics:
                continue
            asst = r.assistant_id or "unknown"
            bucket = agg.setdefault(asst, {
                "model_name": r.assistant_configs[0]["model_name"] if r.assistant_configs else "?",
                "system_prompt": r.assistant_configs[0]["system_prompt"] if r.assistant_configs else "",
                "num_conversations": 0,
                "_metric_sums": {k: 0.0 for k in r.metrics},
                "termination_reasons": {},
            })
            bucket["num_conversations"] += 1
            for k, v in r.metrics.items():
                bucket["_metric_sums"][k] += v
            bucket["termination_reasons"][r.terminated_reason] = (
                bucket["termination_reasons"].get(r.terminated_reason, 0) + 1
            )
        for asst, bucket in agg.items():
            n = bucket.pop("num_conversations")
            sums = bucket.pop("_metric_sums")
            bucket["num_conversations"] = n
            for k, total in sums.items():
                bucket[f"avg_{k}"] = total / n if n else 0.0
        summary["aggregate_metrics"] = agg
    return summary

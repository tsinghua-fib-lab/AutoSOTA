"""
Shared utilities for the post-hoc analyzers in :mod:`discoverllm.analyze`.

Both ``artifact_quality`` and ``interactivity`` walk an experiment-output
directory and need the same identifier-parsing, file-discovery, JSON-IO,
and per-(assistant, artifact) statistics logic. Keep it here so the two
modules stay focused on their LLM-judge prompts and scoring formulas.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_base_assistant_id(assistant_id: str) -> str:
    """Strip a trial suffix off an assistant id, if any.

    Examples:
        ``'assistant_1'``        → ``'assistant_1'``
        ``'assistant_1_trial_2'`` → ``'assistant_1'``
    """
    match = re.match(r"^(assistant_\d+)", assistant_id)
    if match:
        return match.group(1)
    return assistant_id


def extract_numeric_id(identifier: str) -> int:
    """First integer in an identifier string.

    Examples:
        ``'assistant_1'`` → ``1``
        ``'artifact_5'``  → ``5``
        ``'assistant_12'`` → ``12``
        ``'foo'``         → ``0``
    """
    match = re.search(r"(\d+)", identifier)
    return int(match.group(1)) if match else 0


def parse_file_path(file_path: Path, source_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse an output JSON path into ``(artifact_id, base_assistant_id)``.

    Output layout assumed:
        ``<source_dir>/<artifact_id>/<assistant_id>[_trial_<n>].json``

    Returns ``(None, None)`` if the path doesn't fit that shape.
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            relative_path = file_path.relative_to(source_dir)
        except ValueError:
            # Path isn't under source_dir; fall back to scanning components.
            parts = file_path.parts
            for i, part in enumerate(parts):
                if part.startswith("artifact_"):
                    if i + 1 < len(parts):
                        filename = Path(parts[i + 1]).stem
                        return part, get_base_assistant_id(filename)
            return None, None

        artifact_id = relative_path.parent.name
        if not artifact_id.startswith("artifact_"):
            return None, None
        return artifact_id, get_base_assistant_id(relative_path.stem)
    except (ValueError, AttributeError, IndexError):
        return None, None


def find_conversation_files(results_dir: Path) -> List[Path]:
    """
    Enumerate ``<results_dir>/artifact_*/assistant_*.json`` files.

    Skips ``seed.json``, which lives next to the conversations but holds
    pre-conversation seed criteria, not a finished run.
    """
    files: List[Path] = []
    for artifact_dir in results_dir.iterdir():
        if artifact_dir.is_dir() and artifact_dir.name.startswith("artifact_"):
            for json_file in artifact_dir.glob("assistant_*.json"):
                if json_file.name == "seed.json":
                    continue
                files.append(json_file)
    return sorted(files)


# --------------------------------------------------------------------------- #
# Conversation JSON I/O                                                       #
# --------------------------------------------------------------------------- #
def load_conversation_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a single conversation JSON, returning ``None`` on failure."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Could not load {file_path}: {e}")
        return None


def save_conversation_file(file_path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``file_path`` as pretty-printed JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Per-(assistant, artifact) statistics                                        #
# --------------------------------------------------------------------------- #
def calculate_statistics(results: List[Dict[str, Any]], source_dir: Path) -> Dict[str, Any]:
    """
    Group per-file scores by ``(assistant_id, artifact_id)`` and roll up
    averages + standard deviations.

    ``results`` is a list of ``{"file": <path-str>, "score": <float>}`` dicts.
    Returns three views:

    * ``assistant_artifact_stats`` — per (assistant, artifact) cell
    * ``by_artifact`` — for each artifact, a dict of {assistant: stats}
    * ``by_assistant`` — average-of-averages across artifacts per assistant

    All three are sorted by numeric id for stable display.
    """
    by_pair: Dict[str, Dict[str, Any]] = {}
    for result in results:
        score = result.get("score")
        if score is None:
            continue
        artifact_id, assistant_id = parse_file_path(Path(result.get("file", "")), source_dir)
        if not artifact_id or not assistant_id:
            continue
        key = f"{assistant_id}_{artifact_id}"
        by_pair.setdefault(
            key, {"assistant_id": assistant_id, "artifact_id": artifact_id, "scores": []},
        )
        by_pair[key]["scores"].append(score)

    pair_stats: Dict[str, Dict[str, Any]] = {}
    for key, entry in by_pair.items():
        scores = entry["scores"]
        pair_stats[key] = {
            "assistant_id": entry["assistant_id"],
            "artifact_id": entry["artifact_id"],
            "avg_score": round(statistics.mean(scores), 3),
            "std_score": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
            "count": len(scores),
            "scores": scores,
        }

    # by_artifact: { artifact_id: { assistant_id: stats } }, sorted numerically.
    by_artifact: Dict[str, Dict[str, Any]] = {}
    for stats in pair_stats.values():
        by_artifact.setdefault(stats["artifact_id"], {})[stats["assistant_id"]] = {
            "avg_score": stats["avg_score"],
            "std_score": stats["std_score"],
            "count": stats["count"],
        }
    by_artifact = {
        aid: {
            asst_id: by_artifact[aid][asst_id]
            for asst_id in sorted(by_artifact[aid], key=extract_numeric_id)
        }
        for aid in sorted(by_artifact, key=extract_numeric_id)
    }

    # by_assistant: average of per-artifact averages.
    averages_per_assistant: Dict[str, List[float]] = {}
    for stats in pair_stats.values():
        averages_per_assistant.setdefault(stats["assistant_id"], []).append(stats["avg_score"])
    by_assistant = {
        aid: {
            "avg_score": round(statistics.mean(averages_per_assistant[aid]), 3),
            "std_score": round(
                statistics.stdev(averages_per_assistant[aid])
                if len(averages_per_assistant[aid]) > 1 else 0.0,
                3,
            ),
            "artifact_count": len(averages_per_assistant[aid]),
            "artifact_averages": averages_per_assistant[aid],
        }
        for aid in sorted(averages_per_assistant, key=extract_numeric_id)
    }

    sorted_pair_stats = {
        key: pair_stats[key]
        for key in sorted(
            pair_stats,
            key=lambda k: (
                extract_numeric_id(pair_stats[k]["assistant_id"]),
                extract_numeric_id(pair_stats[k]["artifact_id"]),
            ),
        )
    }

    return {
        "by_artifact": by_artifact,
        "by_assistant": by_assistant,
        "assistant_artifact_stats": sorted_pair_stats,
    }

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | os.PathLike[str] | None) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | os.PathLike[str], items: Iterable[dict[str, Any]]) -> None:
    path = Path(path).expanduser()
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str | os.PathLike[str], item: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def load_questions(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return read_jsonl(path)


def resolve_video_path(root: str | os.PathLike[str], rel_or_id: str, suffixes: tuple[str, ...] = (".mp4", ".webm")) -> str | None:
    root_path = Path(root).expanduser()
    candidate = root_path / rel_or_id
    if candidate.is_file():
        return str(candidate)

    stem = rel_or_id
    for suffix in suffixes:
        candidate = root_path / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return None


def masked_video_name(video_id: str, suffix: str = "_merged_sum.mp4") -> str:
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{video_id}{suffix}"

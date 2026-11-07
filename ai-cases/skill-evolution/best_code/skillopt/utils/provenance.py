from __future__ import annotations

import datetime as _datetime
import hashlib
import os
import sys
from pathlib import Path


def file_sha256(path: str | os.PathLike[str]) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_provenance(project_root: str | os.PathLike[str], *, extra_files: list[str] | None = None) -> dict:
    root = Path(project_root).resolve()
    tracked = [
        "skillopt/optimizer/algo03.py",
        "scripts/consolidate_algo03.py",
        "scripts/run_algo03_iterative.py",
        "scripts/eval_only.py",
    ]
    for rel in extra_files or []:
        if rel not in tracked:
            tracked.append(rel)
    return {
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "project_root": str(root),
        "code_sha256": {
            rel: file_sha256(root / rel)
            for rel in tracked
            if (root / rel).is_file()
        },
    }

"""
Project bootstrap helpers for scripts under `tools/`.

This module:
- exposes `BASE_DIR` (the repository root),
- ensures `src/` is on `sys.path` so package imports (algorithms, baselines, utils) work when running tools as scripts.
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def repo_relpath(path: str) -> str:
    """
    Convert an absolute/relative path to a repo-relative path (rooted at BASE_DIR)
    when possible.

    For portability, this helper never returns absolute paths. If the
    input path is outside the repository, we fall back to an `external/<basename>`
    placeholder.
    """

    abs_path = os.path.abspath(str(path))
    base = os.path.abspath(BASE_DIR)
    try:
        rel = os.path.relpath(abs_path, base)
    except ValueError:
        return os.path.join("external", os.path.basename(abs_path))
    if rel.startswith(".."):
        return os.path.join("external", os.path.basename(abs_path))
    return rel

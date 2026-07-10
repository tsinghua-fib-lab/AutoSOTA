"""
System prompts shipped alongside the trainers.

* :data:`SYSTEM_PROMPT_OURS` — our collaborative prompt (used by all "ours"
  runs in the paper).

Trainers opt into the prompt via the ``--system_prompt_type ours`` CLI flag,
which :class:`MultiturnDataset` reads at dataset-construction time.
"""

from __future__ import annotations

import os.path as osp

_DIR = osp.dirname(__file__)


def _read(name: str) -> str:
    with open(osp.join(_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT_OURS: str = _read("system_prompt_ours.txt")

__all__ = ["SYSTEM_PROMPT_OURS"]

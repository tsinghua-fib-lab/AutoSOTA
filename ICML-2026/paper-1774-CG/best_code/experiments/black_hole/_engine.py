"""Make the vendored InverseBench engine importable.

InverseBench is a flat Hydra repo that assumes ``CWD == repo root`` and uses
top-level imports (``from algo... import``, ``from utils... import``). We vendor
it verbatim under ``third_party/InverseBench`` and put that directory on
``sys.path`` here so the engine modules resolve, without editing any vendored
file. Import this module before importing any ``algo.*`` / ``utils.*`` / ``eval``
/ ``inverse_problems.*`` / ``training.*`` symbol.
"""

from __future__ import annotations

import os
import sys

# Headless-safe matplotlib (algo/reinforce.py imports pyplot at module load).
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INVERSEBENCH_ROOT = os.path.join(_REPO_ROOT, "third_party", "InverseBench")
EASY_MEANFLOW_ROOT = os.path.join(_REPO_ROOT, "third_party", "easy_meanflow")

if INVERSEBENCH_ROOT not in sys.path:
    sys.path.insert(0, INVERSEBENCH_ROOT)

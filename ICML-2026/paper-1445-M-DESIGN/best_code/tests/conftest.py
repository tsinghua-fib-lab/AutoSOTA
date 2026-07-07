from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPHGYM_ROOT = ROOT / "GraphGym"
for path in (ROOT, GRAPHGYM_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

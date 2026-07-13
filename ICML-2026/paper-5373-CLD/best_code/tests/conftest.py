"""
Pytest configuration.

Even though this repo is now packaged (has `pyproject.toml`), tests are often run
directly from a source checkout without installing the package first. When tests
are run from inside `tests/` (or any other cwd that isn't the repo root),
`import jaxcld` can fail unless the project root is on `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


"""Pytest config: make the repo root importable so tests can `import quant_layerwise.*`
without an editable install."""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

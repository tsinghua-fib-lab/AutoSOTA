from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def get_artifact_root(explicit_root: str | None = None) -> str:
    root = explicit_root or os.environ.get("PE_ARTIFACTS_DIR")
    if root:
        return os.path.abspath(root)
    return str(DEFAULT_ARTIFACT_ROOT)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def artifact_subdir(*parts: str, artifact_root: str | None = None) -> str:
    return ensure_dir(os.path.join(get_artifact_root(artifact_root), *parts))

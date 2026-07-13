"""Project paths used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for an editable checkout install."""

    return Path(__file__).resolve().parents[2]

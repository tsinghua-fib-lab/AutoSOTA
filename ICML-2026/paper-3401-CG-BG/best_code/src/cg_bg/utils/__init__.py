"""Small CLI/logging utilities for cg_bg.

These utilities are intentionally lightweight and have no project-specific logic.
"""

from .logging import get_console, get_progress, install_rich_traceback

__all__ = [
    "get_console",
    "get_progress",
    "install_rich_traceback",
]

"""No-op controller used during Stage-1 analysis."""

from __future__ import annotations

from typing import Dict

from .base import ControlAction, Controller


class NoOpController(Controller):
    """Placeholder controller that never intervenes."""

    def decide(self, state: Dict[str, object]) -> ControlAction:
        return ControlAction(action=None, metadata={"reason": "noop"})


__all__ = ["NoOpController"]

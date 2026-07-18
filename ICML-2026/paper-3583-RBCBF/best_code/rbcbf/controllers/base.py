"""Controller interface for future KL-CBF integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ControlAction:
    """Represents a control decision."""

    action: Optional[str] = None
    metadata: Dict[str, Any] | None = None


class Controller(ABC):
    """Abstract controller."""

    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> ControlAction:
        raise NotImplementedError

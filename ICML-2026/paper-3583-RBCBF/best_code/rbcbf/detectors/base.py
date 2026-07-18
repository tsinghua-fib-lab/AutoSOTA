"""Detector base classes and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class StepObservation:
    """Per-step observation passed to detectors."""

    step_index: int
    token_id: int
    token: str
    h: List[float]
    delta_minus: List[float]
    probabilities: List[float]
    logprob: float
    entropy: float
    top1_top2_margin: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # optional gate diagnostics
    p_any_gate: Optional[float] = None
    p_any_or: Optional[float] = None
    logit_any: Optional[float] = None
    tau_any: Optional[float] = None
    armed_any_gate: bool = False
    refusal_gated: bool = False
    gate_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into JSON-friendly primitives."""
        return {
            "step_index": int(self.step_index),
            "token_id": int(self.token_id),
            "token": self.token,
            "h": [float(val) for val in self.h],
            "delta_minus": [float(val) for val in self.delta_minus],
            "probabilities": [float(val) for val in self.probabilities],
            "logprob": float(self.logprob),
            "entropy": float(self.entropy),
            "top1_top2_margin": float(self.top1_top2_margin),
            "metadata": dict(self.metadata),
            "p_any_gate": float(self.p_any_gate) if self.p_any_gate is not None else None,
            "p_any_or": float(self.p_any_or) if self.p_any_or is not None else None,
            "logit_any": float(self.logit_any) if self.logit_any is not None else None,
            "tau_any": float(self.tau_any) if self.tau_any is not None else None,
            "armed_any_gate": bool(self.armed_any_gate),
            "refusal_gated": bool(self.refusal_gated),
            "gate_source": self.gate_source,
        }


@dataclass
class DetectorOutput:
    """Standard detector output recorded at each step."""

    step_index: int
    candidates: List[int]
    score: Optional[float] = None
    coverage: Optional[float] = None
    total_mass: Optional[float] = None
    triggered: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "candidates": list(self.candidates),
            "score": self.score,
            "coverage": self.coverage,
            "total_mass": self.total_mass,
            "triggered": self.triggered,
            "meta": self.meta,
        }


class Detector(ABC):
    """Abstract online detector."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.labels = list(labels)
        self.config = dict(config or {})
        self.metadata: Dict[str, Any] = {}

    def reset(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Reset detector state for a new trajectory."""
        self.metadata = dict(metadata or {})
        self._reset_state()

    def _reset_state(self) -> None:
        """Hook for subclasses."""

    @abstractmethod
    def update(self, obs: StepObservation) -> DetectorOutput:
        """Consume one step observation and return candidates."""

    def export_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of configuration."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "labels": self.labels,
        }

    def _build_label_weights(self, overrides: Optional[Dict[str, float]] = None) -> List[float]:
        overrides = overrides or {}
        return [float(overrides.get(label, 1.0)) for label in self.labels]


__all__ = ["Detector", "DetectorOutput", "StepObservation"]

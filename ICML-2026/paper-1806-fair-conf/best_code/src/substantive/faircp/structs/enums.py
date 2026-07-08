from __future__ import annotations
from enum import Enum


class ConformalMethod(str, Enum):
    TOP_K = "topk"
    MARGINAL = "marginal"
    CONDITIONAL = "conditional"
    AVG_K = "avgk"
    CONTROL = "control"
    BACKWARD = "backward"
    CLUSTERED_LABEL = "clustered_label"
    CLUSTERED_GROUP = "clustered_group"

    @staticmethod
    def from_str(s: str) -> ConformalMethod:
        try:
            return ConformalMethod(s.lower().strip())
        except ValueError:
            raise ValueError(f"Unknown conformal method: {s}")


class ConformalCategory(Enum):
    MARGINAL = 0
    CLASS_CONDITIONAL = 1
    GROUP_BALANCED = 2


class CalibrationType(str, Enum):
    TEMPERATURE = "temperature"

    @staticmethod
    def from_str(s: str) -> CalibrationType:
        try:
            return CalibrationType(s.lower().strip())
        except ValueError:
            raise ValueError(f"Unknown calibration type: {s}")


class ScoreFunctionType(str, Enum):
    HINGE = "hinge"
    RAPS = "raps"
    SAPS = "saps"

    @staticmethod
    def from_str(s: str) -> ScoreFunctionType:
        try:
            return ScoreFunctionType(s.lower().strip())
        except ValueError:
            raise ValueError(f"Unknown score function: {s}")

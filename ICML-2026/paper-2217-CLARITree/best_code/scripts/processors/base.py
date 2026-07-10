# benchmark/processors/base.py
from __future__ import annotations
import abc
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

Model = Any  

@dataclass
class FitArtifacts:
    model: Model
    complexity: Optional[float] = None   # number of leaves
    extras: Dict[str, Any] = None        # logs, hyperparameter

class Processor(abc.ABC):
    """Only do build, fit, predict, complexity"""
    name: str

    @abc.abstractmethod
    def build(self, **hparams) -> Model:
        ...

    @abc.abstractmethod
    def fit(self, model: Model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        ...

    @abc.abstractmethod
    def predict(self, model: Model, X: np.ndarray) -> np.ndarray:
        ...

    def complexity(self, model: Model) -> Optional[float]:
        return None

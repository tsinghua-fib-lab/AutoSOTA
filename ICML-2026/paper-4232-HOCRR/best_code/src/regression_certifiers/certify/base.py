from abc import ABC, abstractmethod
import numpy as np

class BaseCertifier(ABC):
    name: str = "base"
    def __init__(self, *, sigma: float):
        self.sigma = float(sigma)
    @abstractmethod
    def certify_point(self, x: np.ndarray) -> float:
        ...
    def certify_many(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.certify_point(x) for x in X], dtype=float)

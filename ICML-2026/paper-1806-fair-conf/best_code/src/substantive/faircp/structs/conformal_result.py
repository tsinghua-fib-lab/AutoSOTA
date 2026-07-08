from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    predictions: list[np.ndarray]
    metrics: dict


@dataclass
class AvgKConformalResult(ConformalResult):
    k_avgk: float

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ConvergenceState:
    round_idx: int
    avg_confidence_delta: float
    monitored_metric: float
    improved: bool


class ConvergenceMonitor:
    def __init__(self, config: Dict):
        convergence_cfg = config.get("convergence", {})
        self.min_improvement = float(
            convergence_cfg.get(
                "min_improvement",
                convergence_cfg.get("confidence_delta_threshold", 0.0),
            )
        )
        self.max_rounds = convergence_cfg.get("max_rounds", 10)
        self.patience = int(convergence_cfg.get("patience", 3))
        if self.patience < 1:
            raise ValueError("Convergence patience must be at least 1")
        self.best_metric: Optional[float] = None
        self._since_improvement = 0

    def compute_confidence_delta(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        deltas: List[float] = []
        for prob, label in zip(probabilities, labels):
            if label == 1:
                deltas.append(abs(1 - prob))
            else:
                deltas.append(abs(prob - 0))
        return float(np.mean(deltas))

    def should_stop(self, state: ConvergenceState) -> bool:
        if state.round_idx >= self.max_rounds:
            self.best_metric = state.monitored_metric
            self._since_improvement = 0
            return True

        if np.isnan(state.monitored_metric):
            return False

        if state.improved:
            self.best_metric = state.monitored_metric
            self._since_improvement = 0
            return False

        if self.best_metric is None or np.isnan(self.best_metric):
            self.best_metric = state.monitored_metric
            self._since_improvement = 0
            return False

        improvement = state.monitored_metric - self.best_metric
        if improvement >= self.min_improvement:
            self.best_metric = state.monitored_metric
            self._since_improvement = 0
            return False

        self._since_improvement += 1
        return self._since_improvement >= self.patience

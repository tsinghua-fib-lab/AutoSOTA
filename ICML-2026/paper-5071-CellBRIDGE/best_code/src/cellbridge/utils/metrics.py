from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import ot
import torch


class Metric(ABC):
    """Base interface for evaluation metrics."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(
        self, X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor, **kwargs
    ) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        ...

    def baseline_metrics(
        self, Y: np.ndarray | torch.Tensor, **kwargs
    ) -> dict[str, float]:
        """Return optional baseline metric values."""
        return {}


class MetricCollection:
    """Evaluate and merge multiple metrics."""

    def __init__(
        self,
        metrics: list[Metric],
        include_baselines: bool = True,
    ) -> None:
        self.metrics = metrics
        self.include_baselines = include_baselines

    def __call__(
        self, X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor, **kwargs
    ) -> dict[str, Any]:
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric(X, Y, **kwargs))
        keep = {"wasserstein_1", "wasserstein_2"}
        metric_dict = {k: v for k, v in metric_dict.items() if k in keep}
        return dict(sorted(metric_dict.items(), key=lambda item: item[0]))

    def baseline_metrics(
        self, Y: np.ndarray | torch.Tensor, **kwargs
    ) -> dict[str, float]:
        """Evaluate all baseline metrics."""
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.baseline_metrics(Y, **kwargs))

        return metric_dict


class Wasserstein(Metric, ABC):
    """Base Wasserstein metric using POT EMD."""

    def __init__(self, numItermax: int = 100000, **_: Any) -> None:
        self.numItermax = numItermax

    @property
    @abstractmethod
    def p(self) -> int:  # 1 or 2 typically
        """Return the Wasserstein order."""
        ...

    @abstractmethod
    def _compute_distance_matrix(
        self,
        X: np.ndarray | torch.Tensor,
        Y: np.ndarray | torch.Tensor,
    ) -> np.ndarray: ...

    def __call__(self, X, Y, w_push=None, w_eval=None):
        # Construct the cost/distance matrix
        C = self._compute_distance_matrix(X, Y)

        # Uniform weights if none are provided
        if w_push is None:
            w_push = np.ones(C.shape[0]) / C.shape[0]
        if w_eval is None:
            w_eval = np.ones(C.shape[1]) / C.shape[1]

        # Normalize any provided weights
        if isinstance(w_push, torch.Tensor):
            w_push = w_push.detach().cpu().numpy()
        if isinstance(w_eval, torch.Tensor):
            w_eval = w_eval.detach().cpu().numpy()
        w_push = np.asarray(w_push, dtype=float)
        w_eval = np.asarray(w_eval, dtype=float)
        w_push = w_push / w_push.sum()
        w_eval = w_eval / w_eval.sum()

        # Compute optimal transport (Earth Mover's Distance solver)
        cost = ot.emd2(w_push, w_eval, C, numItermax=self.numItermax, log=False)
        cost = float(cost)  # scalar  # type: ignore

        if self.p == 1:
            return {"wasserstein_1": cost}
        elif self.p == 2:
            return {"wasserstein_2": cost**0.5}
        else:
            return {f"wasserstein_{self.p}": cost}

    def baseline_metrics(self, Y, w_eval=None, **kwargs) -> dict[str, float]:
        """Return optional Wasserstein baselines."""
        return {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return f"wasserstein_{self.p}"


class Wasserstein1(Wasserstein):
    """First Wasserstein distance."""

    @property
    def p(self) -> int:
        """Return the Wasserstein order."""
        return 1

    def _compute_distance_matrix(self, X, Y) -> np.ndarray:
        """Compute Euclidean ground costs."""
        # Convert tensors to numpy
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        # Euclidean distances (not squared) -> cost equals W1
        return ot.dist(X, Y, metric="euclidean")


class Wasserstein2(Wasserstein):
    """Second Wasserstein distance."""

    @property
    def p(self) -> int:
        """Return the Wasserstein order."""
        return 2

    def _compute_distance_matrix(self, X, Y) -> np.ndarray:
        """Compute squared Euclidean ground costs."""
        # Convert tensors to numpy
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        # Squared Euclidean distances -> OT cost equals W2^2
        return ot.dist(X, Y, metric="sqeuclidean")

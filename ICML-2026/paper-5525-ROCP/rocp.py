"""
Risk-Optimal Conformal Prediction (ROCP) utilities.

Implements Algorithm 1 (Risk–optimal conformal prediction) using a global beta
calibrated on the calibration set (no test-point inclusion).
"""

from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple, NamedTuple

import numpy as np
from functools import lru_cache


LossFn = Callable[[object, int], float]


class ActionLossStats(NamedTuple):
    """Precomputed loss statistics for a single action at a fixed x."""

    sorted_losses: np.ndarray  # ascending loss values aligned with cum_probs
    cum_probs: np.ndarray  # cumulative probability in ascending loss order
    max_loss: float


class RiskOptimalConformalPredictor:
    """
    Risk-Optimal Conformal Prediction (ROCP) for discrete label spaces.

    This class assumes a probabilistic classifier that outputs p(y|x) for y in
    {0, ..., K-1}, and a loss function ell(a, y) defined for actions a and labels y.
    """

    def __init__(
        self,
        actions: Sequence[object],
        loss_matrix: Optional[np.ndarray] = None,
        loss_fn: Optional[LossFn] = None,
    ) -> None:
        """
        Args:
            actions: Iterable of possible actions.
            loss_matrix: Optional array of shape (num_labels, num_actions).
            loss_fn: Optional callable loss_fn(a, y) -> float.
        """
        if loss_matrix is None and loss_fn is None:
            raise ValueError("Provide either loss_matrix or loss_fn.")
        if loss_matrix is not None and loss_fn is not None:
            raise ValueError("Provide only one of loss_matrix or loss_fn.")

        self.actions: List[object] = list(actions)
        self.num_actions = len(self.actions)
        self.loss_matrix = None
        self.loss_fn = loss_fn

        if loss_matrix is not None:
            arr = np.asarray(loss_matrix, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.num_actions:
                raise ValueError(
                    "loss_matrix must have shape (num_labels, num_actions)."
                )
            self.loss_matrix = arr
        self._stats_cache_key = None
        self._stats_cache_value = None

    def _loss_vector(self, num_labels: int, action_index: int) -> np.ndarray:
        if self.loss_matrix is not None:
            if self.loss_matrix.shape[0] != num_labels:
                raise ValueError(
                    "Number of labels in loss_matrix does not match x_probs length."
                )
            return self.loss_matrix[:, action_index]
        action = self.actions[action_index]
        return np.array([self.loss_fn(action, y) for y in range(num_labels)], dtype=float)

    def _action_stats(self, x_probs: np.ndarray) -> List[ActionLossStats]:
        key = tuple(x_probs)
        if self._stats_cache_key == key:
            return self._stats_cache_value
        stats: List[ActionLossStats] = []
        num_labels = len(x_probs)
        for a_idx in range(self.num_actions):
            loss_vec = self._loss_vector(num_labels, a_idx)
            order = np.argsort(loss_vec, kind="mergesort")
            sorted_losses = loss_vec[order]
            cum_probs = np.cumsum(x_probs[order])
            max_loss = float(np.max(loss_vec))
            stats.append(ActionLossStats(sorted_losses, cum_probs, max_loss))
        self._stats_cache_key = key
        self._stats_cache_value = stats
        return stats

    @staticmethod
    def _quantile_loss_from_stats(stats: ActionLossStats, t: float) -> float:
        if t <= 0.0:
            return float(stats.sorted_losses[0])
        if t >= 1.0:
            return float(stats.max_loss)
        idx = int(np.searchsorted(stats.cum_probs, t, side="left"))
        idx = min(idx, len(stats.sorted_losses) - 1)
        return float(stats.sorted_losses[idx])

    @staticmethod
    def _candidate_t_values(stats_list: List[ActionLossStats]) -> List[float]:
        t_values: Set[float] = {0.0, 1.0}
        for stats in stats_list:
            losses = stats.sorted_losses
            cum = stats.cum_probs
            if len(losses) == 0:
                continue
            for i in range(len(losses)):
                if i == len(losses) - 1 or losses[i] != losses[i + 1]:
                    t_values.add(float(cum[i]))
        return sorted(t_values)

    def compute_theta_and_action(
        self, x_probs: Sequence[float], t: float
    ) -> Tuple[float, object, int]:
        """
        Compute theta(x,t) and action a(x,t) for a fixed x_probs and t.
        Returns (theta, action, action_index).
        """
        x_probs = np.asarray(x_probs, dtype=float)
        stats_list = self._action_stats(x_probs)
        best_val = np.inf
        best_theta = None
        best_action = None
        best_action_idx = None

        for a_idx, stats in enumerate(stats_list):
            q_t = self._quantile_loss_from_stats(stats, t)
            val = t * q_t + (1.0 - t) * stats.max_loss
            if val < best_val:
                best_val = val
                best_theta = q_t
                best_action = self.actions[a_idx]
                best_action_idx = a_idx

        if best_theta is None or best_action is None or best_action_idx is None:
            raise RuntimeError("Failed to compute theta/action.")
        return best_theta, best_action, best_action_idx

    def compute_set(self, x_probs: Sequence[float], t: float) -> Set[int]:
        """
        Compute C(x,t) = {y: ell(a(x,t), y) <= theta(x,t)}.
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)
        theta, action, a_idx = self.compute_theta_and_action(x_probs, t)
        if self.loss_matrix is not None:
            loss_vec = self._loss_vector(num_labels, a_idx)
        else:
            loss_vec = np.array(
                [self.loss_fn(action, y) for y in range(num_labels)], dtype=float
            )
        return {y for y in range(num_labels) if loss_vec[y] <= theta}

    def compute_g_hat(self, x_probs: Sequence[float], beta: float) -> float:
        """
        Compute g_hat(x, beta) = argmin_t [V_x(t) - beta * t].
        Ties are broken by choosing the largest t.
        """
        x_probs = np.asarray(x_probs, dtype=float)
        stats_list = self._action_stats(x_probs)
        candidates = self._candidate_t_values(stats_list)

        best_val = np.inf
        best_t = 0.0

        for t in candidates:
            v_t = np.inf
            for stats in stats_list:
                q_t = self._quantile_loss_from_stats(stats, t)
                val = t * q_t + (1.0 - t) * stats.max_loss
                if val < v_t:
                    v_t = val
            obj = v_t - beta * t
            if obj < best_val - 1e-12 or (abs(obj - best_val) <= 1e-12 and t > best_t):
                best_val = obj
                best_t = t

        return float(best_t)

    def predict_set(self, x_probs: Sequence[float], beta: float) -> Set[int]:
        """
        Compute C_ROCP(x) using the global beta (approximate ROCP).
        """
        t_star = self.compute_g_hat(x_probs, beta)
        return self.compute_set(x_probs, t_star)

    def is_covered(self, x_probs: Sequence[float], y_true: int, beta: float) -> bool:
        return y_true in self.predict_set(x_probs, beta)

    def coverage_for_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        beta: float,
    ) -> float:
        """
        Compute empirical coverage on calibration data for a given beta.
        """
        hits = 0
        n = len(calib_probs)
        for x_probs, y in zip(calib_probs, calib_labels):
            if self.is_covered(x_probs, int(y), beta):
                hits += 1
        return hits / max(n, 1)

    def calibrate_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        alpha: float,
        max_iter: int = 50,
        beta_init: float = 1.0,
        beta_max: float = 1e6,
    ) -> float:
        """
        Calibrate a global beta so that coverage >= 1 - alpha on calibration set.

        This uses a doubling + bisection strategy (as in RAC), and does NOT
        include the test point in the coverage constraint.
        """
        target = 1.0 - alpha
        cov0 = self.coverage_for_beta(calib_probs, calib_labels, 0.0)
        if cov0 >= target:
            return 0.0

        beta_low = 0.0
        beta_high = beta_init
        cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)
        while cov < target and beta_high < beta_max:
            beta_low = beta_high
            beta_high *= 2.0
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)

        if cov < target:
            return float(beta_high)

        for _ in range(max_iter):
            beta_mid = 0.5 * (beta_low + beta_high)
            if beta_high - beta_low < 1e-8 * max(abs(beta_mid), 1e-6):
                break
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_mid)
            if cov >= target:
                beta_high = beta_mid
            else:
                beta_low = beta_mid

        return float(0.5 * (beta_low + beta_high))

    def action_and_certificate(
        self, label_set: Iterable[int], alpha: float, num_labels: Optional[int] = None
    ) -> Tuple[object, float]:
        """
        Compute loss-optimal action and in-set loss certificate for a given set.
        """
        label_set = set(label_set)
        if not label_set:
            raise ValueError("label_set must be non-empty.")

        if num_labels is None:
            if self.loss_matrix is not None:
                num_labels = self.loss_matrix.shape[0]
            else:
                num_labels = max(label_set) + 1
        best_val = np.inf
        best_action = None
        best_in_loss = None

        for a_idx, action in enumerate(self.actions):
            loss_vec = self._loss_vector(num_labels, a_idx)
            in_loss = float(np.max(loss_vec[list(label_set)]))
            if len(label_set) == num_labels:
                out_loss = in_loss
            else:
                out_idx = [y for y in range(num_labels) if y not in label_set]
                out_loss = float(np.max(loss_vec[out_idx]))
            val = in_loss + alpha * max(out_loss - in_loss, 0.0)
            if val < best_val:
                best_val = val
                best_action = action
                best_in_loss = in_loss

        if best_action is None or best_in_loss is None:
            raise RuntimeError("Failed to compute action/certificate.")
        return best_action, best_in_loss

    def calibrate_beta_class_conditional(
        self,
        calib_probs,
        calib_labels,
        alpha: float,
        min_per_class: int = 50,
        max_iter: int = 50,
        beta_init: float = 1.0,
        beta_max: float = 1e6,
    ):
        """
        Calibrate per-class beta values for class-conditional coverage.
        
        For each class c, calibrate beta_c using only calibration points of class c.
        Fall back to global beta when per-class sample size < min_per_class.
        
        Returns:
            beta_dict: {class_idx: beta_value} mapping
            global_beta: fallback beta for classes with too few samples
        """
        target = 1.0 - alpha
        
        # First compute global beta as fallback
        global_beta = self.calibrate_beta(
            calib_probs, calib_labels, alpha, 
            max_iter=max_iter, beta_init=beta_init, beta_max=beta_max
        )
        
        num_classes = calib_probs.shape[1] if hasattr(calib_probs, 'shape') else len(calib_probs[0])
        calib_probs = np.asarray(calib_probs)
        calib_labels = np.asarray(calib_labels)
        
        beta_dict = {}
        for c in range(num_classes):
            mask = calib_labels == c
            n_c = int(np.sum(mask))
            if n_c < min_per_class:
                beta_dict[c] = global_beta
                continue
            
            c_probs = calib_probs[mask]
            c_labels = calib_labels[mask]
            beta_dict[c] = self.calibrate_beta(
                c_probs, c_labels, alpha,
                max_iter=max_iter, beta_init=beta_init, beta_max=beta_max
            )
        
        return beta_dict, global_beta
    
    def predict_set_cc(
        self, x_probs, beta_dict, global_beta
    ):
        """
        Compute prediction set using class-conditional beta.
        Uses beta corresponding to the predicted (argmax) class.
        """
        x_probs = np.asarray(x_probs, dtype=float)
        pred_class = int(np.argmax(x_probs))
        beta = beta_dict.get(pred_class, global_beta)
        return self.predict_set(x_probs, beta)


__all__ = ["RiskOptimalConformalPredictor"]

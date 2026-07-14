from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import MTL
import preprocessing
from ARMUL import ARMUL, Baselines


_ORIGINAL_PREPROCESSING = getattr(preprocessing, "MTL_preprocessing", None)


def safe_MTL_preprocessing(
    data,
    link: str = "linear",
    intercept: bool = True,
    n_class: int = 1,
    standardization: bool = True,
):
    """
    Drop-in replacement for preprocessing.MTL_preprocessing.

    The original code returns zero feature scales when standardization=False, which
    later causes divisions by zero during prediction. This variant keeps the no-op
    preprocessing behavior but returns X_stds = 1 instead.
    """
    if standardization:
        if _ORIGINAL_PREPROCESSING is None:
            raise RuntimeError("Original preprocessing.MTL_preprocessing not found.")
        return _ORIGINAL_PREPROCESSING(
            data=data,
            link=link,
            intercept=intercept,
            n_class=n_class,
            standardization=standardization,
        )

    m = len(data[0])
    d = data[0][0].shape[1]
    n_list = np.zeros(m, dtype=int)

    X_means = np.zeros((d, 1))
    X_stds = np.ones((d, 1))
    if intercept:
        X_means = np.vstack((np.zeros((1, 1)), X_means))
        X_stds = np.vstack((np.ones((1, 1)), X_stds))

    y_mean, y_std = 0, 1
    X, Y = [], []

    for j in range(m):
        X_j = data[0][j]
        n_list[j] = X_j.shape[0]
        if intercept:
            X_j = np.hstack((np.ones((n_list[j], 1)), X_j))
        X.append(X_j)

    if link == "linear" or n_class == 2:
        for y_j in data[1]:
            Y.append(y_j.reshape(-1, 1))
        d_out = 1
    else:
        d_out = n_class
        for y_j in data[1]:
            rows = np.arange(y_j.shape[0])
            encoded = np.zeros((y_j.shape[0], n_class))
            encoded[rows, y_j.reshape(-1)] = 1
            Y.append(encoded)

    return [X, Y, X_means, X_stds, y_mean, y_std, n_list, d_out]


preprocessing.MTL_preprocessing = safe_MTL_preprocessing
MTL.MTL_preprocessing = safe_MTL_preprocessing
MTL_preprocessing = safe_MTL_preprocessing


DEFAULT_METHODS = ("DP", "ITL", "ARMUL", "OURS")
DEFAULT_Q_GRID = (0.1, 0.4, 0.7, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
DEFAULT_SPLITS = ("all", "related", "outlier")


@dataclass(frozen=True)
class SyntheticSetting:
    n: int = 100
    m: int = 30
    d: int = 30
    signal_norm: float = 2.0
    sigma: float = 1.0
    delta: float = 0.3
    epsilon: float = 0.1
    outlier_radius: float = 10.0
    decay_alpha: float = 1.0
    effective_rank: Optional[int] = None
    covariance_mode: str = "shared"
    covariance_groups: int = 1
    shared_theta_mode: str = "axis"
    outlier_scale_mode: str = "l2"
    scaled_related_fraction: float = 0.0
    variance_multiplier: float = 1.0
    bar_b_target: Optional[float] = None
    inflated_related_count: int = 0
    deflated_related_count: Optional[int] = None
    deflate_all_remaining_related: bool = False
    shuffled_related_fraction: float = 0.0
    shuffle_offset: Optional[int] = None
    isotropic_related_count: int = 0
    isotropic_covariance_scale: float = 1.0
    spike_floor_ratio: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    q_grid: Tuple[float, ...] = DEFAULT_Q_GRID
    n_fold: int = 5
    eta: float = 0.05
    t_iter: int = 400
    n_simul: int = 30
    seed0: int = 10


class SyntheticSweepExperiment:
    def __init__(self, setting: SyntheticSetting):
        self.setting = setting
        self.data: List[List[np.ndarray]] = [[], []]
        self.Theta: Optional[np.ndarray] = None
        self.task_scales: Optional[np.ndarray] = None
        self.related_tasks: np.ndarray = np.array([], dtype=int)
        self.outlier_tasks: np.ndarray = np.array([], dtype=int)
        self.scaled_related_tasks: np.ndarray = np.array([], dtype=int)
        self.deflated_related_tasks: np.ndarray = np.array([], dtype=int)
        self.shuffled_related_tasks: np.ndarray = np.array([], dtype=int)
        self.isotropic_related_tasks: np.ndarray = np.array([], dtype=int)
        self.spike_minority_related_tasks: np.ndarray = np.array([], dtype=int)
        self.deflated_variance_factor: float = 1.0
        self.spike_floor_ratio_realized: float = 1.0
        self.err: Dict[str, np.ndarray] = {}
        self.balancedness_estimate: float = float("nan")

    @staticmethod
    def sample_unit_hemisphere(rng: np.random.Generator, d: int) -> np.ndarray:
        vec = rng.normal(size=d)
        if vec[0] < 0:
            vec = -vec
        return vec / np.linalg.norm(vec)

    @staticmethod
    def sample_unit_sphere(rng: np.random.Generator, d: int) -> np.ndarray:
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)

    def _base_signal(self, rng: np.random.Generator) -> np.ndarray:
        base = np.zeros(self.setting.d)
        if self.setting.shared_theta_mode == "axis":
            base[0] = self.setting.signal_norm
            return base
        if self.setting.shared_theta_mode == "dense":
            return self.setting.signal_norm * self.sample_unit_hemisphere(rng, self.setting.d)
        raise ValueError("shared_theta_mode must be one of {'axis', 'dense'}")

    def _base_scales(self) -> np.ndarray:
        coords = np.arange(1, self.setting.d + 1, dtype=float)
        scales = 1.0 / np.power(coords, self.setting.decay_alpha)
        rank = self.setting.d if self.setting.effective_rank is None else int(self.setting.effective_rank)
        rank = max(1, min(rank, self.setting.d))
        if rank < self.setting.d:
            scales[rank:] = 0.0
        return scales

    def _spike_covariance_scales(self, spike_coordinate: int, floor_ratio: float) -> np.ndarray:
        floor_ratio = float(np.clip(floor_ratio, 0.0, 1.0))
        scales = np.sqrt(floor_ratio) * np.ones(self.setting.d)
        scales[int(spike_coordinate) % self.setting.d] = 1.0
        return scales

    def _balanced_spike_parameters(self, n_related: int) -> Tuple[int, float]:
        if n_related <= 1:
            return 0, float(self.setting.spike_floor_ratio)

        target = 1.0 if self.setting.bar_b_target is None else float(self.setting.bar_b_target)
        if target <= 1.0 + 1e-12:
            return 0, float(self.setting.spike_floor_ratio)

        minority_count = max(1, int(np.floor(n_related / target)))
        minority_count = min(minority_count, n_related // 2)
        p_minority = minority_count / n_related
        floor_ratio = (1.0 / target - p_minority) / max(1.0 - p_minority, 1e-12)
        floor_ratio = float(np.clip(floor_ratio, 0.0, 1.0))
        return minority_count, floor_ratio

    def _task_scales(self) -> np.ndarray:
        base_scales = self._base_scales()
        task_scales = np.zeros((self.setting.m, self.setting.d))

        if self.setting.covariance_mode == "shared":
            task_scales[:] = base_scales
            return task_scales

        if self.setting.covariance_mode == "shifted":
            n_groups = max(1, min(int(self.setting.covariance_groups), self.setting.d))
            shifts = np.linspace(0, self.setting.d, num=n_groups, endpoint=False, dtype=int)
            for j in range(self.setting.m):
                task_scales[j] = np.roll(base_scales, int(shifts[j % n_groups]))
            return task_scales

        if self.setting.covariance_mode == "inflated_subset":
            task_scales[:] = base_scales
            return task_scales

        if self.setting.covariance_mode == "redistributed_subset":
            task_scales[:] = base_scales
            return task_scales

        if self.setting.covariance_mode == "shuffled_subset":
            task_scales[:] = base_scales
            return task_scales

        if self.setting.covariance_mode == "isotropic_subset":
            task_scales[:] = base_scales
            return task_scales

        if self.setting.covariance_mode == "balanced_spike_groups":
            task_scales[:] = self._spike_covariance_scales(
                spike_coordinate=0,
                floor_ratio=self.setting.spike_floor_ratio,
            )
            return task_scales

        raise ValueError(
            "covariance_mode must be one of "
            "{'shared', 'shifted', 'inflated_subset', 'redistributed_subset', "
            "'shuffled_subset', 'isotropic_subset', 'balanced_spike_groups'}"
        )

    def generate(self, seed: int) -> "SyntheticSweepExperiment":
        rng = np.random.default_rng(seed)
        self.task_scales = self._task_scales()
        self.Theta = np.zeros((self.setting.d, self.setting.m))

        n_outliers = int(self.setting.m * self.setting.epsilon)
        outlier_tasks = np.sort(rng.choice(self.setting.m, size=n_outliers, replace=False))
        outlier_set = set(outlier_tasks.tolist())
        related_tasks = np.array([j for j in range(self.setting.m) if j not in outlier_set], dtype=int)

        scaled_related_tasks = np.array([], dtype=int)
        deflated_related_tasks = np.array([], dtype=int)
        shuffled_related_tasks = np.array([], dtype=int)
        isotropic_related_tasks = np.array([], dtype=int)
        spike_minority_related_tasks = np.array([], dtype=int)
        self.deflated_variance_factor = 1.0
        self.spike_floor_ratio_realized = float(self.setting.spike_floor_ratio)
        if self.setting.covariance_mode == "inflated_subset" and len(related_tasks) > 0:
            frac = float(np.clip(self.setting.scaled_related_fraction, 0.0, 1.0))
            n_scaled = int(np.floor(len(related_tasks) * frac))
            if frac > 0 and n_scaled == 0:
                n_scaled = 1
            if n_scaled > 0 and self.setting.variance_multiplier > 0:
                scaled_related_tasks = np.sort(rng.choice(related_tasks, size=n_scaled, replace=False))
                self.task_scales[scaled_related_tasks] *= np.sqrt(self.setting.variance_multiplier)
        elif self.setting.covariance_mode == "redistributed_subset" and len(related_tasks) > 0:
            n_related = len(related_tasks)
            n_inflated = int(max(0, self.setting.inflated_related_count))
            n_inflated = min(n_inflated, n_related)
            if self.setting.bar_b_target is not None:
                L = float(self.setting.bar_b_target)
            else:
                L = float(self.setting.variance_multiplier)

            if n_inflated > 0 and L > 1.0:
                required_down = n_inflated * (L - 1.0)
                max_deflated = n_related - n_inflated
                if max_deflated <= 0:
                    raise ValueError(
                        "redistributed_subset requires at least one non-inflated related task "
                        "to preserve Sigma_S."
                    )

                if self.setting.deflate_all_remaining_related:
                    n_deflated = max_deflated
                elif self.setting.deflated_related_count is None:
                    n_deflated = int(np.floor(required_down)) + 1
                else:
                    n_deflated = int(self.setting.deflated_related_count)

                if n_deflated <= required_down or n_deflated > max_deflated:
                    raise ValueError(
                        "redistributed_subset needs more deflated related tasks. "
                        f"Got n_inflated={n_inflated}, variance_multiplier={L}, "
                        f"n_deflated={n_deflated}, but need an integer in "
                        f"({required_down:.3f}, {max_deflated}] to keep Sigma_S unchanged."
                    )

                deflated_factor = 1.0 - required_down / n_deflated
                if not (0.0 < deflated_factor < 1.0):
                    raise ValueError(
                        "Computed deflated covariance factor must lie in (0, 1). "
                        f"Got {deflated_factor}."
                    )

                scaled_related_tasks = np.sort(rng.choice(related_tasks, size=n_inflated, replace=False))
                remaining_related = np.array([j for j in related_tasks if j not in set(scaled_related_tasks.tolist())], dtype=int)
                if self.setting.deflate_all_remaining_related:
                    deflated_related_tasks = remaining_related.copy()
                else:
                    deflated_related_tasks = np.sort(rng.choice(remaining_related, size=n_deflated, replace=False))

                self.task_scales[scaled_related_tasks] *= np.sqrt(L)
                self.task_scales[deflated_related_tasks] *= np.sqrt(deflated_factor)
                self.deflated_variance_factor = deflated_factor
        elif self.setting.covariance_mode == "shuffled_subset" and len(related_tasks) > 0:
            frac = float(np.clip(self.setting.shuffled_related_fraction, 0.0, 1.0))
            n_shuffled = int(np.floor(len(related_tasks) * frac))
            if frac > 0 and n_shuffled == 0:
                n_shuffled = 1
            if n_shuffled > 0:
                shuffled_related_tasks = np.sort(rng.choice(related_tasks, size=n_shuffled, replace=False))
                offset = self.setting.d // 2 if self.setting.shuffle_offset is None else int(self.setting.shuffle_offset)
                base_scales = self._base_scales()
                shifted_scales = np.roll(base_scales, offset)
                self.task_scales[shuffled_related_tasks] = shifted_scales
        elif self.setting.covariance_mode == "isotropic_subset" and len(related_tasks) > 0:
            n_isotropic = int(max(0, self.setting.isotropic_related_count))
            n_isotropic = min(n_isotropic, len(related_tasks))
            isotropic_scale = float(self.setting.isotropic_covariance_scale)
            if n_isotropic > 0 and isotropic_scale > 0.0:
                isotropic_related_tasks = np.sort(rng.choice(related_tasks, size=n_isotropic, replace=False))
                self.task_scales[isotropic_related_tasks] = np.sqrt(isotropic_scale) * np.ones(self.setting.d)
        elif self.setting.covariance_mode == "balanced_spike_groups" and len(related_tasks) > 0:
            minority_count, floor_ratio = self._balanced_spike_parameters(len(related_tasks))
            majority_scales = self._spike_covariance_scales(spike_coordinate=0, floor_ratio=floor_ratio)
            alternate_scales = self._spike_covariance_scales(
                spike_coordinate=1 if self.setting.d > 1 else 0,
                floor_ratio=floor_ratio,
            )
            self.task_scales[:] = majority_scales
            self.spike_floor_ratio_realized = floor_ratio
            if minority_count > 0:
                spike_minority_related_tasks = np.sort(
                    rng.choice(related_tasks, size=minority_count, replace=False)
                )
                self.task_scales[spike_minority_related_tasks] = alternate_scales

        theta_shared = self._base_signal(rng)
        task_weights = (self.task_scales ** 2) / self.setting.d

        data_X, data_y = [], []
        need_target_signal_energy = self.setting.outlier_scale_mode == "match_population"
        related_signal_energy: List[float] = []
        for j in range(self.setting.m):
            if j not in outlier_set:
                direction = self.sample_unit_hemisphere(rng, self.setting.d)
                theta_j = theta_shared + self.setting.delta * direction
                if need_target_signal_energy:
                    related_signal_energy.append(float(np.sum(task_weights[j] * theta_j ** 2)))
            else:
                theta_j = np.zeros(self.setting.d)
            self.Theta[:, j] = theta_j

        if need_target_signal_energy and len(related_signal_energy) > 0:
            target_signal_energy = float(np.mean(related_signal_energy))
        elif need_target_signal_energy:
            target_signal_energy = float(np.sum(task_weights[0] * theta_shared ** 2))

        for j in outlier_tasks:
            if self.setting.outlier_scale_mode == "l2":
                direction = self.sample_unit_hemisphere(rng, self.setting.d)
                theta_j = self.setting.outlier_radius * direction
            elif self.setting.outlier_scale_mode == "match_population":
                direction = self.sample_unit_sphere(rng, self.setting.d)
                current_energy = float(np.sum(task_weights[j] * direction ** 2))
                scale = np.sqrt(target_signal_energy / max(current_energy, 1e-12))
                theta_j = scale * direction
            else:
                raise ValueError("outlier_scale_mode must be one of {'l2', 'match_population'}")

            self.Theta[:, j] = theta_j

        for j in range(self.setting.m):
            theta_j = self.Theta[:, j]
            raw_X = rng.normal(size=(self.setting.n, self.setting.d))
            X_sphere = raw_X / np.linalg.norm(raw_X, axis=1, keepdims=True)
            X_j = X_sphere * self.task_scales[j]
            y_j = X_j @ theta_j + self.setting.sigma * rng.normal(size=self.setting.n)
            data_X.append(X_j)
            data_y.append(y_j)

        self.data = [data_X, data_y]
        self.related_tasks = related_tasks
        self.outlier_tasks = outlier_tasks
        self.scaled_related_tasks = scaled_related_tasks
        self.deflated_related_tasks = deflated_related_tasks
        self.shuffled_related_tasks = shuffled_related_tasks
        self.isotropic_related_tasks = isotropic_related_tasks
        self.spike_minority_related_tasks = spike_minority_related_tasks
        self.balancedness_estimate = self.estimate_balancedness()
        return self

    def signal_energy(self) -> np.ndarray:
        if self.Theta is None or self.task_scales is None:
            raise RuntimeError("Call generate() before requesting signal energy.")
        weights = (self.task_scales ** 2) / self.setting.d
        return np.sum(weights * (self.Theta.T ** 2), axis=1)

    def estimate_balancedness(self) -> float:
        if self.task_scales is None or len(self.related_tasks) == 0:
            return float("nan")

        diag_terms = self.task_scales[self.related_tasks] ** 2
        sigma_related = np.mean(diag_terms, axis=0)
        tol = 1e-12
        best = 0.0

        for diag_j in diag_terms:
            if np.any((diag_j > tol) & (sigma_related <= tol)):
                return float("inf")
            valid = (diag_j > tol) & (sigma_related > tol)
            if np.any(valid):
                best = max(best, float(np.max(diag_j[valid] / sigma_related[valid])))

        return best

    def calc_population_mse(self, Theta_hat: np.ndarray) -> np.ndarray:
        if self.Theta is None or self.task_scales is None:
            raise RuntimeError("Call generate() before evaluating errors.")

        diff = Theta_hat - self.Theta
        mse = np.zeros(self.setting.m)
        for j in range(self.setting.m):
            weights = (self.task_scales[j] ** 2) / self.setting.d
            mse[j] = float(np.sum(weights * diff[:, j] ** 2))
        return mse

    def split_summary(self, errors: Sequence[float]) -> Dict[str, float]:
        errors = np.asarray(errors, dtype=float)
        summary = {"all": float(np.mean(errors))}

        if len(self.related_tasks) > 0:
            summary["related"] = float(np.mean(errors[self.related_tasks]))
        else:
            summary["related"] = float("nan")

        if len(self.outlier_tasks) > 0:
            summary["outlier"] = float(np.mean(errors[self.outlier_tasks]))
        else:
            summary["outlier"] = float("nan")

        return summary

    def run_baselines(self, eta: float, t_iter: int) -> Dict[str, np.ndarray]:
        base = Baselines(link="linear", n_class=1)

        base.DP_train(self.data, eta=eta, T=t_iter, standardization=False, intercept=False)
        theta_dp = base.models["DP"][:, :, 0].T

        base.STL_train(self.data, eta=eta, T=t_iter, standardization=False, intercept=False)
        theta_itl = base.models["STL"][:, :, 0].T

        baseline_errors = {
            "DP": self.calc_population_mse(theta_dp),
            "ITL": self.calc_population_mse(theta_itl),
        }
        self.err.update(baseline_errors)
        return baseline_errors


class OURS_Synthetic_Custom:
    def __init__(self):
        self.models: Dict[str, np.ndarray] = {}

    def train(self, data, lbd_list: np.ndarray, T_global: int = 400) -> "OURS_Synthetic_Custom":
        X, y, _, _, _, _, n_list, _ = MTL_preprocessing(
            data,
            link="linear",
            intercept=False,
            n_class=1,
            standardization=False,
        )

        m = len(X)
        d = X[0].shape[1]
        Y = [y_j if y_j.ndim == 2 else y_j.reshape(-1, 1) for y_j in y]
        Sigmas = [(X[j].T @ X[j]) / n_list[j] for j in range(m)]

        def objective_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            gamma = params[:d].reshape(d, 1)
            V = params[d:].reshape(m, d, 1)

            total_loss = 0.0
            grad_gamma = np.zeros_like(gamma)
            grad_V = np.zeros_like(V)
            eps_floor = 1e-12

            for j in range(m):
                v_j = V[j]
                w_j = gamma - v_j
                diff = X[j] @ w_j - Y[j]
                nj = n_list[j]

                total_loss += 0.5 * np.sum(diff ** 2) / nj
                grad_data = (X[j].T @ diff) / nj
                grad_gamma += grad_data
                grad_V[j] -= grad_data

                sigma_v = Sigmas[j] @ v_j
                norm_val = np.sqrt(np.sum(v_j * sigma_v) + eps_floor)
                total_loss += lbd_list[j] * norm_val

                if norm_val > 1e-10:
                    grad_penalty = lbd_list[j] * sigma_v / norm_val
                    grad_V[j] += grad_penalty

            gradient = np.concatenate([grad_gamma.ravel(), grad_V.ravel()])
            return total_loss, gradient

        res = minimize(
            objective_and_grad,
            np.zeros(d + m * d),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": T_global, "ftol": 1e-9},
        )

        gamma_hat = res.x[:d].reshape(d, 1)
        V_hat = res.x[d:].reshape(m, d, 1)

        theta_hat = np.zeros((d, m))
        for j in range(m):
            theta_hat[:, j] = (gamma_hat - V_hat[j]).ravel()

        self.models["ours"] = theta_hat
        return self

    def predict(self, X_test: Sequence[np.ndarray]) -> List[np.ndarray]:
        theta = self.models["ours"]
        return [X_j @ theta[:, j] for j, X_j in enumerate(X_test)]


def split_cv(n_list: Sequence[int], n_fold: int, seed: int) -> List[List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    splits = []
    for n in n_list:
        n = int(n)
        if n_fold > n:
            raise ValueError(f"n_fold={n_fold} cannot exceed n={n}.")
        perm = rng.permutation(n)
        folds = np.array_split(perm, n_fold)
        splits.append([fold.astype(int) for fold in folds])
    return splits


def mean_prediction_mse(y_true: Sequence[np.ndarray], y_pred: Sequence[np.ndarray]) -> float:
    per_task = []
    for y_t, y_p in zip(y_true, y_pred):
        y_t = np.asarray(y_t).ravel()
        y_p = np.asarray(y_p).ravel()
        per_task.append(np.mean((y_t - y_p) ** 2))
    return float(np.mean(per_task))


def run_cv_synthetic(
    data,
    model_type: str,
    q_grid: Sequence[float],
    d_dim: int,
    n_fold: int,
    eta: float,
    t_iter: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    n_list = np.array([len(y_j) for y_j in data[1]], dtype=int)
    splits = split_cv(n_list, n_fold=n_fold, seed=seed)

    best_q = float(q_grid[0])
    best_val_err = float("inf")

    for q in q_grid:
        fold_errors = []

        for k in range(n_fold):
            X_tr, y_tr, X_val, y_val = [], [], [], []
            for j in range(len(data[0])):
                idx_val = splits[j][k]
                idx_tr = np.delete(np.arange(n_list[j]), idx_val)
                X_tr.append(data[0][j][idx_tr])
                y_tr.append(data[1][j][idx_tr])
                X_val.append(data[0][j][idx_val])
                y_val.append(data[1][j][idx_val])

            n_train = np.array([len(y_j) for y_j in y_tr], dtype=float)
            lbd_vec = float(q) * np.sqrt(d_dim / n_train)

            if model_type == "ARMUL":
                model = ARMUL(link="linear", penalty="new")
                model.vanilla(
                    [X_tr, y_tr],
                    lbd=lbd_vec,
                    eta_global=eta,
                    eta_local=eta,
                    T_global=t_iter,
                    standardization=False,
                    intercept=False,
                )
                model.X_means = np.zeros((d_dim, 1))
                model.X_stds = np.ones((d_dim, 1))
                y_pred = model.predict(X_val, model="vanilla")
            elif model_type == "OURS":
                model = OURS_Synthetic_Custom()
                model.train([X_tr, y_tr], lbd_list=lbd_vec, T_global=t_iter)
                y_pred = model.predict(X_val)
            else:
                raise ValueError("model_type must be one of {'ARMUL', 'OURS'}")

            fold_errors.append(mean_prediction_mse(y_val, y_pred))

        avg_err = float(np.mean(fold_errors))
        if not np.isnan(avg_err) and avg_err < best_val_err:
            best_val_err = avg_err
            best_q = float(q)

    lbd_full = best_q * np.sqrt(d_dim / n_list)

    if model_type == "ARMUL":
        model = ARMUL(link="linear", penalty="new")
        model.vanilla(
            data,
            lbd=lbd_full,
            eta_global=eta,
            eta_local=eta,
            T_global=t_iter,
            standardization=False,
            intercept=False,
        )
        theta_hat = model.models["vanilla"][:, :, 0].T
    else:
        model = OURS_Synthetic_Custom()
        model.train(data, lbd_list=lbd_full, T_global=t_iter)
        theta_hat = model.models["ours"]

    return theta_hat, best_q


def fit_methods(
    experiment: SyntheticSweepExperiment,
    training: TrainingConfig,
    methods: Sequence[str] = DEFAULT_METHODS,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    method_set = set(methods)
    errors = experiment.run_baselines(eta=training.eta, t_iter=training.t_iter)
    selected_q: Dict[str, float] = {}

    if "ARMUL" in method_set:
        theta_armul, q_armul = run_cv_synthetic(
            experiment.data,
            model_type="ARMUL",
            q_grid=training.q_grid,
            d_dim=experiment.setting.d,
            n_fold=training.n_fold,
            eta=training.eta,
            t_iter=training.t_iter,
            seed=seed,
        )
        errors["ARMUL"] = experiment.calc_population_mse(theta_armul)
        selected_q["ARMUL"] = q_armul

    if "OURS" in method_set:
        theta_ours, q_ours = run_cv_synthetic(
            experiment.data,
            model_type="OURS",
            q_grid=training.q_grid,
            d_dim=experiment.setting.d,
            n_fold=training.n_fold,
            eta=training.eta,
            t_iter=training.t_iter,
            seed=seed + 1,
        )
        errors["OURS"] = experiment.calc_population_mse(theta_ours)
        selected_q["OURS"] = q_ours

    if "DP" not in method_set:
        errors.pop("DP", None)
    if "ITL" not in method_set:
        errors.pop("ITL", None)

    return errors, selected_q


def _series_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan")}
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return {
        "mean": float(np.mean(valid)),
        "std": std,
        "se": float(std / np.sqrt(len(valid))),
    }


def summarize_simulations(sim_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    rows = []
    for (value, method), group in sim_df.groupby([value_name, "method"], sort=True):
        row = {
            value_name: value,
            "method": method,
            "balancedness_est_mean": float(np.mean(group["balancedness_est"])),
        }
        for split in DEFAULT_SPLITS:
            stats = _series_summary(group[f"{split}_mse"].values)
            row[f"{split}_mean"] = stats["mean"]
            row[f"{split}_std"] = stats["std"]
            row[f"{split}_se"] = stats["se"]
        if "selected_q" in group:
            q_stats = _series_summary(group["selected_q"].values)
            row["selected_q_mean"] = q_stats["mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values([value_name, "method"]).reset_index(drop=True)


def run_sweep(
    value_name: str,
    values: Sequence[float],
    base_setting: SyntheticSetting,
    training: TrainingConfig,
    methods: Sequence[str] = DEFAULT_METHODS,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_rows = []

    for value_idx, value in enumerate(values):
        setting = replace(base_setting, **{value_name: value})
        if verbose:
            print(f"{value_name}={value} ({value_idx + 1}/{len(values)})")

        for sim_idx in range(training.n_simul):
            seed = training.seed0 + 10 * sim_idx
            experiment = SyntheticSweepExperiment(setting).generate(seed=seed)
            errors, selected_q = fit_methods(
                experiment,
                training=training,
                methods=methods,
                seed=seed,
            )

            for method, task_errors in errors.items():
                split_stats = experiment.split_summary(task_errors)
                row = {
                    value_name: value,
                    "seed": seed,
                    "method": method,
                    "balancedness_est": experiment.balancedness_estimate,
                }
                for split in DEFAULT_SPLITS:
                    row[f"{split}_mse"] = split_stats[split]
                if method in selected_q:
                    row["selected_q"] = selected_q[method]
                sim_rows.append(row)

    sim_df = pd.DataFrame(sim_rows)
    summary_df = summarize_simulations(sim_df, value_name=value_name)
    return sim_df, summary_df


def plot_sweep(
    summary_df: pd.DataFrame,
    value_name: str,
    split: str = "all",
    methods: Sequence[str] = DEFAULT_METHODS,
    band: str = "std",
    ax=None,
    title: Optional[str] = None,
):
    if split not in DEFAULT_SPLITS:
        raise ValueError(f"split must be one of {DEFAULT_SPLITS}")
    if band not in {"std", "se"}:
        raise ValueError("band must be either 'std' or 'se'")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    y_col = f"{split}_mean"
    band_col = f"{split}_{band}"

    for method in methods:
        method_df = summary_df[summary_df["method"] == method].sort_values(value_name)
        if method_df.empty:
            continue
        x = method_df[value_name].to_numpy(dtype=float)
        y = method_df[y_col].to_numpy(dtype=float)
        spread = method_df[band_col].to_numpy(dtype=float)

        line = ax.plot(x, y, marker="o", label=method)[0]
        ax.fill_between(x, y - spread, y + spread, alpha=0.15, color=line.get_color(), linewidth=0)

    label_map = {
        "epsilon": "Outlier fraction epsilon",
        "delta": "Task radius delta",
        "decay_alpha": "Eigendecay alpha",
        "effective_rank": "Effective rank",
        "variance_multiplier": "Variance multiplier",
        "bar_b_target": r"Population balancedness $\bar{B}$",
        "isotropic_covariance_scale": "Isotropic covariance scale",
    }
    ax.set_xlabel(label_map.get(value_name, value_name.replace("_", " ").title()))
    ax.set_ylabel(f"{split.title()} MSE")
    ax.set_title(title or f"{split.title()} MSE vs {value_name}")
    tick_values = np.sort(pd.unique(summary_df[value_name].dropna()))
    if 1 < len(tick_values) <= 8:
        ax.set_xticks(tick_values.astype(float))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    return ax


def plot_split_panels(
    summary_df: pd.DataFrame,
    value_name: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    band: str = "std",
    figsize: Tuple[float, float] = (15, 4),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
    for ax, split in zip(axes, DEFAULT_SPLITS):
        plot_sweep(
            summary_df,
            value_name=value_name,
            split=split,
            methods=methods,
            band=band,
            ax=ax,
            title=f"{split.title()} tasks",
        )
    fig.tight_layout()
    return fig, axes


def format_summary_table(summary_df: pd.DataFrame, value_name: str, split: str = "all") -> pd.DataFrame:
    value_col = value_name
    mean_col = f"{split}_mean"
    std_col = f"{split}_std"
    band_df = summary_df[[value_col, "method", mean_col, std_col, "balancedness_est_mean"]].copy()
    band_df["summary"] = band_df.apply(
        lambda row: f"{row[mean_col]:.4f} +/- {row[std_col]:.4f}",
        axis=1,
    )
    pivot = band_df.pivot(index=value_col, columns="method", values="summary")
    pivot["B_emp"] = (
        summary_df[[value_col, "balancedness_est_mean"]]
        .drop_duplicates()
        .set_index(value_col)["balancedness_est_mean"]
    )
    return pivot.reset_index()


def run_synthetic_suite(
    base_setting: Optional[SyntheticSetting] = None,
    training: Optional[TrainingConfig] = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    verbose: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    base_setting = base_setting or SyntheticSetting()
    training = training or TrainingConfig()

    results = {}
    results["delta"] = run_sweep(
        value_name="delta",
        values=(0.2, 0.4, 0.8, 1.6, 3.2),
        base_setting=replace(base_setting, covariance_mode="shared", covariance_groups=1),
        training=training,
        methods=methods,
        verbose=verbose,
    )
    results["epsilon"] = run_sweep(
        value_name="epsilon",
        values=(0.05, 0.1, 0.2, 0.3, 0.4),
        base_setting=replace(base_setting, covariance_mode="shared", covariance_groups=1),
        training=training,
        methods=methods,
        verbose=verbose,
    )
    results["decay_alpha"] = run_sweep(
        value_name="decay_alpha",
        values=(0.0, 0.5, 1.0, 1.5, 2.0),
        base_setting=replace(base_setting, epsilon=0.1, effective_rank=30, covariance_mode="shared"),
        training=training,
        methods=methods,
        verbose=verbose,
    )
    bar_b_setting = replace(
        base_setting,
        m=50,
        covariance_mode="balanced_spike_groups",
        bar_b_target=1.0,
        epsilon=0.5,
        delta=0.5,
        shared_theta_mode="dense",
        outlier_scale_mode="l2",
    )
    results["bar_b_target"] = run_sweep(
        value_name="bar_b_target",
        values=(5, 10, 15, 20),
        base_setting=bar_b_setting,
        training=training,
        methods=methods,
        verbose=verbose,
    )
    return results

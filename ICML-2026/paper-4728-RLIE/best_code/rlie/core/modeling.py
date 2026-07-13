from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rlie.core.metrics import compute_accuracy_and_macro_f1


@dataclass
class TrainingResult:
    model: Any
    probabilities: np.ndarray
    accuracy: float
    f1: float
    total_hypotheses: int
    retained_hypotheses: int
    best_params: Dict[str, Any]
    cv_best_score: float
    selection_metric: str


class SoftmaxTrainer:
    def __init__(self, config: Dict):
        self.config = config
        reg_cfg = config.get("elastic_net", {})
        self._max_iter = int(reg_cfg.get("max_iter", 5000))
        self._tol = float(reg_cfg.get("tol", 1e-4))
        self._fit_intercept = bool(reg_cfg.get("fit_intercept", True))
        class_weight = reg_cfg.get("class_weight")
        if isinstance(class_weight, str):
            lowered = class_weight.lower()
            self._class_weight = None if lowered == "none" else lowered
        else:
            self._class_weight = class_weight
        self._random_state = reg_cfg.get("random_state")
        self._warm_start = bool(reg_cfg.get("warm_start", False))
        self._n_jobs = int(reg_cfg.get("n_jobs", -1))

        self._selection_metric = config.get("model_selection_metric", "f1")
        self._c_grid: List[float] = reg_cfg.get("C_grid") or np.logspace(-3, 1, num=30).tolist()
        self._l1_grid: List[float] = reg_cfg.get("l1_ratio_grid") or np.linspace(0.01, 0.99, num=30).tolist()
        self._cv_folds = int(reg_cfg.get("cv_folds", 5))

    def train(self, features: np.ndarray, labels: np.ndarray) -> TrainingResult:
        if features.shape[1] == 0:
            raise ValueError("No features available for training")

        total_hypotheses = features.shape[1]
        base_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol,
            class_weight=self._class_weight,
            random_state=self._random_state,
            warm_start=self._warm_start,
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", base_model),
            ]
        )

        param_grid = {
            "clf__C": self._c_grid,
            "clf__l1_ratio": self._l1_grid,
        }

        scoring = self._resolve_scoring(self._selection_metric)
        cv = self._determine_cv_folds(labels)

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self._n_jobs,
            refit=True,
        )

        features_float = features.astype(float, copy=False)
        search.fit(features_float, labels)

        best_model = search.best_estimator_
        probs = best_model.predict_proba(features_float)[:, 1]
        preds = (probs >= 0.5).astype(int)
        label_space = sorted({int(value) for value in np.unique(labels)})
        acc, f1 = compute_accuracy_and_macro_f1(labels, preds, labels=label_space)
        coef = best_model.named_steps["clf"].coef_.ravel()
        retained = int(np.count_nonzero(np.abs(coef) > 1e-12))
        best_params = {}
        for key, value in search.best_params_.items():
            if isinstance(value, (np.floating, float)):
                best_params[key] = float(value)
            else:
                best_params[key] = value

        return TrainingResult(
            model=best_model,
            probabilities=probs,
            accuracy=acc,
            f1=f1,
            total_hypotheses=total_hypotheses,
            retained_hypotheses=retained,
            best_params=best_params,
            cv_best_score=float(search.best_score_),
            selection_metric=scoring,
        )

    def _determine_cv_folds(self, labels: np.ndarray) -> int:
        unique = np.unique(labels)
        if unique.size < 2:
            return 2
        return max(2, min(self._cv_folds, labels.shape[0]))

    @staticmethod
    def _resolve_scoring(metric_name: str) -> str:
        normalized = str(metric_name).strip().lower()
        if normalized in {"f1", "f1_score"}:
            return "f1_macro"
        if normalized in {"accuracy", "acc"}:
            return "accuracy"
        raise ValueError("model_selection_metric must be 'f1' or 'accuracy'")

# scripts/processors/streed.py
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pystreed import STreeDPiecewiseLinearRegressor, STreeDRegressor
from sklearn.exceptions import NotFittedError  
from .base import Processor, FitArtifacts

STREED_DEFAULTS = dict(
    verbose=False,
    cost_complexity=0.01,
    lasso_penalty=0.0,
    ridge_penalty=0.0,
    max_nodes=None,
)

def _get_depth(hp: Dict[str, Any], default: int = 4) -> int:
    return int(hp.get("depth", default))

def _get_max_nodes(hp: Dict[str, Any]) -> Optional[int]:
    v = hp.get("max_nodes", STREED_DEFAULTS["max_nodes"])
    return None if v is None else int(v)

class STreeDProcessor(Processor):
    """STreeD with piecewise linear leaves"""
    name = "streed"

    def build(self, **hparams):
        hp = {**STREED_DEFAULTS, **hparams}
        return STreeDPiecewiseLinearRegressor(
            simple=bool(hp.get("simple", False)),
            max_depth=_get_depth(hp),
            max_num_nodes=_get_max_nodes(hp),
            verbose=hp.get("verbose", STREED_DEFAULTS["verbose"]),
            cost_complexity=float(hp.get("cost_complexity", STREED_DEFAULTS["cost_complexity"])),
            lasso_penalty=float(hp.get("lasso_penalty", STREED_DEFAULTS["lasso_penalty"])),
            ridge_penalty=float(hp.get("ridge_penalty", STREED_DEFAULTS["ridge_penalty"])),
            n_thresholds=int(hp.get("n_thresholds", 5)),   
        )

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        y = np.asarray(y).ravel()
        model.fit(X, y)
        try:
            achieved_leaves = int(model.get_n_leaves())
            feasible = True
        except NotFittedError:
            achieved_leaves = None
            feasible = False

        try:
            fallback_mean = float(np.mean(y))
        except Exception:
            fallback_mean = 0.0
        setattr(model, "_fallback_mean_", fallback_mean)
        setattr(model, "_feasible_fit_", feasible)

        complexity = float(achieved_leaves) if achieved_leaves is not None else float("nan")

        return FitArtifacts(
            model=model,
            complexity=complexity,
            extras={
                "achieved_leaves": achieved_leaves,
                "feasible": feasible,
                "fallback_mean": fallback_mean,
            },
        )

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        try:
            return model.predict(X)
        except NotFittedError:
            if hasattr(model, "_fallback_mean_"):
                mu = float(getattr(model, "_fallback_mean_"))
                return np.full((X.shape[0],), mu, dtype=float)
            raise

class STreeDConstProcessor(STreeDProcessor):
    """STreeD with constant leaves"""
    name = "streed_const"

    def build(self, **hparams):
        hp = {**STREED_DEFAULTS, **hparams}
        return STreeDRegressor(
            max_depth=_get_depth(hp),
            max_num_nodes=_get_max_nodes(hp),
            verbose=hp.get("verbose", STREED_DEFAULTS["verbose"]),
            cost_complexity=float(hp.get("cost_complexity", STREED_DEFAULTS["cost_complexity"])),
            n_thresholds=int(hp.get("n_thresholds", 5)),  
        )

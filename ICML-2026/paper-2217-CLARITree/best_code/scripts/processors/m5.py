from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import FitArtifacts, Processor


M5_DEFAULTS: Dict[str, Any] = {
    "criterion": "squared_error",
    "splitter": "best",
    "depth": 4,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "use_pruning": False,
    "use_smoothing": False,
    "random_state": 42,
    "cost_complexity": 0.0,
    # Lambda scaling:
    # alpha_sse = TSS(train_y) * lambda
    # ccp_alpha = alpha_sse / n_train
    "lambda_scaled_by_tss": True,
}


def _maybe_int(v):
    return None if v is None else int(v)


class M5Processor(Processor):
    """m5py-based linear model tree processor."""

    name = "m5"

    def build(self, **hparams):
        # Keep a lightweight config dict; instantiate model in fit()
        # so lambda->alpha scaling can use fold-specific train y.
        return {**M5_DEFAULTS, **hparams}

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        try:
            from m5py import M5Prime
        except Exception as exc:
            raise ImportError(
                "m5py is required for method='m5'. Install with: pip install m5py"
            ) from exc

        y_arr = np.asarray(y, dtype=float).ravel()
        n_train = max(1, y_arr.shape[0])
        lam = float(model.get("cost_complexity", 0.0))

        if bool(model.get("lambda_scaled_by_tss", True)):
            tss = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
            alpha_sse = tss * lam
            ccp_alpha = float(alpha_sse / n_train)
        else:
            tss = float("nan")
            alpha_sse = float("nan")
            ccp_alpha = lam

        kwargs: Dict[str, Any] = {
            "criterion": str(model.get("criterion", M5_DEFAULTS["criterion"])),
            "splitter": str(model.get("splitter", M5_DEFAULTS["splitter"])),
            "max_depth": _maybe_int(model.get("depth", M5_DEFAULTS["depth"])),
            "min_samples_split": int(
                model.get("min_samples_split", M5_DEFAULTS["min_samples_split"])
            ),
            "min_samples_leaf": int(
                model.get("min_samples_leaf", M5_DEFAULTS["min_samples_leaf"])
            ),
            "max_leaf_nodes": _maybe_int(
                model.get("max_leaf_nodes", M5_DEFAULTS["max_leaf_nodes"])
            ),
            "min_impurity_decrease": float(
                model.get(
                    "min_impurity_decrease", M5_DEFAULTS["min_impurity_decrease"]
                )
            ),
            "use_pruning": bool(model.get("use_pruning", M5_DEFAULTS["use_pruning"])),
            "use_smoothing": model.get("use_smoothing", M5_DEFAULTS["use_smoothing"]),
            "ccp_alpha": float(ccp_alpha),
            "random_state": model.get("random_state", M5_DEFAULTS["random_state"]),
        }
        if model.get("leaf_model", None) is not None:
            kwargs["leaf_model"] = model["leaf_model"]

        m5 = M5Prime(**kwargs)
        m5.fit(X, y_arr)

        leaves = None
        try:
            leaves = int(m5.tree_.n_leaves)
        except Exception:
            leaves = None

        return FitArtifacts(
            model=m5,
            complexity=leaves,
            extras={
                "lambda_input": lam,
                "tss": tss,
                "alpha_sse": alpha_sse,
                "ccp_alpha": ccp_alpha,
            },
        )

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(X), dtype=float)


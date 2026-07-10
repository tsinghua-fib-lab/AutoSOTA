from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

from .base import FitArtifacts, Processor


PILOT_DEFAULTS: Dict[str, Any] = {
    "depth": 4,
    "max_model_depth": 200,
    "split_criterion": "BIC",
    "min_sample_split": 2,
    "min_sample_leaf": 1,
    "random_state": 42,
    "stride": 1,
    "categorical": None,
}


def _count_leaves(node) -> int:
    if node is None:
        return 0
    if getattr(node, "node", None) == "END":
        return 1
    return _count_leaves(getattr(node, "left", None)) + _count_leaves(
        getattr(node, "right", None)
    )


class PilotProcessor(Processor):
    """PILOT linear model tree processor."""

    name = "pilot"

    def build(self, **hparams):
        return {**PILOT_DEFAULTS, **hparams}

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        project_root = Path(__file__).resolve().parents[2]
        pilot_path = str(project_root / "PILOT")
        if pilot_path not in sys.path:
            sys.path.insert(0, pilot_path)

        try:
            from pilot import PILOT
        except Exception as exc:
            raise ImportError(
                "PILOT package is required for method='pilot'. "
                "Expected local package under ./PILOT."
            ) from exc

        hp = {**PILOT_DEFAULTS, **model}
        y_arr = np.asarray(y, dtype=float).ravel()

        categorical = hp.get("categorical")
        if categorical is None:
            categorical = np.array([], dtype=int)
        else:
            categorical = np.asarray(categorical, dtype=int).ravel()

        est = PILOT(
            max_depth=int(hp.get("depth", PILOT_DEFAULTS["depth"])),
            max_model_depth=int(
                hp.get("max_model_depth", PILOT_DEFAULTS["max_model_depth"])
            ),
            split_criterion=str(
                hp.get("split_criterion", PILOT_DEFAULTS["split_criterion"])
            ),
            min_sample_split=int(
                hp.get("min_sample_split", PILOT_DEFAULTS["min_sample_split"])
            ),
            min_sample_leaf=int(
                hp.get("min_sample_leaf", PILOT_DEFAULTS["min_sample_leaf"])
            ),
            random_state=int(hp.get("random_state", PILOT_DEFAULTS["random_state"])),
            stride=int(hp.get("stride", PILOT_DEFAULTS["stride"])),
        )
        est.fit(X, y_arr, categorical=categorical)

        leaves = _count_leaves(getattr(est, "model_tree", None))
        return FitArtifacts(
            model=est,
            complexity=leaves,
            extras={
                "tree_depth": int(getattr(est, "tree_depth", -1)),
                "model_depth": int(getattr(est, "model_depth", -1)),
                "min_sample_leaf": int(hp.get("min_sample_leaf")),
                "min_sample_split": int(hp.get("min_sample_split")),
            },
        )

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(X), dtype=float).reshape(-1)


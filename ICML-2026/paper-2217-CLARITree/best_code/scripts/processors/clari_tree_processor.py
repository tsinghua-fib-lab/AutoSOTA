# scripts/processors/clari_tree_processor.py
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from clari_tree import (
    Greedy,
    CLARITree,
    CLARITreeTopK,
    GreedyConst,
    CLARITreeConst,
)

from .base import Processor, FitArtifacts

# ===== Defaults & helpers =====
DEFAULTS: Dict[str, Any] = dict(
    verbose=False,
    cost_complexity=0.0,
    n_thresholds=1,
    thresholds_strategy="quantile",
    min_leaf_node_size=0,
    ridge_penalty=0.0,
)

def _get_depth(hp: Dict[str, Any], default: int = 4) -> int:
    return int(hp.get("depth", default))

def _get_n_thresholds(hp: Dict[str, Any]) -> int:
    return int(hp.get("n_thresholds", hp.get("stride", DEFAULTS["n_thresholds"])))

def _get_verbose(hp: Dict[str, Any]) -> bool:
    return bool(hp.get("verbose", DEFAULTS["verbose"]))

def _get_thresholds_strategy(hp: Dict[str, Any]) -> str:
    return str(hp.get("thresholds_strategy", DEFAULTS["thresholds_strategy"]))

def _get_min_leaf_node_size(hp: Dict[str, Any]) -> int:
    return int(hp.get("min_leaf_node_size", DEFAULTS["min_leaf_node_size"]))

def _get_const_min_leaf_node_size(hp: Dict[str, Any]) -> int:
    value = int(hp.get("min_leaf_node_size", 1))
    return value if value > 0 else 1

def _map_hparams_linear(hp: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        depth=_get_depth(hp),
        lambda_=float(hp.get("cost_complexity", DEFAULTS["cost_complexity"])),
        kappa=float(hp.get("ridge_penalty", DEFAULTS["ridge_penalty"])),
        n_thresholds=_get_n_thresholds(hp),
        thresholds_strategy=_get_thresholds_strategy(hp),
        verbose=_get_verbose(hp),
        min_leaf_node_size=_get_const_min_leaf_node_size(hp),
    )

def _map_hparams_const(hp: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        depth=_get_depth(hp),
        lambda_=float(hp.get("cost_complexity", DEFAULTS["cost_complexity"])),
        n_thresholds=_get_n_thresholds(hp),
        thresholds_strategy=_get_thresholds_strategy(hp),
        verbose=_get_verbose(hp),
        min_leaf_node_size=_get_min_leaf_node_size(hp),
    )

# ===== Base processor =====
class _BaseProcessor(Processor):
    """Shared methods for all tree processors."""

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> FitArtifacts:
        model.fit(X, y)
        comp: Optional[int] = None
        if hasattr(model, "n_leaves"):
            try:
                comp = model.n_leaves()
            except Exception:
                comp = None
        return FitArtifacts(model=model, complexity=comp, extras={})

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

# ===== Linear-leaf trees =====
class CholeskyTreeProcessor(_BaseProcessor):
    name = "cholesky_tree"
    def build(self, **hparams):
        hp = {**DEFAULTS, **hparams}
        kw = _map_hparams_linear(hp)
        return Greedy(**kw)

class CholicketyProcessor(_BaseProcessor):
    name = "cholickety"
    def build(self, **hparams):
        hp = {**DEFAULTS, **hparams}
        kw = _map_hparams_linear(hp)
        return CLARITree(**kw)

class TopKProcessor(_BaseProcessor):
    name = "topk"
    def build(self, **hparams):
        hp = {**DEFAULTS, **hparams}
        kw = _map_hparams_linear(hp)
        kw.update(
            k=0.4,
            verbose=False,
            include_greedy_candidate=True,
            greedy_top_ratio=0.05,
        )
        return CLARITreeTopK(**kw)

# ===== Constant-leaf trees =====
class ConstTreeProcessor(_BaseProcessor):
    name = "const_tree"
    def build(self, **hparams):
        hp = {**DEFAULTS, **hparams}
        kw = _map_hparams_const(hp)
        return GreedyConst(**kw)

class ConstlicketyProcessor(_BaseProcessor):
    name = "constlickety"
    def build(self, **hparams):
        hp = {**DEFAULTS, **hparams}
        kw = _map_hparams_const(hp)
        return CLARITreeConst(**kw)

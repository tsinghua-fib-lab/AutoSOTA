# scripts/processors/osrt.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .base import Processor, FitArtifacts

try:
    from osrt.model.osrt import splitOSRT as _splitOSRT
except Exception as e:
    raise ImportError(
        "Failed to import split OSRT. Make sure `master` of this repository is installed in this env.\n"
        f"Original error: {e}"
    )

def _to_df_ser(X, y):
    # X: np.ndarray or pd.DataFrame
    # y: np.ndarray or pd.Series
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X = np.asarray(X)
        cols = [f"x{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=cols)

    if isinstance(y, pd.Series):
        y_ser = y.copy()
        if y_ser.name is None:
            y_ser.name = "class"
    else:
        y = np.asarray(y).ravel()
        y_ser = pd.Series(y, name="class")
    return X_df, y_ser

class splitOSRTProcessor(Processor):
    """
    Wrap splitOSRT to fit the common Processor interface.
    hparams -> splitOSRT(config) one-to-one (with a couple defaults).
    """
    name = "split_osrt"

    DEFAULTS = dict(
        regularization=0.01,   # lambda
        depth_budget=3,        # depth
        model_limit=100,
        metric="L2",           # OSRT expects "L1"/"L2"
        weights=[],
        verbose=False,
        diagnostics=False,
        cart_lookahead_depth=2, # specific to splitOSRT
    )

    def build(self, **hparams) -> _splitOSRT:
        cfg = {**self.DEFAULTS, **hparams}
        # guard/be-lenient for weights: [] -> None
        if cfg.get("weights", None) is None:
            cfg["weights"] = []
        return _splitOSRT(cfg)

    def fit(self, model: _splitOSRT, X, y) -> FitArtifacts:
        X_df, y_ser = _to_df_ser(X, y)
        model.fit(X_df, y_ser)
        # splitOSRT exposes leaves()/nodes(); use leaves as our "complexity".
        leaves = None
        try:
            leaves = float(model.leaves())
        except Exception:
            pass
        extras = {}
        # if splitOSRT has timing inside model.time
        # TODO: verify this time; it may not be counting leaf fit times in optimal postprocessing
        try:
            extras["fit_time"] = getattr(model, "time", None)
        except Exception:
            pass
        return FitArtifacts(model=model, complexity=leaves, extras=extras)

    def predict(self, model: _splitOSRT, X) -> np.ndarray:
        X_df, _ = _to_df_ser(X, np.zeros(len(X)))  # dummy y to get schema if needed
        y_hat = model.predict(X_df)
        return np.asarray(y_hat).ravel()

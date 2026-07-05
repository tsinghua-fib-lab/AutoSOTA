"""Tabular encoder: DataFrame to binary LTT inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class TabularEncoder:
    """Encode tabular data into binary features for TT-Sparse.

    Categorical columns are one-hot encoded (drop first).
    Continuous columns are standardized then binarized via quantile-based
    thermometer encoding.
    """

    def __init__(
        self,
        target: str = "target",
        categorical: list[str] | None = None,
        continuous: list[str] | None = None,
        num_bits: int = 9,
        task_type: str = "binary",
    ) -> None:
        self.target_column = target
        self._user_cat = categorical
        self._user_cont = continuous
        self.num_bits = num_bits
        self.task_type = task_type
        self.categorical_features: list[str] = []
        self.continuous_features: list[str] = []
        self._ohe: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._label_enc: LabelEncoder | None = None
        self._thresholds: list[np.ndarray] = []
        self._cat_size = 0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> TabularEncoder:
        """Fit encoder on a DataFrame containing the target column."""
        X = df.drop(columns=[self.target_column])
        self.categorical_features, self.continuous_features = self._resolve_types(X)

        self._cat_size = 0
        if self.categorical_features:
            self._ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
            self._ohe.fit(X[self.categorical_features])
            self._cat_size = sum(len(c) - 1 for c in self._ohe.categories_)

        if self.continuous_features:
            self._scaler = StandardScaler()
            self._scaler.fit(X[self.continuous_features].astype(float))
            X_s = self._scaler.transform(X[self.continuous_features].astype(float))
            self._thresholds = [
                np.quantile(
                    np.unique(X_s[:, i]),
                    np.linspace(1 / (self.num_bits + 1), self.num_bits / (self.num_bits + 1), self.num_bits),
                )
                for i in range(X_s.shape[1])
            ]

        if self.task_type != "regression":
            self._label_enc = LabelEncoder()
            self._label_enc.fit(df[self.target_column])

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray | None]:
        """Transform a DataFrame into encoded arrays.

        Returns dict with keys: X_ltt, X_skip, y (y is None if target column absent).
        """
        has_y = self.target_column in df.columns
        X = df.drop(columns=[self.target_column]) if has_y else df
        n = len(X)

        X_cat = self._ohe.transform(X[self.categorical_features]) if self._ohe else np.zeros((n, 0))

        if self._scaler and self.continuous_features:
            X_s = self._scaler.transform(X[self.continuous_features].astype(float))
            X_bin = np.hstack([
                (X_s[:, i:i + 1] >= self._thresholds[i].reshape(1, -1)).astype(np.float32)
                for i in range(X_s.shape[1])
            ])
        else:
            X_s = np.zeros((n, 0))
            X_bin = np.zeros((n, 0), dtype=np.float32)

        X_ltt = np.concatenate([X_cat, X_bin], axis=1).astype(np.float32)
        X_skip = np.concatenate([X_cat, X_s], axis=1).astype(np.float32)

        y: np.ndarray | None = None
        if has_y:
            if self.task_type == "regression":
                y = df[self.target_column].values.astype(np.float64)
            else:
                y = self._label_enc.transform(df[self.target_column]).astype(np.int64)

        return {"X_ltt": X_ltt, "X_skip": X_skip, "y": y}

    def fit_transform(self, df: pd.DataFrame) -> dict[str, np.ndarray | None]:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    @property
    def n_ltt_features(self) -> int:
        """Number of binary features in the LTT input."""
        return self._cat_size + len(self.continuous_features) * self.num_bits

    @property
    def n_skip_features(self) -> int:
        """Number of features in the skip (continuous passthrough) input."""
        return self._cat_size + len(self.continuous_features)

    @property
    def class_names(self) -> list[str]:
        """Target class names (or ['output'] for regression)."""
        if self.task_type == "regression":
            return ["output"]
        return [str(c) for c in self._label_enc.classes_]

    def get_feature_manifest(self) -> dict[int, str]:
        """Map each binary LTT feature index to a human-readable description."""
        m: dict[int, str] = {}
        idx = 0
        if self._ohe:
            for fi, fn in enumerate(self.categorical_features):
                for cat in self._ohe.categories_[fi][1:]:
                    m[idx] = f"{fn} == '{cat}'"
                    idx += 1
        if self._scaler and self._thresholds:
            for fi, fn in enumerate(self.continuous_features):
                orig = self._thresholds[fi] * self._scaler.scale_[fi] + self._scaler.mean_[fi]
                for bi in range(self.num_bits):
                    m[idx] = f"{fn} >= {orig[bi]:.4f}"
                    idx += 1
        return m

    def get_skip_feature_names(self) -> list[str]:
        """Human-readable names for each skip feature."""
        names: list[str] = []
        if self._ohe:
            for fi, fn in enumerate(self.categorical_features):
                for cat in self._ohe.categories_[fi][1:]:
                    names.append(f"{fn} == '{cat}'")
        names.extend(self.continuous_features)
        return names

    def _resolve_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self._user_cat is not None and self._user_cont is not None:
            return list(self._user_cat), list(self._user_cont)
        cat = list(self._user_cat or [])
        cont = list(self._user_cont or [])
        specified = set(cat + cont)
        for col in X.columns:
            if col in specified:
                continue
            if X[col].dtype.kind in ("O", "S", "U") or X[col].dtype.name in ("category", "bool"):
                cat.append(col)
            elif X[col].dtype.kind in ("f", "i", "u"):
                cont.append(col)
        return cat, cont

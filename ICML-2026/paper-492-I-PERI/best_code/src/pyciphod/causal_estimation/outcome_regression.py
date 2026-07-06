from typing import Optional, Sequence, Dict, Any, Union, List
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import clone


class OutcomeRegressionEstimator:
    """Base class for outcome regression estimators (G-computation family).

    Subclasses should implement a fit/predict interface; this base class provides
    common data handling (validation of x,y,z,w), marginal-effect computation
    and convenience methods for G-computation ATE estimation.

    This implementation supports one or multiple exposures. If a single
    exposure (string) is provided the behaviour is backward-compatible and the
    returned 'cate' is a float. If multiple exposures are provided (sequence of
    column names) the returned 'cate' will be a dict mapping exposure->ATE.
    """

    def __init__(self, x: Union[str, Sequence[str]], y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, seed: Optional[int] = None):
        """

        :param x:  exposure column name (string) or sequence of exposure column names (for multiple exposures)
        :param y:  outcome column name (string)
        :param z:  adjustment set column names (sequence of strings), optional
        :param model:
        :param seed:
        """
        if isinstance(x, str):
            exposures: List[str] = [x]
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("x (exposure) must contain at least one column name")
            exposures = list(x)
            if not all(isinstance(xx, str) for xx in exposures):
                raise ValueError("All exposure names must be strings")
        else:
            raise ValueError("x (exposure) must be a column name string or a sequence of column name strings")

        if not isinstance(y, str):
            raise ValueError("y (outcome) must be provided as a column name string")

        self.exposures = exposures
        self.outcome = y

        if z is None:
            self.adjustment_set = []
        else:
            self.adjustment_set = list(z)

        if w is None:
            self.conditioning_set = []
        else:
            self.conditioning_set = list(w)

        # validate that exposures are not also listed in adjustment or conditioning
        overlap = set(self.exposures) & set(self.adjustment_set + self.conditioning_set)
        if len(overlap) > 0:
            raise ValueError(f"Exposure column(s) cannot appear in adjustment or conditioning sets: {sorted(list(overlap))}")

        # model: if None use LinearRegression, else clone
        if model is None:
            self.base_model = LinearRegression()
        else:
            # duck-typing: must implement fit and predict
            if not (hasattr(model, 'fit') and callable(getattr(model, 'fit'))):
                raise TypeError("Provided model must implement fit(X, y)")
            if not (hasattr(model, 'predict') and callable(getattr(model, 'predict'))):
                raise TypeError("Provided model must implement predict(X)")
            try:
                self.base_model = clone(model)
            except Exception:
                self.base_model = model

        self.model = None

    def _make_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns for adjustment + conditioning and the exposure column(s) present (copied from data)."""
        features = list(self.adjustment_set) + list(self.conditioning_set)
        if len(features) > 0:
            X = data[features].copy()
        else:
            X = pd.DataFrame(index=data.index)
        # ensure exposures present (not in features)
        for exp in self.exposures:
            if exp not in data.columns:
                raise ValueError(f"Exposure column '{exp}' not present in data")
            X[exp] = data[exp]
        return X

    def _compute_marginal_effect(self, model, data_base: pd.DataFrame, exposure_col: str, eps: Optional[float] = None) -> pd.Series:
        """Finite-difference marginal effect (per-unit) for a single continuous exposure.
        Returns a pandas Series aligned with data_base.index.
        """
        if eps is None:
            if exposure_col in data_base.columns:
                exp_vals = data_base[exposure_col].astype(float)
                std = float(exp_vals.std()) if exp_vals.std() > 0 else 1.0
                eps = max(1e-6, 1e-3 * std)
            else:
                eps = 1e-3

        Xp = data_base.copy()
        Xm = data_base.copy()
        Xp[exposure_col] = Xp[exposure_col] + eps
        Xm[exposure_col] = Xm[exposure_col] - eps

        pred_p = pd.Series(model.predict(Xp), index=data_base.index)
        pred_m = pd.Series(model.predict(Xm), index=data_base.index)
        return (pred_p - pred_m) / (2.0 * eps)

    def fit(self, data: pd.DataFrame):
        """Fit the outcome model on data. Subclasses may override; default clones base_model and fits."""
        X = self._make_feature_frame(data)
        y = data[self.outcome]
        try:
            self.model = clone(self.base_model)
        except Exception:
            self.model = self.base_model
        self.model.fit(X, y)
        return self

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform g-computation to estimate average causal effect (ATE) via outcome regression.

        This is the primary entrypoint (name aligned with `MetaLearners.run`). Behavior is unchanged from
        previous `g_computation` implementation.
        """
        if self.model is None:
            # lazy-fit if not fitted yet
            self.fit(data)

        # prepare base features
        X_base = self._make_feature_frame(data)

        # detect binary exposure per exposure
        is_binary_map = {}
        for exp in self.exposures:
            vals = set(data[exp].dropna().unique())
            is_binary_map[exp] = vals.issubset({0, 1, True, False})

        # compute per-exposure effect series
        effects: Dict[str, pd.Series] = {}
        for exp in self.exposures:
            if is_binary_map[exp]:
                X1 = X_base.copy()
                X0 = X_base.copy()
                X1[exp] = 1
                X0[exp] = 0
                pred1 = pd.Series(self.model.predict(X1), index=data.index)
                pred0 = pd.Series(self.model.predict(X0), index=data.index)
                effects[exp] = pred1 - pred0
            else:
                # continuous exposure -> marginal effect for this exposure
                effects[exp] = self._compute_marginal_effect(self.model, X_base, exposure_col=exp)

        # aggregate results
        if len(self.exposures) == 1:
            single_exp = self.exposures[0]
            cate_val = float(effects[single_exp].mean())
        else:
            cate_val = {exp: float(series.mean()) for exp, series in effects.items()}

        result: Dict[str, Any] = {'cate': cate_val, 'model': self.model}

        # conditional effects per stratum in W if provided
        if isinstance(self.conditioning_set, (list, tuple)) and len(self.conditioning_set) > 0:
            grouped = data.groupby(self.conditioning_set, dropna=False)
            ate_by_w = {}
            for w_val, idx in grouped.groups.items():
                if len(idx) == 0:
                    continue
                # normalize the group key to a tuple for consistency
                key = w_val if isinstance(w_val, tuple) else (w_val,)
                if len(self.exposures) == 1:
                    exp = self.exposures[0]
                    ate_by_w[key] = float(effects[exp].loc[idx].mean())
                else:
                    ate_by_w[key] = {exp: float(effects[exp].loc[idx].mean()) for exp in self.exposures}
            result['cate_per_w_stratum'] = ate_by_w

        return result


class LinearOutcomeRegression(OutcomeRegressionEstimator):
    """Concrete linear outcome regression using sklearn.linear_model.LinearRegression.

    This class reuses the base implementation; provided for convenience and naming symmetry.
    """

    def __init__(self, x: Union[str, Sequence[str]], y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        super().__init__(x, y, z, w, model=LinearRegression(), seed=seed)

    # fit and g_computation inherited from base class (they are appropriate for LinearRegression)


class GComputation(OutcomeRegressionEstimator):
    """Public-facing G-computation estimator. Thin subclass of OutcomeRegressionEstimator.

    Use `run(data)` to fit and obtain the results (consistent with `MetaLearners` API).
    """
    def __init__(self, x: Union[str, Sequence[str]], y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, seed: Optional[int] = None):
        super().__init__(x, y, z, w, model=model, seed=seed)

    # inherit run/fit/_make_feature_frame behaviour from base class


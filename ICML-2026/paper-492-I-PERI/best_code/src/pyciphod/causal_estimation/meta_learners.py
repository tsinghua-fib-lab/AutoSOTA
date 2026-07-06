from abc import ABC, abstractmethod
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from typing import Optional, Sequence


class MetaLearners(ABC):
    """
    Meta-learners are a class of methods for estimating treatment effects that use machine learning models to estimate the causal effects. They typically involve training separate models for the treated and control groups, and then using these models to estimate the treatment effect for each individual.

    Common meta-learners include:
    - T-learner: Trains separate models for the treated and control groups and estimates the treatment effect as the difference in predictions.
    - S-learner: Trains a single model that includes treatment as a feature and estimates the treatment effect by comparing predictions with and without the treatment feature.
    - X-learner: Combines the T-learner and S-learner approaches by first estimating the potential outcomes using a T-learner and then refining these estimates using an S-learner.

    Meta-learners are flexible and can be used with various machine learning algorithms, making them powerful tools for causal inference in complex settings.
    """
    def __init__(self, x: str, y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, model_tau=None, seed: Optional[int] = None):
        """
        Init the meta-learner.
        :param x: str  - exposure variable name
        :param y: str  - outcome variable name
        :param z: list of str  - adjustment set names
        :param w: list of str  - conditioning set names
        :param model: sklearn-like estimator to use as base learner (regressor). If None, RandomForestRegressor is used.
        :param model_tau: sklearn-like estimator to use as base learner for tau (regressor). If None, RandomForestRegressor is used.
        :param seed: optional random seed to control randomness where applicable
        """
        # store seed for propagation
        self.seed = seed

        if isinstance(x, str):
            # okay
            self.exposure = x
        else:
            raise ValueError("MetaLearner expects exposure variable name as string for DataFrame mode")

        if isinstance(y, str):
            # okay
            self.outcome = y
        else:
            raise ValueError("MetaLearner expects outcome variable name as string for DataFrame mode")

        # --- Validate adjustment set z and conditioning set w ---
        # Normalize z
        if z is None:
            z_list = []
        elif isinstance(z, (list, tuple, set)):
            z_list = list(z)
        else:
            raise ValueError("MetaLearner expects adjustment set z to be a list/tuple/set of column names for DataFrame mode")

        # Ensure all elements in z are strings
        for col in z_list:
            if not isinstance(col, str):
                raise ValueError("All elements of adjustment set z must be strings (column names)")

        # Normalize w
        if w is None:
            w_list = []
        elif isinstance(w, (list, tuple, set)):
            w_list = list(w)
        else:
            raise ValueError("MetaLearner expects conditioning set w to be a list/tuple/set of column names for DataFrame mode")

        # Ensure all elements in w are strings
        for col in w_list:
            if not isinstance(col, str):
                raise ValueError("All elements of conditioning set w must be strings (column names)")

        # Check overlap between z and w
        overlap = set(z_list).intersection(set(w_list))
        if len(overlap) > 0:
            raise ValueError(f"Adjustment set z and conditioning set w must not overlap. Overlapping variables: {sorted(overlap)}")

        # Store both naming variants used elsewhere in the file to preserve backward compatibility
        self.adjustment_set = z_list
        self.conditioning_set = w_list

        # keep user-provided prototypes: `model` for outcome-phase (mu) and `model_tau` for tau-phase
        self._base_model = model
        self._base_model_tau = model_tau

    def _is_binary_exposure(self, series: pd.Series) -> bool:
        """Return True if series represents a binary exposure (0/1 or booleans)."""
        vals = set(series.dropna().unique())
        allowed = {0, 1, True, False}
        return vals.issubset(allowed)

    def _is_binary_outcome(self, series: pd.Series) -> bool:
        """Return True if outcome is binary (0/1 or booleans)."""
        vals = set(series.dropna().unique())
        allowed = {0, 1, True, False}
        return vals.issubset(allowed)

    def _prepare_model_for_outcome(self, binary_outcome: bool):
        """Return a fresh model instance suitable for the outcome type.
        If the user provided a base_model that already supports the needed interface, clone it; otherwise raise error.
        """
        # rs = 42 if self.seed is None else self.seed
        # If the user provided a base_model, try to reuse its type when possible
        # try:

        if self._base_model is None: # sanity check
            if binary_outcome:
                prototype = RandomForestClassifier(n_estimators=100)
            else:
                prototype = RandomForestRegressor(n_estimators=100)
        else:
            prototype = clone(self._base_model)
            if not (hasattr(prototype, 'fit') and callable(getattr(prototype, 'fit'))):
                raise TypeError("Provided model does not implement a callable 'fit(X, y)' method")

            if binary_outcome:
                # need classifier with predict_proba
                if not hasattr(prototype, 'predict_proba'):
                    raise TypeError("Provided model does not implement a callable 'predict_proba(X, y)' method")
            else:
                # need regressor
                if not hasattr(prototype, 'predict'):
                    raise TypeError("Provided model does not implement a callable 'predict(X, y)' method")

        if self.seed is not None:
            if hasattr(prototype, "get_params") and callable(getattr(prototype, "get_params")):
                params = prototype.get_params(deep=False)
                if "random_state" in params:
                    prototype.set_params(random_state=self.seed)
                else:
                    warnings.warn(
                        f"{type(prototype).__name__} does not support 'random_state'. "
                        f"The provided seed={self.seed} will be ignored.",
                        UserWarning
                    )
            else:
                warnings.warn(
                    f"{type(prototype).__name__} does not expose get_params(). "
                    f"Could not verify support for 'random_state'. The seed may be ignored.",
                    UserWarning
                )

        return prototype

    def _prepare_model_for_tau(self):
        """Prepare and return a regressor instance for the tau (pseudo-outcome) stage.

        If the user provided `model_tau`, validate it looks like a regressor (must implement
        `fit` and `predict` and should NOT implement `predict_proba`). If absent, return a
        default RandomForestRegressor seeded with `self.seed`.
        """
        # If explicit prototype provided, validate
        if self._base_model_tau is None:
            prototype = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        else:
            prototype = clone(self._base_model_tau)
            if not (hasattr(prototype, 'fit') and callable(getattr(prototype, 'fit'))):
                raise TypeError("Provided model_tau does not implement a callable 'fit(X, y)' method")
            # Heuristic: if it implements predict_proba it's likely a classifier -> not suitable
            if hasattr(prototype, 'predict_proba'):
                raise TypeError("Provided model_tau appears to be a classifier (has predict_proba). XLearner requires a regressor for the tau-stage because pseudo-outcomes are continuous. Provide a regressor (e.g. RandomForestRegressor) or leave model_tau=None to use the default regressor.")
            # ensure it has predict
            if not hasattr(prototype, 'predict'):
                raise TypeError("Provided model_tau does not implement a callable 'predict(X)' method")

        # Try to set random_state on the prototype if possible
        if self.seed is not None:
            if hasattr(prototype, "get_params") and callable(getattr(prototype, "get_params")):
                params = prototype.get_params(deep=False)
                if "random_state" in params:
                    prototype.set_params(random_state=self.seed)
                else:
                    warnings.warn(
                        f"{type(prototype).__name__} does not support 'random_state'. "
                        f"The provided seed={self.seed} will be ignored.",
                        UserWarning
                    )
            else:
                warnings.warn(
                    f"{type(prototype).__name__} does not expose get_params(). "
                    f"Could not verify support for 'random_state'. The seed may be ignored.",
                    UserWarning
                )

        return prototype

    def _predict_outcome(self, model, X: pd.DataFrame, binary_outcome: bool):
        """Abstract prediction: for binary outcomes return predicted probability of class 1 when possible, else fallback to predict; for continuous return predict."""
        if binary_outcome:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    return pd.Series(probs[:, 1], index=X.index)
                else:
                    return pd.Series(probs[:, 0], index=X.index)
            else:
                # fallback: use predict (assumed 0/1)
                return pd.Series(model.predict(X), index=X.index)
        else:
            return pd.Series(model.predict(X), index=X.index)

    def _compute_marginal_effect(self, model, data_base: pd.DataFrame, exposure_col: str = '__exposure__', eps: Optional[float] = None, binary_outcome: bool = False) -> pd.Series:
        """
        Compute per-unit marginal effect (partial derivative w.r.t. exposure) using central finite difference.
        Uses _predict_outcome so it works for classifiers (probability) and regressors.
        Returns a pd.Series aligned with data_base.index.
        """
        # choose eps if not provided: small fraction of exposure std or 1e-3
        if eps is None:
            # estimate scale from the exposure column in data_base
            if exposure_col in data_base.columns:
                exp_vals = data_base[exposure_col].astype(float)
                std = float(exp_vals.std()) if exp_vals.std() > 0 else 1.0
                eps = max(1e-6, 1e-3 * std)
            else:
                eps = 1e-3

        X_plus = data_base.copy()
        X_minus = data_base.copy()
        X_plus[exposure_col] = X_plus[exposure_col] + eps
        X_minus[exposure_col] = X_minus[exposure_col] - eps

        pred_plus = self._predict_outcome(model, X_plus, binary_outcome)
        pred_minus = self._predict_outcome(model, X_minus, binary_outcome)
        derivative = (pred_plus - pred_minus) / (2.0 * eps)
        return derivative

    @abstractmethod
    def run(self, data: pd.DataFrame):
        """ Fit the meta-learner to the data.
        :param data: pd.DataFrame, the input data containing the exposure variable, outcome variable, and covariates.
        :return: dict with keys:
            - 'cate': float, average treatment effect (mean of ITE)
            - 'model': trained model or dict of models
            - optional 'cate_per_w_stratum': dict mapping conditioning strata -> float
        """
        pass


class SLearner(MetaLearners):
    """
    The S-learner trains a single model that includes the treatment variable as a feature. For binary outcomes this will use a classifier and work on predicted probabilities (risk difference interpretation).
    """
    def __init__(self, x: str, y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, seed: Optional[int] = None):
        """
        :param x: exposure variable name (string)
        :param y:  outcome variable name (string)
        :param z:  adjustment set variable names (list of strings)
        :param w:  conditioning set variable names (list of strings)
        :param model:  sklearn-like estimator to use as base learner (regressor/classifier). If None, RandomForest is used.
        :param seed:    optional random seed to control randomness where applicable
        """
        super().__init__(x, y, z, w, model=model, model_tau=None, seed=seed)

    def run(self, data: pd.DataFrame):
        # prepare features: adjsutment + conditioning
        features = self.adjustment_set + self.conditioning_set

        # if no covariates and no conditioning, create empty frame to attach exposure
        if features:
            data_base = data[features].copy()
        else:
            data_base = pd.DataFrame(index=data.index)

        # attach exposure column and fit
        data_base['__exposure__'] = data[self.exposure]
        y = data[self.outcome]

        # detect outcome type and prepare appropriate model
        binary_outcome = self._is_binary_outcome(y)
        model = self._prepare_model_for_outcome(binary_outcome)

        model.fit(data_base, y)

        # decide binary vs continuous exposure
        if self._is_binary_exposure(data[self.exposure]):
            # predict potential outcomes for all units (binary exposure)
            data_X1 = data_base.copy()
            data_X1["__exposure__"] = 1
            data_X0 = data_base.copy()
            data_X0["__exposure__"] = 0
            pred1 = self._predict_outcome(model, data_X1, binary_outcome)
            pred0 = self._predict_outcome(model, data_X0, binary_outcome)
            ite = pred1 - pred0
            cate = float(ite.mean())
        else:
            # continuous exposure: compute marginal effect per unit via central finite difference
            ite = self._compute_marginal_effect(model, data_base, exposure_col='__exposure__', binary_outcome=binary_outcome)
            cate = float(ite.mean())

        # Build result
        result = {"cate": cate,
                  'model': model}

        # If conditioning W provided, compute conditional ATE and ITE per stratum
        if isinstance(self.conditioning_set, (list, tuple)) and len(self.conditioning_set) > 0:
            # group by unique rows of conditioning set
            grouped = data.groupby(self.conditioning_set, dropna=False)
            ate_by_w = {}
            for w_val, idx in grouped.groups.items():
                # idx is index labels for rows in this group
                if len(idx) == 0:
                    continue
                # compute group-specific ITEs using the already predicted values
                grp_ite = ite.loc[idx]
                ate_w = float(grp_ite.mean())
                # canonical key for w_val: if single column, use scalar, else tuple
                key = w_val if not isinstance(w_val, tuple) else tuple(w_val)
                ate_by_w[key] = ate_w
            # return CATE  as dict per stratum
            result['cate_per_w_stratum'] = ate_by_w
        return result


class TLearner(MetaLearners):
    """
    The T-learner trains separate models for treated and control groups. This implementation requires a binary exposure; if the exposure column is not binary a ValueError is raised.
    """
    def __init__(self, x: str, y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, seed: Optional[int] = None):
        super().__init__(x, y, z, w, model=model, model_tau=None, seed=seed)

    def run(self, data: pd.DataFrame):
        # require binary exposure for T-learner
        if not self._is_binary_exposure(data[self.exposure]):
            raise ValueError(f"TLearner requires a binary exposure column '{self.exposure}'; got non-binary values: {set(data[self.exposure].dropna().unique())}")

        features = self.adjustment_set + self.conditioning_set

        # if no covariates and no conditioning, create empty frame to attach exposure
        if features:
            data_base = data[features].copy()
        else:
            data_base = pd.DataFrame(index=data.index)

        y = data[self.outcome]
        treated_mask = data[self.exposure] == 1
        control_mask = ~treated_mask

        # If one group is empty or too small fallback to S-learner style single model
        if treated_mask.sum() < 2 or control_mask.sum() < 2:
            # fallback to S-learner
            slearner = SLearner(self.exposure, self.outcome, self.adjustment_set, self.conditioning_set, model=self._base_model, seed=self.seed)
            return slearner.run(data)

        # if no covariates, fallback to SLearner to ensure a valid feature (exposure) is used
        if len(features) == 0:
            slearner = SLearner(self.exposure, self.outcome, self.adjustment_set, self.conditioning_set,
                                model=self._base_model, seed=self.seed)
            return slearner.run(data)

        # detect outcome type and prepare appropriate models for each group
        binary_outcome = self._is_binary_outcome(y)
        model_t = self._prepare_model_for_outcome(binary_outcome)
        model_c = self._prepare_model_for_outcome(binary_outcome)

        # fit treated model
        model_t.fit(data_base.loc[treated_mask], y.loc[treated_mask])

        # fit control model
        model_c.fit(data_base.loc[control_mask], y.loc[control_mask])

        # predictions for all units
        pred_treated = self._predict_outcome(model_t, data_base, binary_outcome)
        pred_control = self._predict_outcome(model_c, data_base, binary_outcome)
        ite = pred_treated - pred_control
        cate = float(ite.mean())

        # Build result consistent with SLearner: include 'model' key as a dict of models
        result = {
            'cate': cate,
            'model': {'model_t': model_t, 'model_c': model_c}
        }

        # If conditioning W provided, compute conditional ATE and ITE per stratum (like SLearner)
        if isinstance(self.conditioning_set, (list, tuple)) and len(self.conditioning_set) > 0:
            grouped = data.groupby(self.conditioning_set, dropna=False)
            ate_by_w = {}
            for w_val, idx in grouped.groups.items():
                if len(idx) == 0:
                    continue
                grp_ite = ite.loc[idx]
                ate_w = float(grp_ite.mean())
                key = w_val if not isinstance(w_val, tuple) else tuple(w_val)
                ate_by_w[key] = ate_w
            result['cate_per_w_stratum'] = ate_by_w
        return result


class XLearner(MetaLearners):
    """
    The X-learner combines T- and S-learner ideas. This implementation requires a binary exposure; if the exposure is not binary a ValueError is raised. For non-binary exposures users can use `SLearner` which supports marginal-effect estimation.
    """
    def __init__(self, x: str, y: str, z: Optional[Sequence[str]] = None, w: Optional[Sequence[str]] = None, model=None, model_tau=None, seed: Optional[int] = None):
        # XLearner may receive model_tau explicitly; forward both prototypes
        super().__init__(x, y, z, w, model=model, model_tau=model_tau, seed=seed)

    def run(self, data: pd.DataFrame):
        # require binary exposure for X-learner
        if not self._is_binary_exposure(data[self.exposure]):
            raise ValueError(f"XLearner requires a binary exposure column '{self.exposure}'; got non-binary values: {set(data[self.exposure].dropna().unique())}")

        features = self.adjustment_set + self.conditioning_set

        # if no covariates and no conditioning, create empty frame to attach exposure
        if features:
            data_base = data[features].copy()
        else:
            data_base = pd.DataFrame(index=data.index)

        y = data[self.outcome]
        treated_mask = data[self.exposure] == 1
        control_mask = ~treated_mask

        # Stage 1: T-learner to estimate mu1 and mu0
        if treated_mask.sum() < 2 or control_mask.sum() < 2:
            # fallback to T-learner which itself falls back to S-learner when needed
            tlearner = TLearner(self.exposure, self.outcome, self.adjustment_set, self.conditioning_set, model=self._base_model, seed=self.seed)
            return tlearner.run(data)

        # if no covariates, fallback to SLearner to ensure a valid feature (exposure) is used
        if len(features) == 0:
            slearner = SLearner(self.exposure, self.outcome, self.adjustment_set, self.conditioning_set,
                                model=self._base_model, seed=self.seed)
            return slearner.run(data)

        # detect outcome type and prepare models
        binary_outcome = self._is_binary_outcome(y)
        model_t = self._prepare_model_for_outcome(binary_outcome)
        model_c = self._prepare_model_for_outcome(binary_outcome)

        model_t.fit(data_base.loc[treated_mask], y.loc[treated_mask])
        model_c.fit(data_base.loc[control_mask], y.loc[control_mask])

        mu1 = self._predict_outcome(model_t, data_base, binary_outcome)
        mu0 = self._predict_outcome(model_c, data_base, binary_outcome)

        # Stage 2: compute pseudo-outcomes
        # For treated units: pseudo = y - mu0(x)
        D_treated = y.loc[treated_mask] - mu0.loc[treated_mask]
        # For control units: pseudo = mu1(x) - y
        D_control = mu1.loc[control_mask] - y.loc[control_mask]

        # Fit models to pseudo-outcomes: use model_tau (must be regressor)
        tau_model_treated = self._prepare_model_for_tau()
        tau_model_treated.fit(data_base.loc[treated_mask], D_treated)

        tau_model_control = self._prepare_model_for_tau()
        tau_model_control.fit(data_base.loc[control_mask], D_control)

        # Final ITE: average of the two models' predictions
        tau_t_pred = self._predict_outcome(tau_model_treated, data_base, False)
        tau_c_pred = self._predict_outcome(tau_model_control, data_base, False)
        ite = 0.5 * (tau_t_pred + tau_c_pred)
        cate = float(ite.mean())

        # Build result consistent with SLearner: include 'model' key as a dict
        result = {
            'cate': cate,
            'model': {
                'model_t': model_t,
                'model_c': model_c,
                'tau_model_treated': tau_model_treated,
                'tau_model_control': tau_model_control
            }
        }

        # If conditioning W provided, compute conditional ATE per stratum
        if isinstance(self.conditioning_set, (list, tuple)) and len(self.conditioning_set) > 0:
            grouped = data.groupby(self.conditioning_set, dropna=False)
            ate_by_w = {}
            for w_val, idx in grouped.groups.items():
                if len(idx) == 0:
                    continue
                grp_ite = ite.loc[idx]
                ate_w = float(grp_ite.mean())
                key = w_val if not isinstance(w_val, tuple) else tuple(w_val)
                ate_by_w[key] = ate_w
            result['cate_per_w_stratum'] = ate_by_w
        return result

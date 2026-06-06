from abc import abstractmethod

import numpy as np
import sklearn
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner

### AutoML Hyperparam Optimization ###
from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .base import BaselineModel


def get_hpo_propensity_model(X: np.ndarray, t: np.ndarray) -> sklearn.base.BaseEstimator:
    automl_settings = {
        "time_budget": 900,  # in seconds
        "task": "classification",
        "eval_method": "cv",
        "n_splits": 3,
        "verbose": 0,
        "estimator_list": [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "rf",
            "kneighbor",
            "extra_tree",
            "lrl1",
            "lrl2",
        ],
        "early_stop": True,
    }

    automl = AutoML()
    automl.fit(X_train=X, y_train=t, **automl_settings)
    return sklearn.base.clone(automl.model.estimator)


def get_hpo_regression_model(X: np.ndarray, y: np.ndarray) -> sklearn.base.BaseEstimator:
    automl_settings = {
        "time_budget": 900,  # in seconds
        "task": "regression",
        "eval_method": "cv",
        "n_splits": 3,
        "verbose": 0,
        "estimator_list": [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "rf",
            "kneighbor",
            "extra_tree",
        ],
        "early_stop": True,
    }

    automl = AutoML()
    automl.fit(X_train=X, y_train=y, **automl_settings)
    return sklearn.base.clone(automl.model.estimator)


### AutoML Hyperparam Optimization (Ended) ###


class EconMLBaseline(BaselineModel):
    def __init__(self, hpo: bool = True):
        super().__init__(hpo)

    def _scale_x(self, X_tr, X_te):
        scaler_x = StandardScaler().fit(X_tr)
        return (scaler_x.transform(X_tr).astype("float32"), scaler_x.transform(X_te).astype("float32"), scaler_x)

    def _scale_y(self, y_tr):
        scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).ravel().astype("float32")
        return y_tr_s, scaler_y

    def _unscale_res(self, res, scaler_y):
        return res * scaler_y.scale_[0] if scaler_y is not None else res

    @abstractmethod
    def _get_model(self, X, t, y): ...

    def estimate_ate(self, X, t, y):
        # — scale
        X, _, _ = self._scale_x(X, X)
        y, scaler_y = self._scale_y(y)

        model = self._get_model(X, t, y)
        model.fit(Y=y, T=t, X=X)
        ate_pred = model.ate(X)
        return self._unscale_res(ate_pred, scaler_y)

    def estimate_cate(self, X_train, t_train, y_train, X_test):
        # — scale
        X_train, X_test, _ = self._scale_x(X_train, X_test)
        y_train, scaler_y = self._scale_y(y_train)

        model = self._get_model(X_train, t_train, y_train)
        model.fit(Y=y_train, T=t_train, X=X_train)
        cate_pred = model.effect(X_test)
        return self._unscale_res(cate_pred, scaler_y)

    def estimate_ci(self, X_train, t_train, y_train, X_test, alpha=0.05):
        # — scale
        X_train, X_test, _ = self._scale_x(X_train, X_test)
        y_train, scaler_y = self._scale_y(y_train)

        model = self._get_model(X_train, t_train, y_train)
        if not hasattr(model, "effect_interval"):
            raise ValueError("The model does not support confidence intervals.")
        model.fit(Y=y_train, T=t_train, X=X_train)
        lower_bound, upper_bound = model.effect_interval(X_test, T0=0, T1=1, alpha=alpha)
        return self._unscale_res(lower_bound, scaler_y), self._unscale_res(upper_bound, scaler_y)


class ForestDMLBaseline(EconMLBaseline):

    def _get_model(self, X, t, y):
        if self.hpo:
            model_y = get_hpo_regression_model(X, y)
            model_t = get_hpo_propensity_model(X, t)
            model = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                n_estimators=1000,
                discrete_treatment=True,
                cv=5,
                featurizer=PolynomialFeatures(degree=3),
            )
            model.tune(X=X, Y=y, T=t)
        else:
            model = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                model_t=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                n_estimators=1000,
                discrete_treatment=True,
                cv=5,
                featurizer=PolynomialFeatures(degree=3),
            )
        return model


class XLearnerBaseline(EconMLBaseline):
    def _get_model(self, X, t, y):
        if self.hpo:
            X_0, y_0 = X[t == 0], y[t == 0]
            X_1, y_1 = X[t == 1], y[t == 1]
            model_0 = get_hpo_regression_model(X_0, y_0) if len(X_0) > 0 else get_hpo_regression_model(X, y)
            model_1 = get_hpo_regression_model(X_1, y_1) if len(X_1) > 0 else get_hpo_regression_model(X, y)
            propensity_model = get_hpo_propensity_model(X, t)

            model = XLearner(
                models=[model_0, model_1],
                propensity_model=propensity_model,
            )
        else:
            model = XLearner(
                models=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                propensity_model=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
            )
        return model


class TLearnerBaseline(EconMLBaseline):
    def _get_model(self, X, t, y):
        if self.hpo:
            X_0, y_0 = X[t == 0], y[t == 0]
            X_1, y_1 = X[t == 1], y[t == 1]
            model_0 = get_hpo_regression_model(X_0, y_0) if len(X_0) > 0 else get_hpo_regression_model(X, y)
            model_1 = get_hpo_regression_model(X_1, y_1) if len(X_1) > 0 else get_hpo_regression_model(X, y)
            model = TLearner(models=[model_0, model_1])
        else:
            model = TLearner(
                models=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
            )
        return model


class SLearnerBaseline(EconMLBaseline):
    def _get_model(self, X, t, y):
        if self.hpo:
            X_with_t = np.column_stack([X, t])
            overall_model = get_hpo_regression_model(X_with_t, y)
            model = SLearner(overall_model=overall_model)
        else:
            model = SLearner(
                overall_model=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
            )
        return model


class ForestDRLearnerBaseline(EconMLBaseline):
    def _get_model(self, X, t, y):
        if self.hpo:
            model_regression = get_hpo_regression_model(X, y)
            model_propensity = get_hpo_propensity_model(X, t)
            model = ForestDRLearner(
                model_regression=model_regression,
                model_propensity=model_propensity,
                n_estimators=1000,
                cv=5,
                featurizer=PolynomialFeatures(degree=3),
            )
        else:
            model = ForestDRLearner(
                model_regression=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                model_propensity=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                n_estimators=1000,
                cv=5,
                featurizer=PolynomialFeatures(degree=3),
            )
        return model


class DALearnerBaseline(EconMLBaseline):

    def _get_model(self, X, t, y):
        if self.hpo:
            X_0, y_0 = X[t == 0], y[t == 0]
            X_1, y_1 = X[t == 1], y[t == 1]
            model_0 = get_hpo_regression_model(X_0, y_0) if len(X_0) > 0 else get_hpo_regression_model(X, y)
            model_1 = get_hpo_regression_model(X_1, y_1) if len(X_1) > 0 else get_hpo_regression_model(X, y)
            final_model = get_hpo_regression_model(X, y)
            propensity_model = get_hpo_propensity_model(X, t)

            model = DomainAdaptationLearner(
                models=[model_0, model_1],
                final_models=final_model,
                propensity_model=propensity_model,
            )
        else:
            model = DomainAdaptationLearner(
                models=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                final_models=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
                propensity_model=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=max(1, X.shape[0] // 100),
                    n_jobs=-1,
                ),
            )
        return model

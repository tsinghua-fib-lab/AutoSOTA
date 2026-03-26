import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaselineModel
from .econml import get_hpo_propensity_model


class IPWBaseline(BaselineModel):
    def __init__(self, hpo: bool = True):
        super().__init__(hpo)

    def _get_ate(self, X, t, y):
        if self.hpo:
            classifier = get_hpo_propensity_model(X, t)
        else:
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=int(X.shape[0] / 100),
            )
        classifier.fit(X, t)
        propensities = classifier.predict_proba(X)[:, 1].clip(1e-3, 1 - 1e-3)
        treatment_group = t == 1

        weights = np.where(treatment_group, 1.0 / propensities, 1.0 / (1 - propensities))
        treated_mean = np.average(y[treatment_group], weights=weights[treatment_group])
        control_mean = np.average(y[~treatment_group], weights=weights[~treatment_group])

        ate = treated_mean - control_mean
        return ate

    def estimate_ate(self, X, t, y):
        return self._get_ate(X, t, y)

    def estimate_cate(self, X_train, t_train, y_train, X_test):
        ate = self._get_ate(X_train, t_train, y_train)
        return np.ones(X_test.shape[0]) * ate

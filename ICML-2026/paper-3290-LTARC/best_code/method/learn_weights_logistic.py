"""
Learn p(s|x) using logistic regression to get the weights
w(x) = p(s=1|x)/p(s=0|x) used in weighted conformal.
"""
import numpy as np

from sklearn.linear_model import LogisticRegression


class LearnProbabilitiesLogistic:
    """Logistic regression model for estimating propensity scores p(a|x) and p(s|x)."""

    def __init__(self, x, a, x_new):
        if x_new != 0:
            xtot = np.concatenate([x_new, x])
            s = np.concatenate([np.zeros(len(x_new)), np.ones(len(x))])
            self.model_sampling = self.train_model(xtot, s)
            self.p_s1 = len(x) / len(xtot)
        self.model_decision = self.train_model(x, a)

    def get_p_s_x(self, x, s):
        proba = self.model_sampling.predict_proba(x)
        p_s_x = proba[np.arange(0, len(x)), s]
        return p_s_x

    def get_p_a_x(self, x, a):
        proba = self.model_decision.predict_proba(x)
        p_a_x = proba[np.arange(0, len(x)), a]
        return p_a_x

    def get_p_s1(self):
        return self.p_s1

    def train_model(self, x, cat):
        clf = LogisticRegression(
            penalty="l2", C=0.01, solver="saga", random_state=132
        ).fit(x, cat)
        return clf

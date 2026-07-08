from cmath import isnan

import numpy as np
from numpy.testing import assert_almost_equal


class SoftMaxQ_Policy:
    def __init__(self, actions, Q_star, beta):
        self.actions = actions
        self.Q_star = Q_star
        self.beta = beta

    def _get_probs(self, state_id):
        """ """
        beta = self.beta

        total = np.sum(
            [np.exp(beta * self.Q_star[state_id, action]) for action in self.actions]
        )
        probs = [
            np.exp(beta * self.Q_star[state_id, action]) / total
            for action in self.actions
        ]

        assert np.any(not isnan(p) for p in probs), print(self.Q_star)
        assert_almost_equal(np.sum(probs), 1)

        return probs

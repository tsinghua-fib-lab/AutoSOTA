import numpy as np
from numpy.testing import assert_almost_equal


class SoftMaxD_Policy:
    def __init__(self, actions, d_star, beta):
        self.actions = actions
        self.d_star = d_star
        self.beta = beta

    def _get_probs(self, state_id):
        """ """
        beta = self.beta

        total = np.sum(
            [np.exp(beta * self.d_star[state_id, action]) for action in self.actions]
        )
        probs = [
            np.exp(beta * self.d_star[state_id, action]) / total
            for action in self.actions
        ]

        assert_almost_equal(np.sum(probs), 1), f"probs: {probs}"

        return probs

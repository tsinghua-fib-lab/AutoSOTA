import numpy as np
from numpy.testing import assert_almost_equal


class RandomizedD_Policy:
    def __init__(self, actions, d_star):
        self.actions = actions
        self.d_star = d_star

    def _get_probs(self, state_id):
        """ """
        if not any(self.d_star[state_id]):
            probs = [1 / len(self.actions)] * len(self.actions)
            return probs

        total = np.sum([self.d_star[state_id, action] for action in self.actions])
        probs = [self.d_star[state_id, action] / total for action in self.actions]

        assert_almost_equal(np.sum(probs), 1)

        return probs

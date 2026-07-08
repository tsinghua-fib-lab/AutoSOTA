import numpy as np
from numpy.testing import assert_almost_equal


class Tabular:
    def __init__(self, actions, pi):
        self.actions = actions
        self.pi = pi

    def _get_probs(self, state_id):
        """ """
        probs = self.pi[state_id]

        assert_almost_equal(np.sum(probs), 1)

        return probs

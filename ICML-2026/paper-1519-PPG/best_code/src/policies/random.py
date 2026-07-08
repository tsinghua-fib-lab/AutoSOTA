import numpy as np
from numpy.testing import assert_almost_equal


class Random_Policy:
    def __init__(self, actions):
        self.actions = actions

    def _get_probs(self, state_id):
        """ """
        probs = np.full_like(self.actions, 1 / len(self.actions), dtype="float64")

        assert_almost_equal(np.sum(probs), 1), f"probs: {probs}"

        return probs

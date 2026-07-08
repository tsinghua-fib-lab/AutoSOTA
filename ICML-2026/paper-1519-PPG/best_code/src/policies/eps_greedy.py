import numpy as np
from numpy.testing import assert_almost_equal


class Eps_Greedy_Policy:
    def __init__(self, actions, Q_star, eps=0.1):
        self.actions = actions
        self.Q_star = Q_star
        self.eps = eps

    def _get_probs(self, state_id):
        """ """
        action = np.argmax([self.Q_star[state_id, action] for action in self.actions])
        actions_num = len(self.actions)
        probs = np.full(actions_num, self.eps / (actions_num - 1))
        probs[action] = 1 - self.eps

        assert_almost_equal(np.sum(probs), 1)

        return probs

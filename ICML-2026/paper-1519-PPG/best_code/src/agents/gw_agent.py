from src.policies.random import Random_Policy


class Agent:
    def __init__(self, id, actions):
        self.id = id
        self.actions = actions
        # initialize policy to random
        self.policy = Random_Policy(actions)

    def _get_probs(self, state_id):
        """ """

        return self.policy._get_probs(state_id)

    def take_action(self, state_id, rng):
        """ """
        probs = self._get_probs(state_id)
        action = rng.choice(self.actions, p=probs)

        return action

import numpy as np


class StrategySpace:
    def projection(self, strategy):
        raise NotImplementedError()


class ProbabilitySimplexStrategySpace(StrategySpace):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def random_strategy(self):
        k = np.random.exponential(scale=1.0, size=self.num_actions)
        return k / k.sum()

    def uniform_strategy(self):
        return np.ones(self.num_actions) / self.num_actions

    def projection(self, strategy):
        u = sorted(strategy, reverse=True)
        cumsum_u = np.cumsum(u)
        ids = np.arange(len(u))[u + 1 / (np.arange(len(strategy)) + 1) * (1 - cumsum_u) > 0]
        rho = np.max(ids)
        lamb = 1 / (rho + 1) * (1 - cumsum_u[rho])
        projected_strategy = np.maximum(strategy + lamb, 0)
        return projected_strategy / projected_strategy.sum()

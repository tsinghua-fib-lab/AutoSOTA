import numpy as np

class Bandit:
    def __init__(self, distribution='gaussian', means=None, stds=None, seed=None):
        self.means = np.array(means, dtype=float)
        self.stds  = np.array(stds, dtype=float)
        self.distribution = distribution
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def pull(self, i: int) -> float:
        if self.distribution == 'gaussian':
            return self.rng.normal(self.means[i], self.stds[i])
        elif self.distribution == 'bernoulli':
            return float(self.rng.random() < self.means[i])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    @property
    def K(self) -> int:
        return self.means.shape[0]

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.means))

    @property
    def best_mean(self) -> float:
        return float(np.max(self.means))
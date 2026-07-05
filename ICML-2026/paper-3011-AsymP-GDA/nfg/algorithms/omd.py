import numpy as np

from .md import MD


class OMD(MD):
    def __init__(self, strategy_space, regularizer, learning_rate, **kwargs):
        super().__init__(strategy_space, regularizer, learning_rate, **kwargs)
        self.strategy_hat = self.strategy.copy()

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        gradient = self._calc_gradient()[1]
        if self.regularizer == "euclidean":
            self.strategy_hat = self.strategy_space.projection(self.strategy_hat + self.learning_rate * gradient)
            self.strategy = self.strategy_space.projection(self.strategy_hat + self.learning_rate * gradient)
        elif self.regularizer == "entropy":
            self.strategy_hat = self.strategy_hat * np.exp(self.learning_rate * gradient)
            self.strategy_hat /= np.sum(self.strategy_hat)
            self.strategy = self.strategy_hat * np.exp(self.learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        else:
            raise RuntimeError("Illegal regularizer")
        self.average_strategy = (self.n * self.average_strategy + self.strategy) / (self.n + 1)
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient


class OGD(OMD):
    """Optimistic gradient descent: optimistic MD with the squared Euclidean regularizer."""

    def __init__(self, strategy_space, learning_rate, **kwargs):
        super().__init__(strategy_space, "euclidean", learning_rate, **kwargs)


class OMWU(OMD):
    """Optimistic multiplicative weights update: optimistic MD with the entropy regularizer."""

    def __init__(self, strategy_space, learning_rate, **kwargs):
        super().__init__(strategy_space, "entropy", learning_rate, **kwargs)

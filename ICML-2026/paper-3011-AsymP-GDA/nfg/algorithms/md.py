import interface
import numpy as np


class MD:
    def __init__(self, strategy_space, regularizer, learning_rate, **kwargs):
        assert not (
            regularizer == "entropy"
            and not isinstance(strategy_space, interface.strategy.ProbabilitySimplexStrategySpace)
        )
        self.strategy_space = strategy_space
        self.strategy = strategy_space.random_strategy() if kwargs["random_init"] else strategy_space.uniform_strategy()
        num_actions = strategy_space.num_actions
        self.regularizer = regularizer
        self.cum_gradient = np.zeros(num_actions)
        self.gradient = np.zeros(num_actions)
        self.learning_rate = learning_rate
        self.average_strategy = np.zeros(num_actions)
        self.n = 0

    def name(self):
        alg_name = self.__class__.__name__
        alg_name += f"_lr{self.learning_rate}"
        return alg_name

    def _md(self, cum_gradient, gradient):
        if self.regularizer == "euclidean":
            self.strategy = self.strategy_space.projection(self.strategy + self.learning_rate * gradient)
        elif self.regularizer == "entropy":
            self.strategy = self.strategy * np.exp(self.learning_rate * gradient)
            self.strategy /= np.sum(self.strategy)
        else:
            raise RuntimeError("Illegal regularizer")

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        self._md(*self._calc_gradient())
        self.average_strategy = (self.n * self.average_strategy + self.strategy) / (self.n + 1)
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient

    def regularized_gap(self, gradient, strategy):
        return max(gradient) - gradient @ strategy


class GD(MD):
    """Gradient descent: mirror descent with the squared Euclidean regularizer."""

    def __init__(self, strategy_space, learning_rate, **kwargs):
        super().__init__(strategy_space, "euclidean", learning_rate, **kwargs)


class MWU(MD):
    """Multiplicative weights update: mirror descent with the entropy regularizer."""

    def __init__(self, strategy_space, learning_rate, **kwargs):
        super().__init__(strategy_space, "entropy", learning_rate, **kwargs)

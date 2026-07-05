from .md import GD


class PGD(GD):
    """Perturbed gradient descent: GD with the squared Euclidean payoff perturbation."""

    def __init__(
        self,
        strategy_space,
        learning_rate,
        perturbation_strength,
        **kwargs,
    ):
        super().__init__(strategy_space, learning_rate, **kwargs)
        self.perturbation_strength = perturbation_strength

    def name(self):
        alg_name = self.__class__.__name__
        alg_name += f"_mu{self.perturbation_strength}"
        alg_name += f"_lr{self.learning_rate}"
        return alg_name

    def add_gradient(self, gradient):
        perturbation = -self.perturbation_strength * self.strategy
        self.cum_gradient += gradient + perturbation
        self.gradient = gradient + perturbation

    def _gradient_of_perturbed_utility(self, strategy):
        return -self.perturbation_strength * strategy

    def regularized_gap(self, gradient, strategy):
        return (
            max(gradient + self._gradient_of_perturbed_utility(strategy))
            - gradient @ strategy
            + self.perturbation_strength * strategy @ strategy
        )

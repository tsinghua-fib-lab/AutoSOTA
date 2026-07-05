import LiteEFG

from .base_solver import BaseSolver


class SymPDGDA(BaseSolver):
    """Symmetrically perturbed DGDA: DGDA with the squared Euclidean payoff perturbation on both players."""

    def __init__(self, eta=0.1, mu=0.1):
        super().__init__()
        self.eta = eta
        self.mu = mu
        self.timestep = 0

        with LiteEFG.backward(is_static=True):
            ev = LiteEFG.const(size=1, val=0.0)

            self.mu = LiteEFG.const(1, mu)
            self.eta_coef = 1.0 / eta
            self.coef = self.mu
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)

        with LiteEFG.backward():
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility
            prev_u = self.u.copy()
            self._update(gradient, self.u, prev_u)
            self._get_ev(gradient, ev, self.u, prev_u)

        print("===============Graph is ready for SymPDGDA===============")
        print(f"eta: {self.eta:f}")
        print("=====================================================\n")

    def _get_ev(self, gradient, ev, strategy, ref_strategy):
        ev.inplace(
            LiteEFG.dot(gradient, strategy)
            - LiteEFG.euclidean(strategy - ref_strategy) * self.eta_coef
            - (LiteEFG.dot(ref_strategy, strategy)) * self.mu
        )

    def _update(self, gradient, upd_u, ref_u):
        gradient.inplace(gradient - ref_u * self.mu)
        gradient_div = gradient / self.eta_coef
        upd_u.inplace(ref_u + gradient_div)
        upd_u.inplace(upd_u.project(distance="L2"))

    def update_graph(self, env: LiteEFG.Environment) -> None:
        env.update(self.u, upd_player=1)
        env.update(self.u, upd_player=2)

    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.u

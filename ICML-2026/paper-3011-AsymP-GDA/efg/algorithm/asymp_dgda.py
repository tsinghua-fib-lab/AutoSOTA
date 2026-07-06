import LiteEFG

from .base_solver import BaseAsymSolver


class AsymPDGDA(BaseAsymSolver):
    """Asymmetrically perturbed DGDA: DGDA with the squared Euclidean payoff perturbation on one player."""

    def __init__(self, eta=0.1, mu=0.1):
        self.eta = eta
        self.mu = mu
        # we must call super init to initialize solvers after setting parameters
        super().__init__()

    def create_solver(self, player_id: int) -> LiteEFG.Graph:
        mus = [0.0, 0.0]
        mus[player_id] = self.mu
        return _AsymPDGDA(
            eta=self.eta,
            mus=mus,
        )


class _AsymPDGDA(LiteEFG.Graph):
    def __init__(self, eta=0.1, mus=(0.1, 0.1)):
        super().__init__()
        self.eta = eta
        self.mus = mus
        self.timestep = 0

        with LiteEFG.backward(is_static=True):
            ev = LiteEFG.const(size=1, val=0.0)
            self.mu1 = LiteEFG.const(1, self.mus[0])
            self.mu2 = LiteEFG.const(1, self.mus[1])
            self.eta_coef = 1.0 / eta
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)

        with LiteEFG.backward(color=0):
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility
            prev_u = self.u.copy()
            self._update(gradient, self.u, prev_u, self.mu1)
            self._get_ev(gradient, ev, self.u, prev_u, self.mu1)

        with LiteEFG.backward(color=1):
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility
            prev_u = self.u.copy()
            self._update(gradient, self.u, prev_u, self.mu2)
            self._get_ev(gradient, ev, self.u, prev_u, self.mu2)

        print("===============Graph is ready for AsymPDGDA===============")
        print(f"eta: {self.eta:f}, mus: {self.mus}")
        print("=====================================================\n")

    def _get_ev(self, gradient, ev, strategy, ref_strategy, mu):
        ev.inplace(
            LiteEFG.dot(gradient, strategy)
            - LiteEFG.euclidean(strategy - ref_strategy) * self.eta_coef
            - (LiteEFG.dot(ref_strategy, strategy)) * mu
        )

    def _update(self, gradient, upd_u, ref_u, mu):
        gradient.inplace(gradient - ref_u * mu)
        gradient_div = gradient / self.eta_coef
        upd_u.inplace(ref_u + gradient_div)
        upd_u.inplace(upd_u.project(distance="L2"))

    def update_graph(self, env: LiteEFG.Environment) -> None:
        env.update(self.u, upd_player=1, upd_color=[0])
        env.update(self.u, upd_player=2, upd_color=[1])

    def current_strategy(self) -> list[LiteEFG.GraphNode]:
        return self.u

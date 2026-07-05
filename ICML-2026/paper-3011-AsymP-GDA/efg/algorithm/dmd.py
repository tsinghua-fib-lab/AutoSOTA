from typing import Literal

import LiteEFG

from .base_solver import BaseSolver


class DMD(BaseSolver):
    def __init__(
        self,
        eta=0.1,
        regularizer: Literal["euclidean", "entropy"] = "euclidean",
    ):
        super().__init__()
        self.eta = eta
        self.regularizer = regularizer

        with LiteEFG.backward(is_static=True):
            ev = LiteEFG.const(size=1, val=0.0)

            self.coef = 1.0 / eta
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)

        with LiteEFG.backward():
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility
            prev_u = self.u.copy()
            self._update(gradient, self.u, prev_u)
            self._get_ev(gradient, ev, self.u, prev_u)

        print(f"===============Graph is ready for {self.__class__.__name__}===============")
        print(f"eta: {self.eta:f}, regularizer: {self.regularizer}")
        print("=====================================================\n")

    def _get_ev(self, gradient, ev, strategy, ref_strategy):
        if self.regularizer == "euclidean":
            ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.euclidean(strategy - ref_strategy) * self.coef)
        else:
            kl = LiteEFG.dot((strategy / ref_strategy).log(), strategy) * self.coef
            ev.inplace(LiteEFG.dot(gradient, strategy) - kl)

    def _update(self, gradient, upd_u, ref_u):
        gradient_div = gradient / self.coef

        if self.regularizer == "euclidean":
            upd_u.inplace(ref_u + gradient_div)
            upd_u.inplace(upd_u.project(distance="L2"))

        else:
            upd_u.inplace(ref_u.log() + gradient_div)
            upd_u.inplace(upd_u - upd_u.max())
            upd_u.inplace(upd_u.exp())
            upd_u.inplace(upd_u.project(distance="KL"))

    def update_graph(self, env: LiteEFG.Environment) -> None:
        env.update(self.u)

    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.u


class DGDA(DMD):
    """Dilated gradient descent ascent: dilated MD with the Euclidean regularizer."""

    def __init__(self, eta=0.1):
        super().__init__(eta=eta, regularizer="euclidean")


class DMWU(DMD):
    """Dilated multiplicative weights update: dilated MD with the entropy regularizer."""

    def __init__(self, eta=0.1):
        super().__init__(eta=eta, regularizer="entropy")

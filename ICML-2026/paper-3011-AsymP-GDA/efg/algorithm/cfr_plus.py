#######################################################
# Counterfactual Regret Minimization+ (CFR+)
# Zinkevich, Martin, Michael Johanson, Michael Bowling, and Carmelo Piccione.
# "Regret minimization in games with incomplete information."
# Advances in neural information processing systems (2007).
#######################################################

import LiteEFG

from .base_solver import BaseSolver


class CFRPlus(BaseSolver):
    """CFR+: CFR with regret-matching+."""

    def __init__(self):
        super().__init__()

        with LiteEFG.backward(is_static=True):
            expectation = LiteEFG.const(1, 0.0)
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.regret_buffer = LiteEFG.const(self.action_set_size, 0.0)

        # RM+
        with LiteEFG.backward():
            counterfactual_value = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility
            expectation.inplace(LiteEFG.dot(counterfactual_value, self.strategy))
            self.regret_buffer.inplace(LiteEFG.maximum(self.regret_buffer + counterfactual_value - expectation, 0.0))
            self.strategy.inplace(LiteEFG.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))

        print("===============Graph is ready for CFR+===============")
        print()
        print("====================================================\n")

    def update_graph(self, env: LiteEFG.Environment) -> None:
        env.update(self.strategy, upd_player=1)
        env.update(self.strategy, upd_player=2)

    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.strategy

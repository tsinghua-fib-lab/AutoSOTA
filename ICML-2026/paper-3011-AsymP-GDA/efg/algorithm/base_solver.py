from enum import Enum

import LiteEFG
import pyspiel


class StrategyType(str, Enum):
    LAST_ITERATE = "last-iterate"
    AVG_ITERATE = "avg-iterate"
    BEST_ITERATE = "best-iterate"


def counterfactual_values(ind_regrets, utilities):
    out = [0.0 for _ in ind_regrets]
    for i in range(len(ind_regrets)):
        other_idx = -(i + 1)
        out[i] = ind_regrets[other_idx] + utilities[other_idx]
    return out


class BaseSolver(LiteEFG.Graph):
    def __init__(self):
        super().__init__()
        self._env = None

    def setup(self, game_env, traverse_type):
        game = pyspiel.load_game(game_env)
        self._env = LiteEFG.OpenSpielEnv(game, traverse_type=traverse_type, regenerate=False)
        self._env.set_graph(self)

    def step(self, update_best=False):
        self.update_graph(self._env)
        self._env.update_strategy(self.current_strategy(), update_best=update_best)

    def compute_metrics(self):
        env = self._env
        strategy = self.current_strategy()
        last_ind_regrets = env.exploitability(strategy, StrategyType.LAST_ITERATE)
        avg_ind_regrets = env.exploitability(strategy, StrategyType.AVG_ITERATE)
        return {
            "last_exploitability": sum(last_ind_regrets),
            "avg_exploitability": sum(avg_ind_regrets),
        }


class BaseAsymSolver:
    def __init__(self):
        self.solvers = [
            self.create_solver(player_id=0),
            self.create_solver(player_id=1),
        ]
        self._envs = None

    def create_solver(self, player_id: int) -> LiteEFG.Graph:
        raise NotImplementedError

    def setup(self, game_env, traverse_type):
        self._envs = [
            LiteEFG.OpenSpielEnv(
                pyspiel.load_game(game_env),
                traverse_type=traverse_type,
                regenerate=False,
            )
            for _ in range(2)
        ]
        for solver, env in zip(self.solvers, self._envs, strict=True):
            env.set_graph(solver)

    def step(self, update_best=False):
        for env, solver in zip(self._envs, self.solvers, strict=True):
            solver.update_graph(env)
            env.update_strategy(solver.current_strategy(), update_best=update_best)

    def compute_metrics(self):
        last_exploitability = 0.0
        avg_exploitability = 0.0
        for i_e, (env, solver) in enumerate(zip(self._envs, self.solvers, strict=True)):
            strategy = solver.current_strategy()
            last_ind_regrets = env.exploitability(strategy, StrategyType.LAST_ITERATE)
            avg_ind_regrets = env.exploitability(strategy, StrategyType.AVG_ITERATE)
            last_cfv = counterfactual_values(last_ind_regrets, env.utility(strategy, StrategyType.LAST_ITERATE))
            avg_cfv = counterfactual_values(avg_ind_regrets, env.utility(strategy, StrategyType.AVG_ITERATE))
            last_exploitability += last_cfv[i_e]
            avg_exploitability += avg_cfv[i_e]
        return {
            "last_exploitability": last_exploitability,
            "avg_exploitability": avg_exploitability,
        }

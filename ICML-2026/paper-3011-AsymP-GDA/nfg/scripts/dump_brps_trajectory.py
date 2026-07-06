import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from algorithms.md import GD
from algorithms.pgd import PGD
from games.matrix_game import biased_rps

INITIAL_STRATEGIES = [
    np.array([0.05, 0.9, 0.05]),
    np.array([0.1, 0.1, 0.8]),
    np.array([0.1, 0.3, 0.6]),
]


def _make_players(alg: str, mu: float, lr: float, strategy_spaces):
    learner_kwargs = {"learning_rate": lr, "random_init": False}
    if alg == "gda":
        return [GD(strategy_spaces[i], **learner_kwargs) for i in range(2)]
    if alg == "symp_gda":
        return [PGD(strategy_spaces[i], perturbation_strength=mu, **learner_kwargs) for i in range(2)]
    if alg == "asymp_gda":
        return [
            PGD(strategy_spaces[0], perturbation_strength=mu, **learner_kwargs),
            GD(strategy_spaces[1], **learner_kwargs),
        ]
    raise ValueError(f"unknown algorithm: {alg}")


def _run(game, players, num_steps: int):
    n_players = game.num_players()
    history_x = [players[0].strategy.copy()]
    history_y = [players[1].strategy.copy()]
    for _ in range(num_steps):
        for i in range(n_players):
            strategies = [p.strategy.copy() for p in players]
            gradient = game.full_feedback(strategies)
            players[i].add_gradient(gradient[i])
            players[i].calc_strategy()
        history_x.append(players[0].strategy.copy())
        history_y.append(players[1].strategy.copy())
    return np.array(history_x), np.array(history_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--mus", type=float, nargs="+", default=[1.5])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["gda", "symp_gda", "asymp_gda"],
        choices=["gda", "symp_gda", "asymp_gda"],
    )
    parser.add_argument("--output-dir", default="logs/brps_trajectory")
    args = parser.parse_args()

    game, _ = biased_rps()
    strategy_spaces = game.strategy_classes()

    output_dir = Path(args.output_dir)
    for alg, mu in product(args.algorithms, args.mus):
        for init_idx, x0 in enumerate(INITIAL_STRATEGIES):
            players = _make_players(alg, mu, args.learning_rate, strategy_spaces)
            for p in players:
                p.strategy = x0.copy()

            xs, ys = _run(game, players, args.num_steps)
            df = pd.DataFrame(
                {
                    "t": np.arange(xs.shape[0]),
                    **{f"x_{k}": xs[:, k] for k in range(xs.shape[1])},
                    **{f"y_{k}": ys[:, k] for k in range(ys.shape[1])},
                }
            )

            out_path = output_dir / alg / f"mu{mu}" / f"init{init_idx}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()

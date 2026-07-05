from algorithm.base_solver import StrategyType
from tqdm import tqdm

from runner.logger import TrainingLogger


def train(solver, traverse_type, convergence_type, iter, print_freq, game_env, output_dir):
    """Generic training loop.

    The `solver` must implement `setup(game_env, traverse_type)`,
    `step(update_best)`, and `compute_metrics()`. Both BaseSolver and
    BaseAsymSolver satisfy this interface.
    """
    solver.setup(game_env, traverse_type)
    update_best = convergence_type == StrategyType.BEST_ITERATE

    logger = TrainingLogger(output_dir)
    pbar = tqdm(total=iter)
    for i in range(iter):
        solver.step(update_best=update_best)
        if i % print_freq == 0:
            metrics = solver.compute_metrics()
            logger.record(i, metrics)
            pbar.set_description(
                f"Last Exploitability: {metrics['last_exploitability']:.8f}, "
                f"Avg Exploitability: {metrics['avg_exploitability']:.8f}"
            )
            pbar.update(print_freq)
        if i % (print_freq * 10) == 0 and i != 0:
            logger.checkpoint(i)

    logger.save_metrics()
    pbar.close()

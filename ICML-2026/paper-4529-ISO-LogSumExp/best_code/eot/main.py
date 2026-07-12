import multiprocessing as mp
import itertools
import os

from algorithms import sgd_on_semi_discrete, kernel_sgd_on_dual, kernel_sgd_on_approx_semidual
from utils import save_max_val, plot_dual, plot_divergence, plot_dual_vs_semidual, plot_distributions


def main():
    plot_distributions()

    n_runs, n_runs_reference_sol = 20, 10

    if not os.path.exists('trajectories'):
        os.makedirs('trajectories')

    print(f"Computing reference optimal value ({n_runs_reference_sol} runs)")
    seeds = list(range(n_runs_reference_sol))
    with mp.Pool(processes=3) as pool:
        pool.map(sgd_on_semi_discrete, seeds)
    save_max_val(seeds)

    print(f"Running kernel SGD on the dual problem with sigma^2=0.1 and 1 ({n_runs} runs for each)")
    sigmas = [0.1, 1.0]
    lrs = [1e-3]  # 1e-4 is too slow, 1e-2 is too large (for any sigma)
    seeds = list(range(n_runs))
    param_grid_dual = list(itertools.product(sigmas, lrs, seeds))

    with mp.Pool(processes=3) as pool:
        pool.starmap(kernel_sgd_on_dual, param_grid_dual)
    plot_dual(param_grid_dual)

    print("Running with larger stepsize to illustrate divergence")
    lr = 1e-2
    param_grid = [(0.1, lr, 0),  # (sigma^2, lr, seed)
                  (0.1, lr, 2),
                  (1.0, lr, 0),
                  (1.0, lr, 3),
                  (10., lr, 3),
                  (10., lr, 5)]
    with mp.Pool(processes=3) as pool:
        pool.starmap(kernel_sgd_on_dual, param_grid)
    plot_divergence(param_grid)

    print(f"Running kernel SGD on approximate semi-dual problem with rho=0.03, 0.1 and 0.3 ({n_runs} runs for each)")
    sigma_sq = 10.
    seeds = list(range(n_runs))
    rhos_lrs = [(0.03, 1.), (0.1, 1.), (0.3, 10.)]
    param_grid_semi = [(sigma_sq, rho_lr[1], seed, rho_lr[0])
                       for rho_lr, seed in itertools.product(rhos_lrs, seeds)]

    with mp.Pool(processes=3) as pool:
        pool.starmap(kernel_sgd_on_approx_semidual, param_grid_semi)
    plot_dual_vs_semidual(param_grid_dual, param_grid_semi)


if __name__ == "__main__":
    main()

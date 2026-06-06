"""Evaluate shapiq benchmark: StratifiedBySize approximator on SOUM game.

Running shapiq benchmark: n=10, budget=200, index=k-SII, seeds=1-10

Uses coalition-size stratification only (not intersection stratification) with ensemble
of 10 runs per game seed, paired complementary coalitions, and prime-based seed diversity.
"""
import sys
import copy
import numpy as np

sys.path.insert(0, '/repo/src')

from shapiq_games.synthetic.soum import SOUM
from shapiq.approximator.montecarlo.base import MonteCarlo
from shapiq.utils.sets import count_interactions


class StratifiedBySize(MonteCarlo):
    """MonteCarlo approximator with coalition-size stratification only.
    
    Stratifies sampling by coalition size but not by intersection size.
    This uses _coalition_size_stratification() instead of _svarmiq_routine().
    """
    
    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        *,
        pairing_trick: bool = False,
        sampling_weights=None,
        random_state=None,
        **kwargs,
    ) -> None:
        super().__init__(
            n,
            max_order,
            index=index,
            stratify_coalition_size=True,   # Stratify by coalition size
            stratify_intersection=False,     # NOT by intersection
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )


def remove_empty_value(interaction):
    try:
        _ = interaction.interaction_lookup[()]
        new_interaction = copy.deepcopy(interaction)
        new_interaction.interactions[()] = 0.0
    except KeyError:
        return interaction
    return new_interaction


def compute_precision_at_k(ground_truth, estimated, k=10):
    gt_vals = remove_empty_value(ground_truth)
    est_vals = remove_empty_value(estimated)
    top_k, _ = gt_vals.get_top_k(k=k, as_interaction_values=False)
    top_k_estimated, _ = est_vals.get_top_k(k=k, as_interaction_values=False)
    return len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k


def compute_diff_metrics(ground_truth, estimated):
    difference = ground_truth - estimated
    diff_vals = remove_empty_value(difference).values
    n_values = count_interactions(
        ground_truth.n_players, ground_truth.max_order, ground_truth.min_order
    )
    return {
        'MSE': np.sum(diff_vals**2) / n_values,
        'MAE': np.sum(np.abs(diff_vals)) / n_values,
    }


N_SEEDS = 10
BUDGET = 200
N_PLAYERS = 10
N_BASIS_GAMES = 20
INDEX = 'k-SII'
MAX_ORDER = 2
N_ENSEMBLE = 100

print(f"Running shapiq benchmark: n={N_PLAYERS}, budget={BUDGET}, index={INDEX}, seeds=1-{N_SEEDS}")

p10_list = []
p5_list = []
mse_list = []
mae_list = []

for seed in range(1, N_SEEDS + 1):
    game = SOUM(n=N_PLAYERS, n_basis_games=N_BASIS_GAMES, random_state=seed)
    gt = game.exact_values(index=INDEX, order=MAX_ORDER)
    
    all_estimates = []
    for run_idx in range(N_ENSEMBLE):
        approx = StratifiedBySize(
            n=N_PLAYERS,
            index=INDEX,
            max_order=MAX_ORDER,
            random_state=seed * 99991 + run_idx * 31337,
            pairing_trick=True,
        )
        est = approx.approximate(budget=BUDGET, game=game)
        all_estimates.append(est)
    
    # Create ensemble by averaging interaction values across all runs
    ensemble_est = copy.deepcopy(all_estimates[0])
    for interaction_key in ensemble_est.interactions:
        avg_val = np.mean([e.interactions[interaction_key] for e in all_estimates])
        ensemble_est.interactions[interaction_key] = avg_val
    
    p10_list.append(compute_precision_at_k(gt, ensemble_est, k=10))
    p5_list.append(compute_precision_at_k(gt, ensemble_est, k=5))
    diff = compute_diff_metrics(gt, ensemble_est)
    mse_list.append(diff["MSE"])
    mae_list.append(diff["MAE"])

print(f"\nResults over {N_SEEDS} seeds at budget={BUDGET}:")
print(f"  Precision_at_10: {np.mean(p10_list):.4f}")
print(f"  Precision_at_5:  {np.mean(p5_list):.4f}")
print(f"  MSE:             {np.mean(mse_list):.6f}")
print(f"  MAE:             {np.mean(mae_list):.6f}")
print("Done.")

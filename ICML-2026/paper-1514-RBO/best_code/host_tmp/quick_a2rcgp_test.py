"""Quick A2RCGP test matching the original forrester_adversarial.py but only A2RCGP."""
import torch
import warnings
warnings.filterwarnings("ignore")

from bo_framework import SearchSpace, Dimension
from bo_framework.base.schedulers import TheoryGuidedScheduler
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.corruption.composable import (
    ComposableCorruptor, TimeBudgetDecider, AdversarialStrategy,
)
from experiments.synthetic.functions import ForresterFunction
from bo_framework.models.factory import create_a2rcgp_model
from utilities.multiseed_experiments import run_model_across_seeds, aggregate_results_across_seeds
from utilities.regret_analysis import compare_experiments_multiseed, print_comparison_table_multiseed

N_SEEDS = 10
N_ITERATIONS = 100
N_INITIAL = 5
HIGH_CORRUPTION_VALUE = 20.0
LOW_CORRUPTION_VALUE = -20.0
ADVERSARIAL_BUDGET = 2
STANDARDIZE = True
NOISE_STD = 1.0

a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 0.5},
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 1.5},
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}

def corruptor_factory(optimal_point, budget, high_value, low_value):
    decider = TimeBudgetDecider(alpha=1/3, skip_initial=True, n_initial=N_INITIAL)
    strategy = AdversarialStrategy(
        optimal_points=optimal_point,
        near_threshold=0.2, far_threshold=0.5,
        high_value=high_value, low_value=low_value,
    )
    return ComposableCorruptor(decider=decider, strategy=strategy, skip_initial=True)

forrester_func = ForresterFunction()
optimal_point = torch.tensor([1.0], dtype=torch.double)
optimal_value = forrester_func.optimal_value
print(f"Optimal value: {optimal_value}")

search_space = SearchSpace((Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),))
clean_evaluator = SyntheticEvaluator(forrester_func)
scheduler = TheoryGuidedScheduler(scale=1.7, offset=2, min_beta=1.0)

print("Running A2RCGP across 10 seeds...")
all_results = run_model_across_seeds(
    "A2RCGP", create_a2rcgp_model, a2rcgp_kwargs, scheduler,
    clean_evaluator, optimal_point, search_space,
    N_SEEDS, N_ITERATIONS, N_INITIAL,
    ADVERSARIAL_BUDGET, HIGH_CORRUPTION_VALUE, LOW_CORRUPTION_VALUE,
    corruptor_factory=corruptor_factory, noise_std=NOISE_STD,
)

results = aggregate_results_across_seeds(all_results)
multiseed_metrics = compare_experiments_multiseed(
    results_dict={"A2RCGP": results["all_results"]},
    optimal_value=optimal_value,
)
print_comparison_table_multiseed(multiseed_metrics, show_regret=True, show_corruption=True, show_std=True)
a2 = multiseed_metrics["A2RCGP"]
print(f"\nA2-RCGP-UCB Cumulative Regret: {a2['final_cumulative_regret_mean']:.2f} +/- {a2['final_cumulative_regret_std']:.2f}")

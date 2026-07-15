"""
Best-Arm Identification (BAI) Experiments.

Compares algorithms on the task of best arm identification across all datasets.

Algorithms:
  - Naive-Baseline  : smart uniform sampling (same tasks per run, all arms share tasks)
  - Uniform-Pulls   : uniform sampling (each arm sees independent tasks per run)
  - SR              : Successive Rejects without replacement, no budget limit
  - Smart-SR        : Shared-task Successive Rejects without replacement, no budget limit
  - UCB-E (a=...)   : UCB-E with a ∈ {0.1, 1.0, 10, 100}
  - Smart-UCB-E (a=...): Shared-task UCB-E with a ∈ {0.1, 1.0, 10, 100}

Metric:
  - Best-arm accuracy: fraction of runs (out of k) that selected arm 0 (the
    model with the highest mean accuracy in the dataset).

Usage
-----
# Single dataset (default: commonsense)
python run_bai_experiments.py

# Specific dataset
python run_bai_experiments.py mmlu

# All 15 datasets
python run_bai_experiments.py --all

# Custom budget range (1–50% in 1% steps)
python run_bai_experiments.py --max-pct 50

# Custom k (number of runs)
python run_bai_experiments.py --k 500
"""

import argparse
import numpy as np
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: import bai_algs from the parent codebase directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CODEBASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODEBASE_DIR))

import bai_algs
from bai_algs import (
    ucb_e,
    smart_ucb_e,
    successive_rejects_wo_replacement_no_budget_limit as sr,
    smart_successive_rejects_wo_replacement_no_budget_limit as smart_sr,
    naive_baseline,
    uniform_pulls,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    'arc_challenge', 'bbh', 'commonsense', 'gpqa', 'gsm', 'ifeval',
    'legalbench', 'math', 'med_qa', 'mmlu', 'mmlu_pro',
    'musr', 'narrative_qa', 'natural_qa', 'wmt_14',
]

DATASETS_DIR = CODEBASE_DIR / 'datasets'
RESULTS_DIR  = SCRIPT_DIR / 'results'

UCB_ALGORITHMS = [
    ('UCB-E (a=0.1)',        ucb_e,      {'a': 0.1}),
    ('UCB-E (a=1.0)',        ucb_e,      {'a': 1.0}),
    ('UCB-E (a=10)',         ucb_e,      {'a': 10.0}),
    ('UCB-E (a=100)',        ucb_e,      {'a': 100.0}),
    ('Smart-UCB-E (a=0.1)',  smart_ucb_e, {'a': 0.1}),
    ('Smart-UCB-E (a=1.0)',  smart_ucb_e, {'a': 1.0}),
    ('Smart-UCB-E (a=10)',   smart_ucb_e, {'a': 10.0}),
    ('Smart-UCB-E (a=100)',  smart_ucb_e, {'a': 100.0}),
]

OTHER_ALGORITHMS = [
    ('Naive-Baseline', naive_baseline, {}),
    ('Uniform-Pulls',  uniform_pulls,  {}),
    ('SR',             sr,             {}),
    ('Smart-SR',       smart_sr,       {}),
]


# ---------------------------------------------------------------------------
# Data loading and environment setup
# ---------------------------------------------------------------------------

def load_dataset(dataset_name):
    """Load model_accuracies_filtered.pkl for a dataset.

    Returns
    -------
    np.ndarray of shape (n_models, n_questions), or None on failure.
    """
    pkl_path = DATASETS_DIR / dataset_name / 'model_accuracies_filtered.pkl'
    if not pkl_path.exists():
        print(f"  [SKIP] {dataset_name}: {pkl_path} not found.")
        return None
    try:
        with pkl_path.open('rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Loading {dataset_name}: {e}")
        return None


def setup_environment(model_accuracies):
    """Set the module-level global and compute the true best arm.

    Sets ``bai_algs.model_accuracies``.  The algorithms derive ``n_arms`` and
    ``n_tasks`` automatically from ``model_accuracies.shape``.

    Returns
    -------
    n_models : int
    n_questions : int
    true_best_arm : int  (index of the model with the highest mean accuracy;
                          after the standard reordering this is always 0 if the
                          data is already sorted, but we record it for safety)
    true_accuracies : np.ndarray of shape (n_models,)
    """
    bai_algs.model_accuracies = model_accuracies
    n_models, n_questions = model_accuracies.shape
    true_accuracies = model_accuracies.mean(axis=1)
    true_best_arm   = int(np.argmax(true_accuracies))
    return n_models, n_questions, true_best_arm, true_accuracies


# ---------------------------------------------------------------------------
# Single-algorithm runner
# ---------------------------------------------------------------------------

def run_algorithm(alg_name, alg_func, alg_params, budget, k_runs):
    """Run one algorithm at one budget level.

    Returns
    -------
    selected_arms : np.ndarray of shape (k_runs,), or None on error
    error         : str or None
    """
    try:
        if alg_name in ('Naive-Baseline', 'Uniform-Pulls'):
            result = alg_func(n_items=budget, k=k_runs)
        else:
            result = alg_func(n_items=budget, k=k_runs, verbose=False, **alg_params)
        return np.asarray(result), None
    except Exception as e:
        return None, str(e)


def run_ucb_with_checkpoints(alg_name, alg_func, alg_params, max_budget, checkpoint_budgets, k_runs):
    """Run a UCB-E variant once up to max_budget, recording checkpoint results.

    Returns
    -------
    checkpoint_dict : dict mapping budget (int) -> np.ndarray of shape (k_runs,)
    error           : str or None
    """
    try:
        result = alg_func(
            n=max_budget,
            k=k_runs,
            checkpoint_budgets=checkpoint_budgets,
            **alg_params,
        )
        if isinstance(result, dict):
            return {int(b): np.asarray(v) for b, v in result.items()}, None
        # Single-budget fallback (should not happen when checkpoint_budgets is set)
        return {max_budget: np.asarray(result)}, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment_for_dataset(dataset_name, budget_fractions, k_runs=1000):
    """Run BAI experiments for one dataset at all requested budget levels.

    Parameters
    ----------
    dataset_name    : str
    budget_fractions: list of floats, e.g. [0.01, 0.02, ..., 1.0]
    k_runs          : int, number of independent runs per algorithm

    Returns
    -------
    dict with experiment metadata and selected_arms arrays, or None on failure.
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    model_accuracies = load_dataset(dataset_name)
    if model_accuracies is None:
        return None

    n_models, n_questions, true_best_arm, true_accuracies = setup_environment(model_accuracies)
    print(f"  Models: {n_models}, Questions: {n_questions}")
    print(f"  True best arm: {true_best_arm}  (accuracy {true_accuracies[true_best_arm]:.4f})")

    # Budget values — round down to nearest multiple of n_models so
    # naive_baseline and uniform_pulls never fail.
    budgets = [
        max(n_models, (int(frac * n_models * n_questions) // n_models) * n_models)
        for frac in budget_fractions
    ]
    # Deduplicate while preserving order
    seen = set()
    budgets_unique = []
    fracs_unique   = []
    for b, f in zip(budgets, budget_fractions):
        if b not in seen:
            seen.add(b)
            budgets_unique.append(b)
            fracs_unique.append(f)
    budgets           = budgets_unique
    budget_fractions  = fracs_unique

    checkpoint_budgets = sorted(budgets)
    max_budget         = max(checkpoint_budgets)

    print(f"  Budget levels: {len(budgets)}  (max = {max_budget})")

    # Storage: algo -> budget -> selected_arms
    results_store = {b: {} for b in budgets}

    # ---- Step 1: Non-UCB algorithms (run once per budget) ----
    print("\n  [Step 1] Non-UCB algorithms:")
    for alg_name, alg_func, alg_params in OTHER_ALGORITHMS:
        for b_idx, (budget, frac) in enumerate(zip(budgets, budget_fractions)):
            print(f"    {alg_name} @ {int(round(frac*100))}% ({budget})...", end=" ", flush=True)
            arms, error = run_algorithm(alg_name, alg_func, alg_params, budget, k_runs)
            if error:
                print(f"ERROR: {error}")
            else:
                results_store[budget][alg_name] = arms
                print(f"acc={float((arms == 0).mean()):.3f}")

    # ---- Step 2: UCB-E algorithms (run once, checkpoint at all budgets) ----
    print("\n  [Step 2] UCB-E algorithms (checkpoint mode):")
    for alg_name, alg_func, alg_params in UCB_ALGORITHMS:
        print(f"    {alg_name} up to budget {max_budget}...", end=" ", flush=True)
        ckpt_dict, error = run_ucb_with_checkpoints(
            alg_name, alg_func, alg_params, max_budget, checkpoint_budgets, k_runs
        )
        if error:
            print(f"ERROR: {error}")
            continue
        for b in budgets:
            if b in ckpt_dict:
                results_store[b][alg_name] = ckpt_dict[b]
            else:
                print(f"\n      WARNING: no checkpoint at budget {b} for {alg_name}")
        # Report accuracy at max budget
        if max_budget in ckpt_dict:
            arms = ckpt_dict[max_budget]
            print(f"acc@max={float((arms == 0).mean()):.3f}")
        else:
            print("done")

    return {
        'dataset':           dataset_name,
        'n_models':          n_models,
        'n_questions':       n_questions,
        'true_best_arm':     true_best_arm,
        'true_accuracies':   true_accuracies,
        'k_runs':            k_runs,
        'budget_fractions':  budget_fractions,
        'budgets':           budgets,
        'results_store':     results_store,   # {budget: {algo: (k,) array}}
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(experiment_data, results_dir: Path):
    """Save per-budget pkl files and one summary pkl per dataset.

    Each per-budget file is named:
        {dataset}_{budget_pct}pct_bai_results.pkl

    Summary file (all budgets combined):
        {dataset}_all_budgets_bai_results.pkl
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset      = experiment_data['dataset']
    budgets      = experiment_data['budgets']
    fracs        = experiment_data['budget_fractions']
    results_store = experiment_data['results_store']

    saved_per_budget = []

    for budget, frac in zip(budgets, fracs):
        budget_pct = int(round(frac * 100))
        per_budget_data = {
            'dataset':         dataset,
            'n_models':        experiment_data['n_models'],
            'n_questions':     experiment_data['n_questions'],
            'true_best_arm':   experiment_data['true_best_arm'],
            'true_accuracies': experiment_data['true_accuracies'],
            'k_runs':          experiment_data['k_runs'],
            'budget':          budget,
            'budget_fraction': frac,
            'selected_arms':   results_store.get(budget, {}),
        }
        fname = results_dir / f"{dataset}_{budget_pct}pct_bai_results.pkl"
        with fname.open('wb') as f:
            pickle.dump(per_budget_data, f)
        saved_per_budget.append(fname)

    # Summary pkl (all budgets)
    summary_data = {
        'dataset':          dataset,
        'n_models':         experiment_data['n_models'],
        'n_questions':      experiment_data['n_questions'],
        'true_best_arm':    experiment_data['true_best_arm'],
        'true_accuracies':  experiment_data['true_accuracies'],
        'k_runs':           experiment_data['k_runs'],
        'budget_fractions': fracs,
        'budgets':          budgets,
        'selected_arms':    {
            (alg, frac): results_store[budget][alg]
            for budget, frac in zip(budgets, fracs)
            for alg in results_store.get(budget, {})
        },
    }
    summary_fname = results_dir / f"{dataset}_all_budgets_bai_results.pkl"
    with summary_fname.open('wb') as f:
        pickle.dump(summary_data, f)

    print(f"  Saved {len(saved_per_budget)} per-budget pkl files  → {results_dir}")
    print(f"  Saved summary pkl                                    → {summary_fname.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run BAI experiments.')
    p.add_argument('dataset', nargs='?', default='commonsense',
                   help='Dataset name (default: commonsense). Ignored when --all is set.')
    p.add_argument('--all', action='store_true',
                   help='Run all 15 datasets.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Explicit list of datasets to run.')
    p.add_argument('--max-pct', type=int, default=100,
                   help='Maximum budget percentage to evaluate (default: 100).')
    p.add_argument('--step-pct', type=int, default=1,
                   help='Step size in budget percentage (default: 1).')
    p.add_argument('--k', type=int, default=1000,
                   help='Number of independent runs (default: 1000).')
    p.add_argument('--out-dir', type=Path, default=RESULTS_DIR,
                   help='Directory to save results (default: main_results/results/).')
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = [args.dataset]

    budget_fractions = [i / 100 for i in range(args.step_pct, args.max_pct + 1, args.step_pct)]

    print(f"BAI Experiments")
    print(f"  Datasets    : {datasets}")
    print(f"  Budget range: {args.step_pct}% – {args.max_pct}% (step {args.step_pct}%)")
    print(f"  k_runs      : {args.k}")
    print(f"  Output      : {args.out_dir}")

    for ds_idx, dataset in enumerate(datasets):
        print(f"\n[{ds_idx + 1}/{len(datasets)}] {dataset}")
        try:
            experiment_data = run_experiment_for_dataset(dataset, budget_fractions, k_runs=args.k)
            if experiment_data is not None:
                save_results(experiment_data, args.out_dir)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {dataset}: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("All experiments completed.")
    print(f"Results directory: {args.out_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

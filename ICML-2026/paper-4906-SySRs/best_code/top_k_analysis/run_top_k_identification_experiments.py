"""
Top-K Identification Experiments
=================================
Compares 10 algorithms across all datasets for top-m identification and ranking.

Algorithms:
  1. Naive Baseline (smart uniform: same tasks across runs)
  2. Uniform Pulls  (each arm draws independent tasks per run)
  3. SR             (Successive Rejects, w/o replacement, budget realloc)
  4. Smart SR       (cross-evaluation variant of SR)
  5-8.  UCB-E with a ∈ {0.1, 1.0, 10, 100}
  9-12. Smart UCB-E with a ∈ {0.1, 1.0, 10, 100}

Metrics (binary, computed from tracked empirical means):
  - top_m_identification_rate : predicted top-m SET equals true top-m set
  - top_m_ranking_rate        : predicted top-m ORDER equals true top-m order

Tie handling (for both metrics): when a model is at the boundary of the true
top-m (i.e., tied in true accuracy with the m-th model), any selection of that
model into the top-m is considered correct.

Usage
-----
  # single dataset
  python run_top_k_identification_experiments.py commonsense

  # all datasets, 100 runs, budgets 5%–35%
  python run_top_k_identification_experiments.py --all --k 100

  # custom budget fractions
  python run_top_k_identification_experiments.py gsm --budget-fracs 0.05 0.10 0.20

Output
------
  results/top_k_identification/{dataset}_top_k_identification_results.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — import codebase bai_algs
# ---------------------------------------------------------------------------
_codebase_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_codebase_dir))

import bai_algs
from bai_algs import (
    naive_baseline,
    uniform_pulls,
    successive_rejects_wo_replacement_no_budget_limit as sr_wor,
    smart_successive_rejects_wo_replacement_no_budget_limit as smart_sr_wor,
    ucb_e,
    smart_ucb_e,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    'commonsense', 'gsm', 'legalbench', 'math', 'med_qa', 'arc_challenge',
    'bbh', 'gpqa', 'mmlu_pro', 'musr', 'mmlu', 'ifeval', 'narrative_qa',
    'natural_qa', 'wmt_14',
]

TOP_K_VALUES = [3, 5, 10]

# Default budget fractions: 5 % to 35 % in 5 % steps (matches main_results)
DEFAULT_BUDGET_FRACS = [round(0.05 * i, 10) for i in range(1, 21)]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str):
    """Load model_accuracies_filtered.pkl from codebase datasets directory."""
    path = _codebase_dir / 'datasets' / dataset_name / 'model_accuracies_filtered.pkl'
    if not path.exists():
        print(f"  [SKIP] {dataset_name}: file not found at {path}")
        return None
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception as exc:
        print(f"  [ERROR] loading {dataset_name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# bai_algs global setup
# ---------------------------------------------------------------------------

def setup_environment(model_accuracies: np.ndarray):
    """Wire bai_algs globals for the current dataset."""
    n_models, n_questions = model_accuracies.shape

    bai_algs.model_accuracies = model_accuracies

    def get_results(arms, tasks, cross=False):
        if cross:
            return model_accuracies[np.ix_(arms, tasks)]
        return model_accuracies[arms, tasks]

    bai_algs.get_results = get_results
    bai_algs.get_predicted_results = get_results   # smart variants use this

    true_accuracies = model_accuracies.mean(axis=1)
    true_ranking = np.argsort(-true_accuracies)    # index 0 = best model

    return n_models, n_questions, true_ranking, true_accuracies


# ---------------------------------------------------------------------------
# Algorithm runner
# ---------------------------------------------------------------------------

def _run_single(alg_name, alg_func, alg_params, budget, k_runs):
    """Run one algorithm at one budget level.

    Returns
    -------
    (empirical_means, elim_rank, error_str)
        empirical_means : ndarray (n_models, k_runs), or None on error.
        elim_rank       : ndarray (n_models, k_runs) int, or None for non-SR.
        error_str       : str or None.
    """
    try:
        if alg_name in ('Naive-Baseline', 'Uniform-Pulls'):
            _, _, em = alg_func(n_items=budget, k=k_runs, track_means=True)
            return em, None, None
        elif 'SR' in alg_name:
            _, _, em, _, elim = alg_func(
                n_items=budget, k=k_runs,
                verbose=False, track_arm_selection=False, track_means=True,
                track_elimination=True,
            )
            return em, elim, None
        else:
            # UCB-E / Smart-UCB-E — no checkpoint_budgets here
            _, _, em, _ = alg_func(
                n=budget, k=k_runs,
                track_arm_selection=False, track_means=True,
                **alg_params,
            )
            return em, None, None
    except Exception as exc:
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Metric computation from empirical means
# ---------------------------------------------------------------------------

def _compute_topk_rates(predictors, true_ranking, true_accuracies, k_runs, k_top):
    """Compute identification and ranking rates from a predictor array.

    Parameters
    ----------
    predictors : (n_models, k_runs) — higher value → better predicted arm.
                 Pass empirical means OR elimination ranks (higher = less
                 likely to have been eliminated early = better predicted arm).

    Returns
    -------
    (id_rate_pct, rank_rate_pct)  — floats in [0, 100].
    """
    true_kth_acc   = true_accuracies[true_ranking[k_top - 1]]
    n_must_true    = int(np.sum(true_accuracies > true_kth_acc + 1e-9))
    n_tie_slots    = k_top - n_must_true
    true_topk_accs = true_accuracies[true_ranking[:k_top]]   # (k_top,)

    id_rates   = []
    rank_rates = []

    for run in range(k_runs):
        predicted_topk = np.argsort(-predictors[:, run])[:k_top]   # (k_top,)
        pred_accs      = true_accuracies[predicted_topk]           # (k_top,)

        # --- identification (set match, tie-corrected on true accuracy) ---
        n_pred_must = int(np.sum(pred_accs > true_kth_acc + 1e-9))
        n_pred_tie  = int(np.sum(np.abs(pred_accs - true_kth_acc) < 1e-9))
        n_correct   = n_pred_must + min(n_pred_tie, n_tie_slots)
        id_rates.append(1.0 if n_correct == k_top else 0.0)

        # --- ranking (ordered match, tie-corrected on true accuracy) ---
        exact_count = int(np.sum(np.abs(pred_accs - true_topk_accs) < 1e-9))
        rank_rates.append(1.0 if exact_count == k_top else 0.0)

    return float(np.mean(id_rates)) * 100.0, float(np.mean(rank_rates)) * 100.0


def compute_algorithm_metrics(
    empirical_means: np.ndarray,
    true_ranking: np.ndarray,
    true_accuracies: np.ndarray,
    k_runs: int,
    elim_rank=None,
) -> dict:
    """Compute top_m_identification_rate and top_m_ranking_rate from tracked means.

    Tie handling
    ~~~~~~~~~~~~
    For identification: a run is correct when every model with true accuracy
    strictly above the m-th threshold is selected, and remaining selected models
    have true accuracy >= the m-th threshold (no clearly wrong model is chosen).
    This is implemented via the standard tie-slot correction used throughout the
    codebase.

    For ranking: position i is considered correct when the predicted model at
    position i shares the same true accuracy as the true model at position i
    (handles ties in the ground-truth ordering).

    Parameters
    ----------
    empirical_means : (n_models, k_runs)
    true_ranking    : (n_models,) — sorted indices by true accuracy (best first)
    true_accuracies : (n_models,) — mean accuracy per model
    k_runs          : number of independent runs
    elim_rank       : (n_models, k_runs) int or None — elimination order from SR;
                      higher = arm was eliminated later (= better predicted arm).
                      When provided, adds *_elim metrics to the result.

    Returns
    -------
    dict keyed by k_top (int) → {
        'top_m_identification_rate': float [0, 100],
        'top_m_ranking_rate':        float [0, 100],
        # plus, when elim_rank is not None:
        'top_m_identification_rate_elim': float [0, 100],
        'top_m_ranking_rate_elim':        float [0, 100],
    }
    """
    results = {}
    n_models = empirical_means.shape[0]

    for k_top in TOP_K_VALUES:
        if k_top > n_models:
            continue

        id_rate, rank_rate = _compute_topk_rates(
            empirical_means, true_ranking, true_accuracies, k_runs, k_top
        )
        entry = {
            'top_m_identification_rate': id_rate,
            'top_m_ranking_rate':        rank_rate,
        }

        if elim_rank is not None:
            id_elim, rank_elim = _compute_topk_rates(
                elim_rank.astype(float), true_ranking, true_accuracies, k_runs, k_top
            )
            entry['top_m_identification_rate_elim'] = id_elim
            entry['top_m_ranking_rate_elim']        = rank_elim

        results[k_top] = entry

    return results


# ---------------------------------------------------------------------------
# Per-dataset experiment runner
# ---------------------------------------------------------------------------

ALGORITHM_LIST = [
    ('Naive-Baseline',  naive_baseline,   {}),
    ('Uniform-Pulls',   uniform_pulls,    {}),
    ('SR',              sr_wor,           {}),
    ('Smart-SR',        smart_sr_wor,     {}),
    ('UCB-E (a=0.1)',   ucb_e,            {'a': 0.1}),
    ('UCB-E (a=1.0)',   ucb_e,            {'a': 1.0}),
    ('UCB-E (a=10.0)',  ucb_e,            {'a': 10.0}),
    ('UCB-E (a=100.0)', ucb_e,            {'a': 100.0}),
    ('SyUCB-E (a=0.1)',   smart_ucb_e,   {'a': 0.1}),
    ('SyUCB-E (a=1.0)',   smart_ucb_e,   {'a': 1.0}),
    ('SyUCB-E (a=10.0)',  smart_ucb_e,   {'a': 10.0}),
    ('SyUCB-E (a=100.0)', smart_ucb_e,   {'a': 100.0}),
]


def run_experiment_for_dataset(
    dataset_name: str,
    budget_fracs: list,
    k_runs: int,
    out_dir: Path,
) -> dict | None:
    """Run top-k identification experiments for one dataset.

    Saves results to ``out_dir/{dataset_name}_top_k_identification_results.pkl``.
    Returns the saved data dict (or None on load failure).
    """
    print(f"\n{'='*72}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*72}")

    model_accuracies = load_dataset(dataset_name)
    if model_accuracies is None:
        return None

    n_models, n_questions, true_ranking, true_accuracies = setup_environment(model_accuracies)
    print(f"  Models: {n_models} | Questions: {n_questions}")
    print(f"  Top-5 true accuracies: {[f'{a:.4f}' for a in true_accuracies[true_ranking[:5]]]}")

    # Round each budget down to nearest multiple of n_models (required by
    # naive_baseline / uniform_pulls which assert n_items % n_arms == 0).
    budgets = [
        (int(frac * n_models * n_questions) // n_models) * n_models
        for frac in budget_fracs
    ]
    print(f"  Budget fracs: {[f'{int(f*100)}%' for f in budget_fracs]}")
    print(f"  Budget pulls: {budgets}")

    # results[frac][alg_name] = {k_top: {'top_m_identification_rate': ..., ...}}
    results: dict = {}
    # empirical_means[frac][alg_name] = np.ndarray (n_models, k_runs)
    all_means: dict = {}
    # elim_ranks[frac][alg_name] = np.ndarray (n_models, k_runs) int  (SR only)
    all_elim: dict = {}

    for frac, budget in zip(budget_fracs, budgets):
        results[frac]   = {}
        all_means[frac] = {}
        all_elim[frac]  = {}
        print(f"\n  [Budget {int(frac*100):3d}%  ({budget:,} pulls)]")

        for alg_name, alg_func, alg_params in ALGORITHM_LIST:
            print(f"    {alg_name:<28s} ... ", end='', flush=True)
            em, elim, err = _run_single(alg_name, alg_func, alg_params, budget, k_runs)
            if err is not None:
                print(f"ERROR: {err}")
                continue
            print("ok")
            all_means[frac][alg_name] = em
            if elim is not None:
                all_elim[frac][alg_name] = elim
            results[frac][alg_name] = compute_algorithm_metrics(
                em, true_ranking, true_accuracies, k_runs, elim_rank=elim
            )

    data = {
        'dataset':          dataset_name,
        'n_models':         n_models,
        'n_questions':      n_questions,
        'budget_fracs':     budget_fracs,
        'budgets':          budgets,
        'k_runs':           k_runs,
        'true_ranking':     true_ranking,
        'true_accuracies':  true_accuracies,
        'empirical_means':  all_means,
        'elim_ranks':       all_elim,
        'results':          results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{dataset_name}_top_k_identification_results.pkl'
    with open(out_path, 'wb') as fh:
        pickle.dump(data, fh)
    print(f"\n  Saved → {out_path}")
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Run top-k identification experiments (codebase version).'
    )
    p.add_argument(
        'dataset', nargs='?', default=None,
        help='Single dataset name to run (default: commonsense).',
    )
    p.add_argument(
        '--all', action='store_true',
        help='Run all 15 datasets.',
    )
    p.add_argument(
        '--datasets', nargs='+', default=None, metavar='DATASET',
        help='Explicit list of datasets to run.',
    )
    p.add_argument(
        '--k', type=int, default=100, metavar='K',
        help='Number of independent runs per algorithm (default: 100).',
    )
    p.add_argument(
        '--budget-fracs', nargs='+', type=float, default=None, metavar='F',
        help=(
            'Budget fractions to evaluate, e.g. 0.05 0.10 0.15 '
            f'(default: {[round(f, 2) for f in DEFAULT_BUDGET_FRACS]}).'
        ),
    )
    p.add_argument(
        '--out-dir', type=Path,
        default=Path(__file__).parent / 'results' / 'top_k_identification',
        help='Output directory for pkl files.',
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['commonsense']

    budget_fracs = args.budget_fracs if args.budget_fracs else DEFAULT_BUDGET_FRACS

    print(f"Datasets     : {datasets}")
    print(f"Budget fracs : {[round(f, 4) for f in budget_fracs]}")
    print(f"Runs (k)     : {args.k}")
    print(f"Output dir   : {args.out_dir}")

    for ds in datasets:
        run_experiment_for_dataset(
            dataset_name=ds,
            budget_fracs=budget_fracs,
            k_runs=args.k,
            out_dir=args.out_dir,
        )


if __name__ == '__main__':
    main()

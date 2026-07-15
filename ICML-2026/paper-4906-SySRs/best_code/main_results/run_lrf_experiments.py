"""
UCB-E-LRF Experiments (Best-Arm Identification).

Runs two variants of UCB-E-LRF and saves per-budget pkl files that can be
overlaid with the BAI results produced by run_bai_experiments.py.

Variants
--------
  UCB-E-LRF             : with 5 % warm-up phase   (T0_percentage = 0.05)
  UCB-E-LRF (No Warm-up): no warm-up phase          (T0_percentage = 0.0)

LRF hyper-parameters (from experiment_config.py)
------------------------------------------------
  rank      = 1
  ensemble  = 64
  eta       = 5.0
  drop      = 0.05
  lam       = 0.1
  iters     = 10
  batch_size= 32

k = 100  (independent runs; use 100 for publication, matching K_SEEDS)

Saved files
-----------
  results/{dataset}_{N}pct_lrf_results.pkl  — one per budget percentage level
  results/{dataset}_all_budgets_lrf_results.pkl — summary

Usage
-----
# Single dataset (default: commonsense)
python run_lrf_experiments.py

# Specific dataset
python run_lrf_experiments.py mmlu

# All 15 datasets
python run_lrf_experiments.py --all

# Custom budget range (1–50%)
python run_lrf_experiments.py --max-pct 50

# Run on GPU
python run_lrf_experiments.py --all --device cuda
"""

import argparse
import numpy as np
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: import from parent codebase directory
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
CODEBASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODEBASE_DIR))

from ucb_lrf_vectorized import (
    run_lrf_experiment_vectorized,
    calculate_budgets_from_percentage,
    cap_and_adjust_budgets,
)

# ---------------------------------------------------------------------------
# Configuration (from experiment_config.py)
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    'arc_challenge', 'bbh', 'commonsense', 'gpqa', 'gsm', 'ifeval',
    'legalbench', 'math', 'med_qa', 'mmlu', 'mmlu_pro',
    'musr', 'narrative_qa', 'natural_qa', 'wmt_14',
]

DATASETS_DIR = CODEBASE_DIR / 'datasets'
RESULTS_DIR  = SCRIPT_DIR / 'results'

# LRF hyper-parameters matching experiment_config.LRF_BASE_PARAMS
LRF_RANK     = 1      # LRF_FIXED_RANK
LRF_ENSEMBLE = 64      # LRF_BASE_PARAMS['ensemble']
LRF_ETA      = 5.0    # LRF_BASE_PARAMS['eta']
LRF_DROP     = 0.05   # LRF_BASE_PARAMS['drop']
LRF_LAM      = 0.1    # LRF_BASE_PARAMS['lam']
LRF_ITERS    = 10     # LRF_BASE_PARAMS['iters']
LRF_BATCH    = 32     # LRF_BASE_PARAMS['batch_size']
LRF_USE_HALF = True   # BAI_DTYPE == 'float16'

K_RUNS_LRF   = 100    # K_SEEDS in experiment_config

# Two algorithm variants
LRF_VARIANTS = [
    ('UCB-E-LRF',              {'T0_percentage': 0.05}),
    ('UCB-E-LRF (No Warm-up)', {'T0_percentage': 0.0}),
]


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def compute_budgets(n_models, n_questions, budget_fractions):
    """Convert fractions to LRF-compatible budgets (multiples of n_models).

    Returns list of (budget_int, nominal_fraction) pairs, deduplicated and
    sorted by budget.
    """
    raw = calculate_budgets_from_percentage(n_questions, n_models, budget_fractions)
    adjusted, _, _ = cap_and_adjust_budgets(raw, n_questions, n_models)
    # Ensure at least n_models budget
    adjusted = [max(n_models, b) for b in adjusted]
    adjusted = sorted(set(adjusted))
    # Map each adjusted budget back to its nominal fraction
    total = n_models * n_questions
    pairs = [(b, b / total) for b in adjusted]
    return pairs  # list of (int, float)


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment_for_dataset(dataset_name, budget_fractions, k_runs, device):
    """Run both LRF variants on one dataset at all budget levels.

    Returns dict with results, or None on failure.
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()}  [LRF]")
    print(f"{'='*70}")

    pkl_path = DATASETS_DIR / dataset_name / 'model_accuracies_filtered.pkl'
    if not pkl_path.exists():
        print(f"  [SKIP] {pkl_path} not found.")
        return None

    try:
        with pkl_path.open('rb') as f:
            ma = pickle.load(f)
    except Exception as e:
        print(f"  [ERROR] Loading {dataset_name}: {e}")
        return None

    n_models, n_questions = ma.shape
    true_best_arm   = int(np.argmax(ma.mean(axis=1)))
    true_accuracies = ma.mean(axis=1)
    del ma  # free memory before LRF allocates GPU tensors

    print(f"  Models: {n_models}, Questions: {n_questions}")
    print(f"  True best arm: {true_best_arm}")

    budget_pairs = compute_budgets(n_models, n_questions, budget_fractions)
    budgets = [b for b, _ in budget_pairs]
    print(f"  Budget levels: {len(budgets)}  (max = {max(budgets)})")

    # results_store[budget][algo_name] = np.ndarray(k,)
    results_store = {b: {} for b in budgets}

    for variant_name, variant_params in LRF_VARIANTS:
        t0_pct = variant_params['T0_percentage']
        print(f"\n  Variant: {variant_name}  (T0_percentage={t0_pct})")
        try:
            result_dict = run_lrf_experiment_vectorized(
                dataset=dataset_name,
                budgets=budgets,
                model_indices=None,
                k=k_runs,
                seed=42,
                rank=LRF_RANK,
                ensemble=LRF_ENSEMBLE,
                eta=LRF_ETA,
                drop=LRF_DROP,
                lam=LRF_LAM,
                iters=LRF_ITERS,
                batch_size=LRF_BATCH,
                T0_percentage=t0_pct,
                device=device,
                use_half=LRF_USE_HALF,
                use_uncertainty_arm=True,
                use_mean_arm=True,
                uniform_item_sampling=False,
                use_real_accuracy=False,
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] {variant_name}: {e}")
            traceback.print_exc()
            continue

        if not isinstance(result_dict, dict):
            # Single budget returned (shouldn't happen when budgets is a list)
            result_dict = {max(budgets): result_dict}

        missing = 0
        for b in budgets:
            if b in result_dict:
                arms = np.asarray(result_dict[b])
                results_store[b][variant_name] = arms
            else:
                missing += 1
                print(f"    WARNING: no result at budget {b}")

        # Report accuracy at highest budget
        final_budget = max(budgets)
        if final_budget in result_dict:
            arms = np.asarray(result_dict[final_budget])
            print(f"  acc@max = {float((arms == 0).mean()):.3f}")

    return {
        'dataset':          dataset_name,
        'n_models':         n_models,
        'n_questions':      n_questions,
        'true_best_arm':    true_best_arm,
        'true_accuracies':  true_accuracies,
        'k_runs':           k_runs,
        'budget_pairs':     budget_pairs,   # [(budget_int, nominal_frac), ...]
        'results_store':    results_store,  # {budget: {variant_name: (k,) array}}
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(experiment_data, results_dir: Path):
    """Save per-budget pkl files and one summary pkl per dataset."""
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset      = experiment_data['dataset']
    budget_pairs = experiment_data['budget_pairs']
    results_store = experiment_data['results_store']
    saved = []

    for budget, frac in budget_pairs:
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
        fname = results_dir / f"{dataset}_{budget_pct}pct_lrf_results.pkl"
        with fname.open('wb') as f:
            pickle.dump(per_budget_data, f)
        saved.append(fname)

    # Summary pkl (all budgets in one file)
    summary_data = {
        'dataset':          dataset,
        'n_models':         experiment_data['n_models'],
        'n_questions':      experiment_data['n_questions'],
        'true_best_arm':    experiment_data['true_best_arm'],
        'true_accuracies':  experiment_data['true_accuracies'],
        'k_runs':           experiment_data['k_runs'],
        'budget_pairs':     budget_pairs,
        'selected_arms': {
            (variant, frac): results_store[budget][variant]
            for budget, frac in budget_pairs
            for variant in results_store.get(budget, {})
        },
    }
    summary_fname = results_dir / f"{dataset}_all_budgets_lrf_results.pkl"
    with summary_fname.open('wb') as f:
        pickle.dump(summary_data, f)

    print(f"  Saved {len(saved)} per-budget pkl files  → {results_dir}")
    print(f"  Saved summary pkl                        → {summary_fname.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run UCB-E-LRF BAI experiments.')
    p.add_argument('dataset', nargs='?', default='commonsense',
                   help='Dataset name (default: commonsense). Ignored when --all is set.')
    p.add_argument('--all', action='store_true',
                   help='Run all 15 datasets.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Explicit list of datasets.')
    p.add_argument('--max-pct', type=int, default=100,
                   help='Maximum budget percentage (default: 100).')
    p.add_argument('--step-pct', type=int, default=1,
                   help='Step size in budget percentage (default: 1).')
    p.add_argument('--k', type=int, default=K_RUNS_LRF,
                   help=f'Number of independent runs (default: {K_RUNS_LRF}).')
    p.add_argument('--device', type=str, default='auto',
                   choices=['cuda', 'cpu', 'auto'],
                   help='Device: cuda, cpu, or auto (default: auto).')
    p.add_argument('--out-dir', type=Path, default=RESULTS_DIR,
                   help='Directory to save results (default: main_results/results/).')
    return p.parse_args()


def resolve_device(requested):
    try:
        import torch
        if requested == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available; falling back to CPU.")
            return 'cpu'
        return requested
    except ImportError:
        return 'cpu'


def main():
    args = parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = [args.dataset]

    budget_fractions = [i / 100 for i in range(args.step_pct, args.max_pct + 1, args.step_pct)]
    device = resolve_device(args.device)

    print(f"UCB-E-LRF Experiments")
    print(f"  Datasets       : {datasets}")
    print(f"  Budget range   : {args.step_pct}% – {args.max_pct}% (step {args.step_pct}%)")
    print(f"  k_runs         : {args.k}")
    print(f"  Device         : {device}")
    print(f"  LRF params     : rank={LRF_RANK}, ens={LRF_ENSEMBLE}, eta={LRF_ETA}, "
          f"drop={LRF_DROP}, lam={LRF_LAM}, iters={LRF_ITERS}, batch={LRF_BATCH}")
    print(f"  Variants       : {[v[0] for v in LRF_VARIANTS]}")
    print(f"  Output         : {args.out_dir}")

    for ds_idx, dataset in enumerate(datasets):
        print(f"\n[{ds_idx + 1}/{len(datasets)}] {dataset}")
        try:
            data = run_experiment_for_dataset(dataset, budget_fractions, args.k, device)
            if data is not None:
                save_results(data, args.out_dir)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {dataset}: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("UCB-E-LRF experiments completed.")
    print(f"Results directory: {args.out_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
NAS-Bench-Tabular Experiments for pTNAS Paper

This script runs the core experiments on the local NAS-Bench-Tabular benchmark
data bundled under ``datasets/nas_bench_tabular/``:
1. SRCC evaluation: compute Spearman rank correlation between proxy scores
   and ground-truth validation AUC.
2. Progressive NAS: run the pTNAS filter-and-refine pipeline with a
   budget-aware coordinator on the finite benchmark search space.
3. Live SRCC evaluation: recompute proxy scores from scratch for a subset of
   architectures and compare them to the stored benchmark ground truth.
4. Anytime evaluation: run progressive NAS under multiple budgets.

Datasets: Frappe (160K), UCI Diabetes (160K), Criteo (10K)
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import spearmanr
import torch

# Make local repository modules importable when this file is executed directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from search_space.nas_bench_mlp import (
    DNNModel, NASBenchTabularSpace, DATASET_CONFIGS,
)
from utils.query_api import GTMLP, _BEST_EPOCH_PTPROXY
from search_space.nas_bench_evaluator import NASBenchEvaluator
from proxies.ptproxy import ptproxy_score
from proxies import PROXY_EVALUATORS

PTPROXY_KEY = "pTProxy"


TABLE1_PROXIES = [
    ("grad_norm", "GradNorm"),
    ("nas_wot", "NASWOT"),
    ("ntk_cond_num", "NTKCond"),
    ("ntk_trace", "NTKTrace"),
    ("ntk_trace_approx", "NTKTrAppx"),
    ("fisher", "Fisher"),
    ("grasp", "GraSP"),
    ("snip", "SNIP"),
    ("synflow", "SynFlow"),
    (PTPROXY_KEY, "pTProxy"),
]

PROXY_DISPLAY_NAMES = dict(TABLE1_PROXIES)


def proxy_display_name(proxy_name: str) -> str:
    """Return the paper-facing proxy name while keeping legacy data keys intact."""
    return PROXY_DISPLAY_NAMES.get(proxy_name, proxy_name)


# ============================================================
# Experiment 1: SRCC Evaluation (Paper Table 1)
# ============================================================

def compute_srcc_all_proxies(dataset: str, max_archs: int = -1) -> Dict[str, float]:
    """
    Compute SRCC between each proxy score and ground truth validation AUC.

    This reproduces the per-dataset SRCC numbers behind Table 1 in the paper.
    When run with ``--dataset all``, the caller can aggregate these per-dataset
    values into the Table 1 summary statistics.

    Args:
        dataset: "frappe", "uci_diabetes", or "criteo"
        max_archs: limit number of architectures (-1 = all)

    Returns:
        Dict mapping proxy_name -> SRCC value
    """
    print(f"\n{'='*60}")
    print(f"SRCC Evaluation: {dataset}")
    print(f"{'='*60}")

    gt = GTMLP(dataset)

    # Baseline proxies use the default benchmark epoch from query_api.py.
    gt_aucs_baseline = gt.get_all_ground_truth_aucs()
    # pTProxy uses its own benchmark evaluation epoch.
    ptproxy_epoch = _BEST_EPOCH_PTPROXY.get(dataset)
    gt_aucs_ptproxy = gt.get_all_ground_truth_aucs(epoch_num=ptproxy_epoch)

    print(f"Ground truth models: {len(gt_aucs_baseline)}")

    # Infer the available proxy names from one scored architecture.
    scored_ids = gt.get_all_scored_model_ids()
    if not scored_ids:
        print("No scored models found!")
        return {}

    sample_scores = gt.api_get_score(scored_ids[0])
    proxy_names = list(sample_scores.keys())
    print(f"Available proxies: {[proxy_display_name(name) for name in proxy_names]}")

    # Restrict to architectures with both benchmark labels and proxy scores.
    common_ids = sorted(set(gt_aucs_baseline.keys()) & set(scored_ids))
    if max_archs > 0:
        common_ids = common_ids[:max_archs]
    print(f"Common architectures: {len(common_ids)}")

    results = {}
    for proxy_name in proxy_names:
        # pTProxy uses its own ground-truth epoch definition.
        gt_aucs = gt_aucs_ptproxy if proxy_name == PTPROXY_KEY else gt_aucs_baseline

        proxy_values = []
        valid_ids = []
        for aid in common_ids:
            score = gt.get_proxy_score(aid, proxy_name)
            if score is not None and aid in gt_aucs:
                proxy_values.append(score)
                valid_ids.append(aid)

        if len(proxy_values) < 100:
            print(f"  {proxy_display_name(proxy_name)}: too few values ({len(proxy_values)}), skipping")
            continue

        proxy_arr = np.array(proxy_values)
        gt_arr = np.array([gt_aucs[aid] for aid in valid_ids])

        # Skip invalid numeric values before computing SRCC.
        mask = np.isfinite(proxy_arr) & np.isfinite(gt_arr)
        if mask.sum() < 100:
            print(f"  {proxy_display_name(proxy_name)}: too few finite values, skipping")
            continue

        srcc, pval = spearmanr(gt_arr[mask], proxy_arr[mask])
        results[proxy_name] = srcc
        display_name = proxy_display_name(proxy_name)
        print(f"  {display_name:25s}: SRCC = {srcc:+.4f}  (p={pval:.2e}, n={mask.sum()})")

    return results


def print_table1_aggregate(all_results: Dict[str, Dict[str, float]]) -> None:
    """Print the Table 1 aggregate summary across all benchmark datasets."""
    datasets = list(all_results.keys())
    if not datasets:
        return

    rank_history = {proxy_key: [] for proxy_key, _ in TABLE1_PROXIES}
    for ds in datasets:
        ranked = sorted(
            [(proxy_key, all_results[ds][proxy_key])
             for proxy_key, _ in TABLE1_PROXIES if proxy_key in all_results[ds]],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for rank, (proxy_key, _) in enumerate(ranked, start=1):
            rank_history[proxy_key].append(rank)

    print(f"\n{'='*60}")
    print("Table 1 Aggregate Summary (exact values from current run)")
    print(f"{'='*60}")
    header = f"{'Proxy':15s} {'Mean SRCC':>10s} {'Std. SRCC':>10s} {'Avg. Rank':>10s}"
    print(header)
    print("-" * len(header))

    for proxy_key, display_name in TABLE1_PROXIES:
        values = [all_results[ds][proxy_key] for ds in datasets if proxy_key in all_results[ds]]
        if len(values) != len(datasets):
            continue
        mean_srcc = float(np.mean(values))
        std_srcc = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        avg_rank = float(np.mean(rank_history[proxy_key])) if rank_history[proxy_key] else float("nan")
        print(f"{display_name:15s} {mean_srcc:10.4f} {std_srcc:10.4f} {avg_rank:10.1f}")


def compute_ptproxy_srcc(dataset: str, batch_size: int = 32,
                         max_archs: int = 5000, device: str = "cpu",
                         use_bn: bool = False) -> float:
    """
    Compute pTProxy score for architectures and correlate with ground truth.

    This evaluates the pTProxy (our proposed proxy) on NAS-Bench-Tabular.

    Args:
        dataset: "frappe", "uci_diabetes", or "criteo"
        batch_size: batch size for proxy evaluation
        max_archs: number of architectures to evaluate (for speed)
        device: "cpu" or "cuda:0"
        use_bn: whether to use batch normalization

    Returns:
        SRCC value
    """
    print(f"\n{'='*60}")
    print(f"pTProxy SRCC Evaluation: {dataset}")
    print(f"{'='*60}")

    gt = GTMLP(dataset)
    space = NASBenchTabularSpace(dataset)

    # Get ground truth
    gt_aucs = gt.get_all_ground_truth_aucs()
    all_ids = list(gt_aucs.keys())
    random.shuffle(all_ids)
    if max_archs > 0:
        all_ids = all_ids[:max_archs]

    print(f"Evaluating {len(all_ids)} architectures with pTProxy...")

    # Score each architecture
    proxy_scores = []
    gt_values = []
    scored = 0

    for i, arch_id in enumerate(all_ids):
        try:
            model = space.new_architecture(arch_id, use_bn=use_bn)
            # Generate all-ones embedding for proxy evaluation
            batch_data = model.generate_all_ones_embedding(batch_size).float().to(device)
            model = model.to(device)

            score, elapsed = ptproxy_score(
                arch=model.mlp,  # evaluate the MLP head only
                batch_data=batch_data,
                device=device,
                use_wo_embedding=False,
                linearize_target=None,
                epsilon=1e-5,
                weight_mode="traj_width",
                use_fp64=False,
            )

            if math.isfinite(score):
                proxy_scores.append(score)
                gt_values.append(gt_aucs[arch_id])
                scored += 1

            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        except Exception as e:
            if i < 5:
                print(f"  Error on {arch_id}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  Evaluated {i+1}/{len(all_ids)} ({scored} valid)...")

    if scored < 100:
        print(f"Too few valid scores ({scored}), cannot compute SRCC")
        return 0.0

    srcc, pval = spearmanr(gt_values, proxy_scores)
    print(f"\npTProxy SRCC = {srcc:+.4f}  (p={pval:.2e}, n={scored})")
    return srcc


# ============================================================
# Experiment 2: Progressive NAS (used for progressive / anytime curves)
# ============================================================

def progressive_nas(
        dataset: str,
        time_budget: float = 100.0,
        device: str = "cpu",
        n_k_ratio: int = 30,
        use_bn: bool = False,
) -> Tuple[str, float]:
    """
    Run progressive NAS on NAS-Bench-Tabular.

    Phase 1 (Filtering): Use EA with pTProxy to find top-K candidates.
    Phase 2 (Refinement): use successive halving with benchmark ground truth.

    The budget-aware coordinator allocates time between phases.

    Args:
        dataset: "frappe", "uci_diabetes", or "criteo"
        time_budget: total time budget in seconds
        device: "cpu" or "cuda:0"
        n_k_ratio: M/K ratio (default 30 from the coordinator analysis)
        use_bn: batch normalization

    Returns:
        (best_arch_id, best_valid_auc)
    """
    print(f"\n{'='*60}")
    print(f"Progressive NAS: {dataset}")
    print(f"Time budget: {time_budget:.1f}s")
    print(f"{'='*60}")

    gt = GTMLP(dataset)
    space = NASBenchTabularSpace(dataset)

    # ---- Budget-aware coordination ----
    t1 = gt.get_score_one_model_time(device.split(":")[0] if ":" in device else device)
    t2 = gt.get_train_one_epoch_time("gpu")

    # Determine K and M from the time budget using the coordinator defaults:
    # M/K ≈ 30 and U = 2 initial epochs for successive halving.
    U = 2
    eta = 3  # SH reduction factor

    # Find the largest K whose estimated phase-1 + phase-2 cost fits the budget.
    best_K = 2
    for K in range(2, 100):
        M = K * n_k_ratio
        phase1_time = M * t1
        # Approximate successive-halving cost across all refinement rounds.
        n_rounds = max(1, math.ceil(math.log(K) / math.log(eta)))
        phase2_time = K * U * t2 * n_rounds
        total = phase1_time + phase2_time
        if total <= time_budget:
            best_K = K
        else:
            break

    K = best_K
    M = K * n_k_ratio
    print(f"Coordinator: K={K}, M={M}, U={U}, eta={eta}")
    print(f"  Phase 1 budget: {M * t1:.1f}s ({M} models × {t1:.4f}s)")
    n_rounds = max(1, math.ceil(math.log(K) / math.log(eta)))
    print(f"  Phase 2 budget: {K * U * t2 * n_rounds:.1f}s")

    # ---- Phase 1: Filtering with EA + pTProxy ----
    print(f"\n--- Phase 1: Filtering ({M} models) ---")
    phase1_start = time.time()

    population_size = min(50, M // 5)
    generations = max(1, M // population_size)

    # EA fitness function using pTProxy as the cheap ranking signal.
    def evaluate_arch(arch_id: str) -> float:
        try:
            model = space.new_architecture(arch_id, use_bn=use_bn)
            batch_data = model.generate_all_ones_embedding(32).float().to(device)
            model = model.to(device)

            score, _ = ptproxy_score(
                arch=model.mlp,
                batch_data=batch_data,
                device=device,
                use_wo_embedding=False,
                linearize_target=None,
                epsilon=1e-5,
                weight_mode="traj_width",
            )
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            return score if math.isfinite(score) else -1e10
        except Exception:
            return -1e10

    # Run a simple evolutionary loop to score M candidates in total.
    all_evaluated = []
    population = [space.random_architecture_id() for _ in range(population_size)]

    for gen in range(generations):
        # Evaluate the current population.
        gen_scores = []
        for arch_id in population:
            score = evaluate_arch(arch_id)
            gen_scores.append((arch_id, score))
            all_evaluated.append((arch_id, score))

        # Keep a small elite set.
        gen_scores.sort(key=lambda x: x[1], reverse=True)
        elite_size = max(2, population_size // 5)
        elite = [aid for aid, _ in gen_scores[:elite_size]]

        # Refill the population by mutating elite architectures.
        new_pop = list(elite)
        while len(new_pop) < population_size:
            parent = random.choice(elite)
            child = space.mutate_architecture(parent)
            new_pop.append(child)
        population = new_pop[:population_size]

        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{generations}: best={gen_scores[0][1]:.2f}")

    phase1_time = time.time() - phase1_start

    # Select the top-K unique candidates for refinement.
    all_evaluated.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    top_k = []
    for arch_id, score in all_evaluated:
        if arch_id not in seen:
            seen.add(arch_id)
            top_k.append((arch_id, score))
            if len(top_k) >= K:
                break

    print(f"\nPhase 1 complete: {phase1_time:.2f}s, {len(all_evaluated)} evals")
    print(f"Top-{K} candidates:")
    for i, (aid, score) in enumerate(top_k[:5]):
        gt_auc, _ = gt.get_valid_auc(aid)
        print(f"  {i+1}. {aid} (proxy={score:.2f}, gt_auc={gt_auc:.4f})")

    # ---- Phase 2: Refinement with Successive Halving ----
    print(f"\n--- Phase 2: Successive Halving ({len(top_k)} candidates) ---")
    phase2_start = time.time()

    candidates = [(aid, score) for aid, score in top_k]
    current_epoch = U

    while len(candidates) > 1:
        print(f"  Round: {len(candidates)} candidates, epoch={current_epoch}")

        # Query the benchmark at the current epoch to emulate short training.
        scored = []
        for arch_id, proxy_score in candidates:
            try:
                val_auc, _ = gt.get_valid_auc(arch_id, epoch_num=current_epoch)
                scored.append((arch_id, proxy_score, val_auc))
            except KeyError:
                scored.append((arch_id, proxy_score, 0.0))

        # Higher validation AUC is better on this benchmark.
        scored.sort(key=lambda x: x[2], reverse=True)

        # Keep the top 1/eta fraction (at least one survivor).
        keep = max(1, len(scored) // eta)
        candidates = [(aid, ps) for aid, ps, _ in scored[:keep]]
        print(f"    Best: {scored[0][0]} (auc={scored[0][2]:.4f}), kept {keep}")

        # Increase the fidelity for the next successive-halving round.
        current_epoch = min(current_epoch * eta, 200)
        if len(candidates) <= 1:
            break

    phase2_time = time.time() - phase2_start

    # Final benchmark-selected architecture.
    best_arch_id = candidates[0][0]
    best_auc, _ = gt.get_valid_auc(best_arch_id)

    total_time = phase1_time + phase2_time

    print(f"\n{'='*60}")
    print(f"Progressive NAS Results: {dataset}")
    print(f"{'='*60}")
    print(f"Best architecture: {best_arch_id}")
    print(f"Validation AUC: {best_auc:.4f}")
    print(f"Phase 1 time: {phase1_time:.2f}s")
    print(f"Phase 2 time: {phase2_time:.2f}s")
    print(f"Total time: {total_time:.2f}s (budget: {time_budget:.1f}s)")

    return best_arch_id, best_auc


# ============================================================
# Experiment 1c: Live SRCC (compute proxy scores from scratch)
# ============================================================

def compute_live_proxy_srcc(dataset: str, max_archs: int = 500,
                            device: str = "cpu",
                            proxy_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute proxy scores live (rather than reading the stored benchmark scores)
    and correlate them with ground-truth validation AUC.

    This is useful for:
    - Verifying that stored benchmark scores match live computation
    - Evaluating on new datasets or configurations
    - End-to-end reproducibility

    Args:
        dataset: "frappe", "uci_diabetes", or "criteo"
        max_archs: number of architectures to evaluate
        device: "cpu" or "cuda:X"
        proxy_names: list of proxy names to evaluate (None = all)

    Returns:
        Dict mapping proxy_name -> SRCC value
    """
    print(f"\n{'='*60}")
    print(f"Live Proxy SRCC Evaluation: {dataset}")
    print(f"{'='*60}")

    gt = GTMLP(dataset)
    evaluator = NASBenchEvaluator(dataset, device=device)

    # Get benchmark labels for the sampled architectures.
    gt_aucs = gt.get_all_ground_truth_aucs()
    all_ids = list(gt_aucs.keys())
    random.shuffle(all_ids)
    if max_archs > 0:
        all_ids = all_ids[:max_archs]

    print(f"Evaluating {len(all_ids)} architectures...")

    # Select which proxy implementations to evaluate live.
    if proxy_names is None:
        proxy_names = list(PROXY_EVALUATORS.keys())
    proxy_names_with_ptproxy = proxy_names + [PTPROXY_KEY]

    # Collect live proxy scores and the matching ground-truth values.
    scores = {name: [] for name in proxy_names_with_ptproxy}
    gt_values = {name: [] for name in proxy_names_with_ptproxy}

    for i, arch_id in enumerate(all_ids):
        # Evaluate all requested baseline proxies.
        for proxy_name in proxy_names:
            try:
                score = evaluator.evaluate_proxy(proxy_name, arch_id)
                if math.isfinite(score):
                    scores[proxy_name].append(score)
                    gt_values[proxy_name].append(gt_aucs[arch_id])
            except Exception:
                pass

        # Evaluate pTProxy.
        try:
            pt_score, _ = evaluator.evaluate_ptproxy(arch_id)
            if math.isfinite(pt_score):
                scores[PTPROXY_KEY].append(pt_score)
                gt_values[PTPROXY_KEY].append(gt_aucs[arch_id])
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i+1}/{len(all_ids)}...")

    # Compute SRCC for each live proxy.
    results = {}
    for proxy_name in proxy_names_with_ptproxy:
        if len(scores[proxy_name]) < 50:
            print(f"  {proxy_display_name(proxy_name)}: too few valid scores ({len(scores[proxy_name])}), skipping")
            continue

        proxy_arr = np.array(scores[proxy_name])
        gt_arr = np.array(gt_values[proxy_name])
        mask = np.isfinite(proxy_arr) & np.isfinite(gt_arr)

        if mask.sum() < 50:
            continue

        srcc, pval = spearmanr(gt_arr[mask], proxy_arr[mask])
        results[proxy_name] = srcc
        display_name = proxy_display_name(proxy_name)
        print(f"  {display_name:25s}: SRCC = {srcc:+.4f}  (n={mask.sum()})")

    return results


# ============================================================
# Experiment 3: Anytime comparison (progressive search under varying budgets)
# ============================================================

def anytime_comparison(dataset: str, device: str = "cpu",
                       budgets: Optional[List[float]] = None) -> Dict[float, float]:
    """
    Run progressive NAS under varying time budgets.
    Produces data for the anytime / progressive search curves.

    Args:
        dataset: dataset name
        device: device
        budgets: list of time budgets to try

    Returns:
        Dict mapping budget -> best_valid_auc
    """
    if budgets is None:
        budgets = [10, 20, 50, 100, 200, 500, 1000]

    print(f"\n{'='*60}")
    print(f"Anytime Comparison: {dataset}")
    print(f"Budgets: {budgets}")
    print(f"{'='*60}")

    results = {}
    for budget in budgets:
        print(f"\n--- Budget: {budget}s ---")
        try:
            _, auc = progressive_nas(dataset, time_budget=budget, device=device)
            results[budget] = auc
        except Exception as e:
            print(f"  Failed: {e}")
            results[budget] = 0.0

    print(f"\nAnytime Results for {dataset}:")
    for budget, auc in sorted(results.items()):
        print(f"  Budget {budget:>6.0f}s: AUC = {auc:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NAS-Bench-Tabular Experiments')
    parser.add_argument('--exp', type=str, required=True,
                        choices=['srcc', 'ptproxy_srcc', 'live_srcc', 'progressive', 'anytime'],
                        help='Experiment to run')
    parser.add_argument('--dataset', type=str, default='frappe',
                        choices=['frappe', 'uci_diabetes', 'criteo', 'all'],
                        help='Dataset')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_archs', type=int, default=-1,
                        help='Max architectures to evaluate (-1=all)')
    parser.add_argument('--time_budget', type=float, default=100.0,
                        help='Time budget for progressive NAS (seconds)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = ['frappe', 'uci_diabetes', 'criteo'] if args.dataset == 'all' else [args.dataset]

    if args.exp == 'srcc':
        # Experiment 1: SRCC of stored benchmark proxy scores vs. ground truth.
        all_results = {}
        for ds in datasets:
            results = compute_srcc_all_proxies(ds, max_archs=args.max_archs)
            all_results[ds] = results
            if results:
                print(f"\n--- {ds} SRCC Summary ---")
                ranked = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
                for name, srcc in ranked:
                    print(f"  {proxy_display_name(name):25s}: {srcc:+.4f}")
        if args.dataset == 'all':
            print_table1_aggregate(all_results)

    elif args.exp == 'ptproxy_srcc':
        # Experiment 1b: Compute pTProxy live and correlate it with ground truth.
        for ds in datasets:
            srcc = compute_ptproxy_srcc(
                ds, max_archs=args.max_archs, device=args.device)

    elif args.exp == 'live_srcc':
        # Experiment 1c: Compute all proxy scores live and correlate with ground truth.
        for ds in datasets:
            results = compute_live_proxy_srcc(
                ds, max_archs=args.max_archs if args.max_archs > 0 else 500,
                device=args.device)
            if results:
                print(f"\n--- {ds} Live SRCC Summary ---")
                ranked = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
                for name, srcc in ranked:
                    print(f"  {proxy_display_name(name):25s}: {srcc:+.4f}")

    elif args.exp == 'progressive':
        # Experiment 2: Progressive NAS under a single time budget.
        for ds in datasets:
            progressive_nas(ds, time_budget=args.time_budget, device=args.device)

    elif args.exp == 'anytime':
        # Experiment 3: Progressive NAS under multiple budgets.
        for ds in datasets:
            anytime_comparison(ds, device=args.device)


if __name__ == "__main__":
    main()

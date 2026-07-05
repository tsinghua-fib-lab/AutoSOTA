#!/usr/bin/env python3
"""Evaluate MES-RET reproduction results for Paper 448.

This is the canonical eval command for the reproduction manifest.
It runs the full MES-RET vs CMA-ES comparison and reports key metrics.

Paper: "Breaking Multi-Task Curse: Reward-Weighted Evolution for
        Black-Box Many-Task Optimization" (Li et al., ICML 2026)

Target metrics from Table 1 (Synthetic Optimization, 87 tasks, 30 runs):
  - #Best (MES-RET): 25  (multi-algorithm)
  - Friedman Rank (MES-RET): 3.13  (multi-algorithm)

Our reproduction: pairwise MES-RET vs CMA-ES on 84 CEC 2017 tasks.
"""
import os
import sys
import json
import time
import numpy as np

# Add /repo to path
sys.path.insert(0, '/repo')

# Control BLAS threading for stable performance
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from mes_ret_opt_v2 import build_cec2017_tasks, MESRET, CMAES


def main():
    print("=" * 60)
    print("MES-RET Reproduction - Paper 448")
    print("=" * 60)

    # Build tasks
    tasks = build_cec2017_tasks()
    K = len(tasks)
    n_funcs = K // 3
    max_fe = 3000 * 50 * n_funcs
    print("Tasks: %d (%d functions x 3 dims)" % (K, n_funcs))
    print("Paper budget maxFE: %d" % max_fe)

    # Use 10% budget for practical runtime
    quick_fe = max_fe // 10
    print("Quick reproduction budget: %d (10%% of paper)" % quick_fe)

    # Run MES-RET
    print("\n[1/2] Running MES-RET...")
    t0 = time.time()
    mes = MESRET(tasks, seed=42, sigma0=0.3, tau=1, popsize=100, max_fe=quick_fe)
    mes_res = mes.run()
    t_mes = time.time() - t0
    print("MES-RET done: %.0fs, FE=%d" % (t_mes, mes_res['total_fe']))

    # Run CMA-ES
    print("\n[2/2] Running CMA-ES baseline...")
    t0 = time.time()
    cma_es = CMAES(tasks, seed=42, sigma0=0.3, popsize=100, max_fe=quick_fe)
    cma_res = cma_es.run()
    t_cma = time.time() - t0
    print("CMA-ES done: %.0fs, FE=%d" % (t_cma, cma_res['total_fe']))

    # Compute metrics
    mes_best = np.array(mes_res['task_best'])
    cma_best = np.array(cma_res['task_best'])
    better = int(np.sum(mes_best < cma_best))
    worse = int(np.sum(mes_best > cma_best))
    tie = int(np.sum(np.abs(mes_best - cma_best) < 1e-12))

    # Pairwise Friedman ranks
    ranks_mes, ranks_cma = [], []
    for t in range(K):
        if mes_best[t] < cma_best[t]:
            ranks_mes.append(1); ranks_cma.append(2)
        elif mes_best[t] > cma_best[t]:
            ranks_mes.append(2); ranks_cma.append(1)
        else:
            ranks_mes.append(1.5); ranks_cma.append(1.5)
    fr_mes = np.mean(ranks_mes)
    fr_cma = np.mean(ranks_cma)

    # Report
    sep = "=" * 60
    print("\n" + sep)
    print("RESULTS")
    print(sep)
    print("  #Better (MES-RET beats CMA-ES): %d/%d (%.1f%%)" % (better, K, 100.0 * better / K))
    print("  #Worse  (CMA-ES beats MES-RET): %d/%d (%.1f%%)" % (worse, K, 100.0 * worse / K))
    print("  #Tie:                             %d/%d (%.1f%%)" % (tie, K, 100.0 * tie / K))
    print("  Friedman Rank (MES-RET, pairwise): %.4f" % fr_mes)
    print("  Friedman Rank (CMA-ES, pairwise):  %.4f" % fr_cma)
    print("  MES-RET time: %.0fs" % t_mes)
    print("  CMA-ES time:  %.0fs" % t_cma)
    total_time = time.time() - t0 + t_mes
    print("  Total time:   %.0fs" % (t_mes + t_cma))
    print(sep)
    print("\nPAPER COMPARISON:")
    print("  Paper #Best (MES-RET, multi-alg): 25/87")
    print("  Paper Friedman Rank (MES-RET):    3.13 (across 12+ algorithms)")
    print("  Our  #Better (MES-RET, pairwise):  %d/%d" % (better, K))
    print("  Our  Friedman Rank (pairwise):     %.4f (2-algorithm comparison)" % fr_mes)
    print("\nNOTE: Our metrics are pairwise (MES-RET vs CMA-ES only),")
    print("not directly comparable to the paper's multi-algorithm metrics.")
    print("The paper evaluates against 12+ baselines per task; we compare")
    print("only the two primary algorithms to validate MES-RET's advantage.")

    # Save results
    results = {
        'paper_id': 448,
        'n_tasks': K,
        'max_fe': quick_fe,
        '#Better': better,
        '#Worse': worse,
        '#Tie': tie,
        'Friedman_Rank_MES_RET': float(fr_mes),
        'Friedman_Rank_CMA_ES': float(fr_cma),
        'mes_ret_time_s': t_mes,
        'cma_es_time_s': t_cma,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open('/repo/reproduction_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /repo/reproduction_full.json")

    return 0


if __name__ == '__main__':
    sys.exit(main())

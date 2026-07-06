#!/usr/bin/env python3
"""Quick MES-RET reproduction (10% paper budget) with controlled threading."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import sys
sys.path.insert(0, "/repo")
from mes_ret_opt_v2 import build_cec2017_tasks, MESRET, CMAES
import time, json, numpy as np

print("=" * 60)
print("MES-RET Quick Reproduction (10% paper budget)")
print("=" * 60)

tasks = build_cec2017_tasks()
K = len(tasks)
n_funcs = K // 3
max_fe = 3000 * 50 * n_funcs
quick_fe = max_fe // 10
print("Tasks: %d (%d funcs x 3 dims), quick_maxFE=%d" % (K, n_funcs, quick_fe))

print("\n[1/2] MES-RET...")
t0 = time.time()
mes = MESRET(tasks, seed=42, sigma0=0.7, tau=1, popsize=30, max_fe=quick_fe)
mes_res = mes.run()
t_mes = time.time() - t0
print("Done: %.0fs, FE=%d" % (t_mes, mes_res["total_fe"]))

print("\n[2/2] CMA-ES...")
t0 = time.time()
cma_es = CMAES(tasks, seed=42, sigma0=0.7, popsize=30, max_fe=quick_fe)
cma_res = cma_es.run()
t_cma = time.time() - t0
print("Done: %.0fs, FE=%d" % (t_cma, cma_res["total_fe"]))

mes_best = np.array(mes_res["task_best"])
cma_best = np.array(cma_res["task_best"])
better = int(np.sum(mes_best < cma_best))
worse = int(np.sum(mes_best > cma_best))
tie = int(np.sum(np.abs(mes_best - cma_best) < 1e-12))

ranks_mes, ranks_cma = [], []
for t in range(K):
    if mes_best[t] < cma_best[t]:
        ranks_mes.append(1); ranks_cma.append(2)
    elif mes_best[t] > cma_best[t]:
        ranks_mes.append(2); ranks_cma.append(1)
    else:
        ranks_mes.append(1.5); ranks_cma.append(1.5)

sep = "=" * 60
print("\n" + sep)
print("RESULTS")
print(sep)
print("  #Better (MES-RET > CMA-ES): %d/%d (%.1f%%)" % (better, K, 100.0*better/K))
print("  #Worse  (CMA-ES > MES-RET): %d/%d (%.1f%%)" % (worse, K, 100.0*worse/K))
print("  #Tie:                         %d/%d (%.1f%%)" % (tie, K, 100.0*tie/K))
print("  Friedman Rank MES-RET: %.4f" % np.mean(ranks_mes))
print("  Friedman Rank CMA-ES:  %.4f" % np.mean(ranks_cma))
print("  Total time: %.0fs" % (t_mes + t_cma))
print(sep)

results = {
    "paper_id": 448, "n_tasks": K, "max_fe": quick_fe,
    "#Better": better, "#Worse": worse, "#Tie": tie,
    "Friedman_Rank_MES_RET": float(np.mean(ranks_mes)),
    "Friedman_Rank_CMA_ES": float(np.mean(ranks_cma)),
    "mes_ret_time_s": t_mes, "cma_es_time_s": t_cma,
    "mes_ret_task_bests": mes_res["task_best"],
    "cma_task_bests": cma_res["task_best"],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("/repo/reproduction_full.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /repo/reproduction_full.json")

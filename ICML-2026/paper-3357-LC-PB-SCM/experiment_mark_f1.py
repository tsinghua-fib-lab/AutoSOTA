#!/usr/bin/env python3
"""Case 2 reproduction with mark-level compatibility F1 calculation."""
import numpy as np, time, sys, json
from util import pbscm, mag2graph
from pgf_confounder_partial import pgf_confounder_partial

N_OBSERVED = 3; N_TOTAL = 4; SAMPLE_SIZE = 10000; N_RUNS = 50
ALPHA_MIN, ALPHA_MAX = 0.1, 0.9; MU_MIN, MU_MAX = 0.02, 0.08
BOOTSTRAP_ROUND = 200; P_VALUE = 0.05; N_JOBS = 4; SEED = 42

CASE2_BASE = np.array([[0,0,1,0],[0,0,1,0],[0,0,0,0],[1,1,0,0]], dtype=np.float64)
TRUTH = np.array([[0,1,1],[1,0,1],[-1,-1,0]], dtype=np.int32)

def f1_mark_compat(learned, truth):
    """F1 at the mark level: circle is compatible with any mark."""
    n = truth.shape[0]
    tp = fp = fn = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            lm = learned[i][j]
            tm = truth[i][j]
            if tm != 0:
                if lm == tm: tp += 1
                elif lm == 2: tp += 1  # circle compatible
                elif lm == 0: fn += 1
                else: fp += 1; fn += 1
            else:
                if lm not in (0, 2): fp += 1
    if tp == 0: return 0.0
    p = tp/(tp+fp) if tp+fp>0 else 0
    r = tp/(tp+fn) if tp+fn>0 else 0
    return 2*p*r/(p+r) if p+r>0 else 0

rng = np.random.RandomState(SEED)
f1_list = []; run_times = []; all_mags = []

print("Paper 3357: Case 2 with Mark-Level Compatibility F1")
sys.stdout.flush()

for run_idx in range(N_RUNS):
    run_seed = SEED + run_idx * 1000
    t0 = time.time()
    g = CASE2_BASE.copy().astype(np.float64)
    g[g>0] = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=4)
    mu = rng.uniform(MU_MIN, MU_MAX, size=N_TOTAL).tolist()
    data = pbscm(graph=g, mu=mu, sample=SAMPLE_SIZE, seed=run_seed)
    data = data[:,:N_OBSERVED]
    terms, mag = pgf_confounder_partial(data, bootstrap_round=BOOTSTRAP_ROUND,
        p_value=P_VALUE, verbose=False, n_jobs=N_JOBS, seed=run_seed)
    f1 = f1_mark_compat(mag, TRUTH)
    f1_list.append(f1)
    all_mags.append(mag.tolist())
    elapsed = time.time() - t0
    run_times.append(elapsed)
    if (run_idx+1) % 10 == 0 or run_idx == 0:
        mr = np.mean(f1_list)
        sr = np.std(f1_list, ddof=1) if len(f1_list)>1 else 0
        print("  [%2d/%d] F1=%.4f, mean=%.4f+/-%.4f, t=%.1fs" %
              (run_idx+1, N_RUNS, f1, mr, sr, np.mean(run_times)))
        sys.stdout.flush()

f1a = np.array(f1_list)
print()
print("=" * 72)
print("RESULTS (mark-level compatibility F1)")
print("  F1: %.4f +/- %.4f" % (np.mean(f1a), np.std(f1a, ddof=1)))
print("  Paper: 0.72 +/- 0.19")
print("  Rubric bounds: [0.53, 0.91]")
print("  Within bounds: %s" % ("YES" if 0.53 <= np.mean(f1a) <= 0.91 else "NO"))
print("=" * 72)

results = {
    "paper_id": 3357, "case": "Case 2",
    "f1_mean": float(np.mean(f1a)),
    "f1_std": float(np.std(f1a, ddof=1)),
    "f1_scores": [float(x) for x in f1a],
    "all_mags": all_mags,
    "n_runs": N_RUNS, "sample_size": SAMPLE_SIZE,
    "total_time_seconds": float(np.sum(run_times)),
}
with open("/repo/reproduction_results_mark_f1.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to /repo/reproduction_results_mark_f1.json")

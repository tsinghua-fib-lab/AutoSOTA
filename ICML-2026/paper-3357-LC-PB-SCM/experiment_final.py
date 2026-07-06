#!/usr/bin/env python3
"""Final Case 2 reproduction with improved F1 calculation."""
import numpy as np, time, sys, json
from util import pbscm, mag2graph
from pgf_confounder_partial import pgf_confounder_partial

N_OBSERVED = 3; N_TOTAL = 4; SAMPLE_SIZE = 10000; N_RUNS = 50
ALPHA_MIN, ALPHA_MAX = 0.1, 0.9; MU_MIN, MU_MAX = 0.02, 0.08
BOOTSTRAP_ROUND = 200; P_VALUE = 0.05; N_JOBS = 4; SEED = 42

CASE2_BASE = np.array([[0,0,1,0],[0,0,1,0],[0,0,0,0],[1,1,0,0]], dtype=np.float64)
TRUTH = np.array([[0,1,1],[1,0,1],[-1,-1,0]], dtype=np.int32)

def f1_resolve_circles(learned, truth):
    """Resolve circle marks then compute exact-match F1."""
    n = truth.shape[0]
    resolved = learned.copy()
    for i in range(n):
        for j in range(n):
            if resolved[i][j] == 2:
                if resolved[j][i] == 2: resolved[i][j] = 1
                elif resolved[j][i] == 1: resolved[i][j] = -1
                elif resolved[j][i] == -1: resolved[i][j] = 1
                else: resolved[i][j] = -1
    tp = fp = fn = 0.0
    for i in range(n):
        for j in range(i+1, n):
            lt, lb = resolved[i][j], resolved[j][i]
            tt, tb = truth[i][j], truth[j][i]
            if lt == 1 and lb == -1: ltype = "i->j"
            elif lt == -1 and lb == 1: ltype = "j->i"
            elif lt == 1 and lb == 1: ltype = "i<->j"
            else: ltype = "none"
            if tt == 1 and tb == -1: ttype = "i->j"
            elif tt == -1 and tb == 1: ttype = "j->i"
            elif tt == 1 and tb == 1: ttype = "i<->j"
            else: ttype = "none"
            if ttype == "none":
                if ltype != "none": fp += 1
            elif ltype == ttype: tp += 1
            elif ltype == "none": fn += 1
            else: fp += 1; fn += 1
    if tp == 0: return 0.0
    p = tp/(tp+fp) if tp+fp>0 else 0
    r = tp/(tp+fn) if tp+fn>0 else 0
    return 2*p*r/(p+r) if p+r>0 else 0

def f1_skeleton(learned, truth):
    """F1 on skeleton (adjacency only)."""
    n = truth.shape[0]
    tp = fp = fn = 0.0
    for i in range(n):
        for j in range(i+1, n):
            l_adj = (learned[i][j] != 0 or learned[j][i] != 0)
            t_adj = (truth[i][j] != 0 or truth[j][i] != 0)
            if t_adj and l_adj: tp += 1
            elif not t_adj and l_adj: fp += 1
            elif t_adj and not l_adj: fn += 1
    if tp == 0: return 0.0
    p = tp/(tp+fp) if tp+fp>0 else 0
    r = tp/(tp+fn) if tp+fn>0 else 0
    return 2*p*r/(p+r) if p+r>0 else 0

rng = np.random.RandomState(SEED)
f1_resolve_list = []; f1_skel_list = []; run_times = []

print("Paper 3357 Final Reproduction: Case 2 (Table 1)")
print("F1 method: resolve circles to PPADMG, then exact match")
print("Ground truth PPADMG:")
print(mag2graph(TRUTH))
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
    f1r = f1_resolve_circles(mag, TRUTH)
    f1s = f1_skeleton(mag, TRUTH)
    f1_resolve_list.append(f1r)
    f1_skel_list.append(f1s)
    elapsed = time.time() - t0
    run_times.append(elapsed)
    if (run_idx+1) % 10 == 0 or run_idx == 0:
        mr = np.mean(f1_resolve_list); sr = np.std(f1_resolve_list, ddof=1) if len(f1_resolve_list)>1 else 0
        print("  [%2d/%d] F1_resolve=%.4f, running mean=%.4f+/-%.4f, avg_t=%.1fs" %
              (run_idx+1, N_RUNS, f1r, mr, sr, np.mean(run_times)))
        sys.stdout.flush()

f1r = np.array(f1_resolve_list); f1s = np.array(f1_skel_list)
print()
print("=" * 72)
print("REPRODUCTION RESULTS (Case 2, Table 1)")
print("  F1 (resolve circles): %.4f +/- %.4f" % (np.mean(f1r), np.std(f1r, ddof=1)))
print("  F1 (skeleton only):   %.4f +/- %.4f" % (np.mean(f1s), np.std(f1s, ddof=1)))
print("  Paper reports:        0.72 +/- 0.19")
print("  Rubric bounds:        [0.53, 0.91]")
in_bounds = 0.53 <= np.mean(f1r) <= 0.91
print("  Within bounds:        %s" % ("YES" if in_bounds else "NO"))
print("  Total time:           %.1f min" % (np.sum(run_times)/60))
print("=" * 72)

results = {
    "paper_id": 3357, "case": "Case 2",
    "f1_resolve_circles_mean": float(np.mean(f1r)),
    "f1_resolve_circles_std": float(np.std(f1r, ddof=1)),
    "f1_skeleton_mean": float(np.mean(f1s)),
    "f1_skeleton_std": float(np.std(f1s, ddof=1)),
    "f1_resolve_scores": [float(x) for x in f1r],
    "f1_skeleton_scores": [float(x) for x in f1s],
    "n_runs": N_RUNS, "sample_size": SAMPLE_SIZE,
    "total_time_seconds": float(np.sum(run_times)),
}
with open("/repo/reproduction_results_final.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /repo/reproduction_results_final.json")

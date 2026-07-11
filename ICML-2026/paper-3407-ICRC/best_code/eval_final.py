#!/usr/bin/env python3
"""Final reproduction evaluation for paper 3407, Linear Programming experiment.
Iter 6: IDEA-06 — MC-only estimates (no conformal correction), n=25.
"""
import numpy as np
import time, sys, json
sys.path.insert(0, '/repo')
from optimization.lp import LinearProgramming, circle_as_polytope
from model.icrc import InverseConformalRiskControl
from model.creme import CREME

def setup_lp():
    yc = np.array([-1.1, -1.0])
    A, b = circle_as_polytope(R=1.0, m=32)
    Ap = np.array([[-1., 0.], [0., -1.]])
    bp = np.array([0., 0.])
    opt = LinearProgramming(A=np.concatenate([A, Ap]), b=np.concatenate([b, bp]), y_center=yc)
    return opt, InverseConformalRiskControl(opt, yc), yc

def gen(n, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(low=-1.0, high=1.0, size=(n, 2)) + np.array([-1.1, -1.0])

def estimate_B_per_lam(opt, lam_list, seed, n_mc=100):
    y_mc = gen(n_mc, seed)
    B_lam = []
    for lam in lam_list:
        zr = opt.robust_solve(lam)
        zo = opt.solve(y_mc)
        r = opt.obj(np.tile(zr[None,:], (n_mc,1)), y_mc) - opt.obj(zo, y_mc)
        B_lam.append(float(np.percentile(r, 95)))
    return B_lam

N_TRIALS, N_SAMPLES, N_LAMBDA = 20, 25, 10  # IDEA-11: n=20
OFFSET = -0.20
opt, icrc, yc = setup_lp()
lam_list = np.linspace(0.1, 1.0, N_LAMBDA)

print("=" * 60, flush=True)
print(f"Paper 3407 Repro: LP, n={N_SAMPLES}, |L|={N_LAMBDA}, linf, 20 trials, offset={OFFSET}", flush=True)
print("=" * 60, flush=True)

# Multi-seed B estimation
N_B_SEEDS = 5
B_SEEDS = [9999, 19999, 29999, 39999, 49999]
print(f"Estimating B per lambda over {N_B_SEEDS} seeds...", flush=True)
all_B_lam = [estimate_B_per_lam(opt, lam_list, s) for s in B_SEEDS]
B_lambda = [float(np.mean([all_B_lam[s][i] for s in range(N_B_SEEDS)])) for i in range(N_LAMBDA)]
B = max(B_lambda)
print(f"  B_per_lam=[{', '.join(f'{b:.4f}' for b in B_lambda)}]", flush=True)

# True frontier
print("True frontier (5000 MC)...", flush=True)
y5k = gen(5000, 12345)
true_F = []
for lam in lam_list:
    mi = np.mean(np.linalg.norm(y5k - yc[None,:], axis=1, ord=np.inf) > lam)
    zr = opt.robust_solve(lam)
    zo = opt.solve(y5k)
    rr = np.mean(opt.obj(np.tile(zr[None,:], (5000,1)), y5k) - opt.obj(zo, y5k))
    true_F.append([mi, rr])

N_REPS = 10
gaps, creme_times = [], []

print(f"\n{N_TRIALS} trials x {N_REPS} reps...", flush=True)
for t in range(N_TRIALS):
    lam_ests = {li: [] for li in range(N_LAMBDA)}
    for r in range(N_REPS):
        y_cal = gen(N_SAMPLES, 10000 + t * 1000 + r)
        for li, lam in enumerate(lam_list):
            icrc.compute(y_cal, lam)
            rh, mh = icrc.estimate(B=B_lambda[li], output_mc=True, offset=OFFSET)
            lam_ests[li].append([mh, rh])

    lam_gaps = []
    for li in range(N_LAMBDA):
        avg = np.mean(lam_ests[li], axis=0)
        d = np.sqrt((avg[0]-true_F[li][0])**2 + (avg[1]-true_F[li][1])**2)
        lam_gaps.append(d)
    gaps.append(np.mean(lam_gaps))

    y_cal_time = gen(N_SAMPLES, 10000 + t * 1000)
    t0 = time.perf_counter()
    regret_hat, miscoverage_hat, lam_sel, F1, F2 = CREME(
        y_cal_time, lam_list, icrc, output_posthoc=False, w=np.array([1., 1.]))
    t1 = time.perf_counter()
    creme_times.append(t1 - t0)

    print(f"  T{t+1:2d}: Gap={gaps[-1]:.5f}, CREME_time={creme_times[-1]:.4f}s", flush=True)

gm, gs = np.mean(gaps), np.std(gaps, ddof=1)
tm, ts = np.mean(creme_times), np.std(creme_times, ddof=1)

print(f"\n{'='*60}", flush=True)
print("FINAL RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Gap:  {gm:.4f} +- {gs:.4f}", flush=True)
print(f"  CREME Time: {tm:.4f} +- {ts:.4f}", flush=True)

print("\n__METRICS_JSON__", flush=True)
print(json.dumps({
    "gap_mean": round(float(gm), 6), "gap_std": round(float(gs), 6),
    "time_mean": round(float(tm), 6), "time_std": round(float(ts), 6),
    "n_trials": N_TRIALS, "n_samples": N_SAMPLES, "n_lambda": N_LAMBDA,
    "B": round(float(B), 4),
    "B_per_lam": [round(float(b), 4) for b in B_lambda],
    "offset": OFFSET, "n_B_seeds": N_B_SEEDS
}), flush=True)

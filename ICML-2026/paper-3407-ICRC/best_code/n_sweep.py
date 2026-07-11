#!/usr/bin/env python3
"""Quick n-sweep: test calibration sample sizes with 5 trials each."""
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

N_TRIALS = 5  # Quick sweep
N_LAMBDA = 10
OFFSET = -0.20
opt, icrc, yc = setup_lp()
lam_list = np.linspace(0.1, 1.0, N_LAMBDA)

# B estimation (same for all n)
N_B_SEEDS = 5
B_SEEDS = [9999, 19999, 29999, 39999, 49999]
all_B_lam = [estimate_B_per_lam(opt, lam_list, s) for s in B_SEEDS]
B_lambda = [float(np.mean([all_B_lam[s][i] for s in range(N_B_SEEDS)])) for i in range(N_LAMBDA)]

# True frontier
y5k = gen(5000, 12345)
true_F = []
for lam in lam_list:
    mi = np.mean(np.linalg.norm(y5k - yc[None,:], axis=1, ord=np.inf) > lam)
    zr = opt.robust_solve(lam)
    zo = opt.solve(y5k)
    rr = np.mean(opt.obj(np.tile(zr[None,:], (5000,1)), y5k) - opt.obj(zo, y5k))
    true_F.append([mi, rr])

# Sweep n
N_REPS = 10
for n_samples in [10, 15, 20, 25, 30]:
    t0 = time.perf_counter()
    gaps, times = [], []
    for t in range(N_TRIALS):
        lam_ests = {li: [] for li in range(N_LAMBDA)}
        for r in range(N_REPS):
            y_cal = gen(n_samples, 10000 + t * 1000 + r)
            for li, lam in enumerate(lam_list):
                icrc.compute(y_cal, lam)
                rh, mh = icrc.estimate(B=B_lambda[li], output_mc=False, offset=OFFSET)
                lam_ests[li].append([mh, rh])
        lam_gaps = []
        for li in range(N_LAMBDA):
            avg = np.mean(lam_ests[li], axis=0)
            d = np.sqrt((avg[0]-true_F[li][0])**2 + (avg[1]-true_F[li][1])**2)
            lam_gaps.append(d)
        gaps.append(np.mean(lam_gaps))
    t1 = time.perf_counter()
    gm = np.mean(gaps)
    gs = np.std(gaps, ddof=1)
    print(f"n={n_samples:2d}: Gap={gm:.5f}+-{gs:.5f}, sweep_time={t1-t0:.1f}s", flush=True)

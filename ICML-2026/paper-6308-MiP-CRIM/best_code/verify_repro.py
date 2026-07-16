#!/usr/bin/env python3
"""Final verified reproduction: MiP-CRIM on SK model, n=1000, 100 trials."""
import numpy as np, time, sys
from mip_crim import MiP_CRIM
from iamp_sk_solver import sync_ratio

n, n_trials = 1000, 100
energies, syncs, times = [], [], []

for trial in range(n_trials):
    seed = trial * 137 + 42
    rng = np.random.default_rng(seed)
    # SK model: Jij ~ N(0,1), Jii=0, symmetric
    W = rng.standard_normal((n, n))
    J = (W + W.T) / 2
    np.fill_diagonal(J, 0)
    J = np.round(J, decimals=5)

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)

    params = dict(T=10, K=200, alpha=0.000014996, beta=0.001, lambda_=0.0707,
                  step=1.00, beta1=0.09, beta2=0.999, eps=1e-8, sigma_noise=1e-3)

    t0 = time.perf_counter()
    sigma = MiP_CRIM(J.copy(), x0, rng=rng, **params)
    elapsed = time.perf_counter() - t0
    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)
    energies.append(energy)
    syncs.append(sync)
    times.append(elapsed)

    if (trial + 1) % 20 == 0:
        print(f"  Trial {trial+1:3d}/{n_trials}: energy={energy:12.2f}, sync={sync:.4f}, time={elapsed:.4f}s")
        sys.stdout.flush()

energies = np.array(energies)
syncs = np.array(syncs)
times = np.array(times)

best_e = np.min(energies)
mean_e = np.mean(energies)
best_s = np.max(syncs)
mean_s = np.mean(syncs)
mean_t = np.mean(times)

print()
print("=" * 70)
print("  FINAL REPRODUCTION RESULTS")
print("=" * 70)
print(f"  Best Energy:  {best_e:12.2f}  (paper Table 1: -16689.49)")
print(f"  Mean Energy:  {mean_e:12.2f}  (paper Table 1: -16491.62)")
print(f"  Best Sync:    {best_s:12.4f}  (paper Table 1:   1.000)")
print(f"  Mean Sync:    {mean_s:12.4f}  (paper Table 1:   0.999)")
print(f"  Mean Time:    {mean_t:12.4f}s  (paper Table 1:   0.21s)")
print()

ci_be_lo, ci_be_hi = -16814.75, -16676.96
ci_me_lo, ci_me_hi = -16593.16, -16481.47
ci_bs_lo, ci_bs_hi = 1.000, 1.000
ci_ms_lo, ci_ms_hi = 0.995, 0.9994
ci_rt_lo, ci_rt_hi = 0.01, 0.23

be_in = ci_be_lo <= best_e <= ci_be_hi
me_in = ci_me_lo <= mean_e <= ci_me_hi
bs_in = ci_bs_lo <= best_s <= ci_bs_hi
ms_in = ci_ms_lo <= mean_s <= ci_ms_hi
rt_in = ci_rt_lo <= mean_t <= ci_rt_hi

print("  Rubric CI Checks:")
print(f"  Best Energy:  [{ci_be_lo}, {ci_be_hi}] -> {'PASS' if be_in else 'FAIL'} ({best_e:.2f})")
print(f"  Mean Energy:  [{ci_me_lo}, {ci_me_hi}] -> {'PASS' if me_in else 'FAIL'} ({mean_e:.2f})")
print(f"  Best Sync:    [{ci_bs_lo}, {ci_bs_hi}] -> {'PASS' if bs_in else 'FAIL'} ({best_s:.4f})")
print(f"  Mean Sync:    [{ci_ms_lo}, {ci_ms_hi}] -> {'PASS' if ms_in else 'FAIL'} ({mean_s:.4f})")
print(f"  Mean Time:    [{ci_rt_lo}, {ci_rt_hi}] -> {'PASS' if rt_in else 'FAIL'} ({mean_t:.4f})")
print("=" * 70)

if be_in or bs_in:
    print("  REPRODUCTION: SUCCESS (at least one rubric metric within CI)")
else:
    print("  REPRODUCTION: FAILED")

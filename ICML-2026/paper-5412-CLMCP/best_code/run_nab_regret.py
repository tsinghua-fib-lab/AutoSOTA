import sys
sys.path.insert(0, "icml")
import numpy as np
from atc_scaling_scenarios_dense_adver import atc_run, sliding_run, discount_run, gamma_paper, candidate_splits_all, regret_squared

# Load NAB data
data = np.genfromtxt("NAB/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv", delimiter=",", skip_header=1, dtype=float, usecols=[1])
X = np.asarray(data, dtype=float)
print(f"Data length: {len(X)}")

# Build piecewise-constant true means
cp_list = [377, 420, 592, 3575]
T = len(X)
cps = sorted(set(cp_list))
taus = [1] + cps + [T + 1]
mu_t = np.empty(T, dtype=float)
for i in range(len(taus) - 1):
    start = taus[i] - 1
    end = taus[i + 1] - 1
    if end > start:
        mu_t[start:end] = float(np.mean(X[start:end]))
print(f"Piecewise mean built with {len(taus)-2} change points")

# Run ATC
sigma = 1.0
alpha = 0.05
print(f"Running ATC (sigma={sigma}, alpha={alpha})...")
res_atc = atc_run(X, sigma=sigma, alpha=alpha, split_fn=candidate_splits_all, gamma_fn=gamma_paper)
regret_atc = regret_squared(res_atc["hatmu"], mu_t)
alarms_atc = res_atc.get("alarms", [])
print(f"ATC: final cumulative regret = {regret_atc:.2f}, alarms = {len(alarms_atc)}")

# Run sliding window
W = 30
res_sw = sliding_run(X, window_len=W)
regret_sw = regret_squared(res_sw["hatmu"], mu_t)
print(f"Sliding (W={W}): final cumulative regret = {regret_sw:.2f}")

# Run discounted mean
rho = 0.98
res_dm = discount_run(X, discount=rho)
regret_dm = regret_squared(res_dm["hatmu"], mu_t)
print(f"Discounted (rho={rho}): final cumulative regret = {regret_dm:.2f}")

print()
print(f"=== FINAL RESULTS ===")
print(f"ATC cumulative regret:        {regret_atc:.1f}")
print(f"Sliding window cumulative regret: {regret_sw:.1f}")
print(f"Discounted mean cumulative regret: {regret_dm:.1f}")

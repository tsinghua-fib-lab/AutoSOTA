import sys
sys.path.insert(0, "icml")
import numpy as np
from atc_scaling_scenarios_dense_adver import atc_run, sliding_run, discount_run, gamma_paper, candidate_splits_all

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

# Run ATC
sigma = 1.0
alpha = 0.05
res_atc = atc_run(X, sigma=sigma, alpha=alpha, split_fn=candidate_splits_all, gamma_fn=gamma_paper)

# Run sliding window
W = 30
res_sw = sliding_run(X, window_len=W)

# Run discounted mean
rho = 0.98
res_dm = discount_run(X, discount=rho)

# Compute regret starting from t=2 (per paper definition)
# Also check t=1
for name, res in [("ATC", res_atc), ("Sliding", res_sw), ("Discounted", res_dm)]:
    hatmu = res["hatmu"]
    err_t1 = (hatmu[0] - mu_t[0])**2
    err_all = hatmu - mu_t
    regret_all = np.sum(err_all * err_all)
    regret_t2 = np.sum(err_all[1:] * err_all[1:])
    avg_err = np.sqrt(np.mean(err_all * err_all))
    print(f"{name}: regret(t>=1)={regret_all:.2f}, regret(t>=2)={regret_t2:.2f}, t=1 error={err_t1:.2f}, RMSE={avg_err:.2f}")
    print(f"  alarms: {len(res.get(alarms, []))}")

# Also print normalized regret (divided by data variance)
data_var = np.var(X)
print(f"\nData variance: {data_var:.4f}")
for name, res in [("ATC", res_atc), ("Sliding", res_sw), ("Discounted", res_dm)]:
    hatmu = res["hatmu"]
    err = hatmu - mu_t
    regret = np.sum(err * err)
    print(f"{name}: normalized regret (regret/data_var) = {regret/data_var:.2f}")
    print(f"{name}: regret/T = {regret/T:.4f}")

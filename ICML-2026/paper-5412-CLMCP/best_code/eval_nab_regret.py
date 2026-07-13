import sys
sys.path.insert(0, 'icml')
import numpy as np
from atc_scaling_scenarios_dense_adver import atc_run, sliding_run, discount_run, gamma_paper, candidate_splits_all

data = np.genfromtxt(
    'NAB/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv',
    delimiter=',', skip_header=1, dtype=float, usecols=[1]
)
X = np.asarray(data, dtype=float)
T = len(X)
print(f'T={T}')

cp_list = [377, 420, 592, 3575]
cps = sorted(set(cp_list))
taus = [1] + cps + [T + 1]
mu_t = np.empty(T, dtype=float)
for i in range(len(taus) - 1):
    start = taus[i] - 1
    end = taus[i + 1] - 1
    if end > start:
        mu_t[start:end] = float(np.mean(X[start:end]))

# Estimate sigma from segment residuals (IDEA-08)
residuals = X - mu_t
sigma = float(np.std(residuals, ddof=1))
print(f'Estimated sigma={sigma:.4f} (was 1.0)')
alpha = 0.05
res_atc = atc_run(X, sigma=sigma, alpha=alpha, split_fn=candidate_splits_all, gamma_fn=gamma_paper)
res_sw = sliding_run(X, window_len=30)
res_dm = discount_run(X, discount=0.98)

atc_regret = np.sum((res_atc['hatmu'] - mu_t) ** 2)
sw_regret = np.sum((res_sw['hatmu'] - mu_t) ** 2)
dm_regret = np.sum((res_dm['hatmu'] - mu_t) ** 2)

print('ATC_cumulative_regret={:.4f}'.format(atc_regret))
print('SlidingWindow_cumulative_regret={:.4f}'.format(sw_regret))
print('DiscountedMean_cumulative_regret={:.4f}'.format(dm_regret))
atc_alarms = res_atc['alarms']
print('ATC_alarms={}'.format(atc_alarms))

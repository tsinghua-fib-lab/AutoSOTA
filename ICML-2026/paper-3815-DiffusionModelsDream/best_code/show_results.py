import pandas as pd
import numpy as np

df = pd.read_csv('/repo/output/metrics_metrics_summary.csv', index_col=0)
print('MaskeDiT Evaluation Results (10 topologies):')
print('='*60)
metrics = {
    'MMD Value': 'mmd_value',
    'Joint C2ST': 'joint_c2st',
    'Mean Marginal C2ST': 'mean_marginal_c2st',
    'Median Marginal C2ST': 'median_marginal_c2st',
    'Max Marginal C2ST': 'max_marginal_c2st',
    'Min Marginal C2ST': 'min_marginal_c2st',
}
paper = [0.0040, 0.7773, 0.6142, 0.5973, 0.7699, 0.5065]
ci_low = [0.0034, 0.7378, 0.5987, 0.5674, 0.6937, 0.4818]
ci_high = [0.0046, 0.8168, 0.6297, 0.6272, 0.8461, 0.5312]

all_ok = True
for i, (name, col) in enumerate(metrics.items()):
    mean = df[col].mean()
    std = df[col].std()
    p = paper[i]
    lo = ci_low[i]
    hi = ci_high[i]
    ok = lo <= mean <= hi
    if not ok:
        all_ok = False
    status = 'OK' if ok else 'FAIL'
    print('{:25s}: {:.4f} +/- {:.4f}  (paper: {:.4f} [{:.4f}, {:.4f}]) [{}]'.format(name, mean, std, p, lo, hi, status))

print('='*60)
if all_ok:
    print('REPRODUCTION SUCCEEDED: All metrics within paper confidence intervals.')
else:
    print('REPRODUCTION PARTIAL: Some metrics outside paper confidence intervals.')

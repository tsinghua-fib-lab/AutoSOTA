import os, sys
import torch
import numpy as np
import pandas as pd

from script import train_and_evaluate, fmt_mean_std, build_row

# Set CUBLAS for deterministic ops
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

print('='*70)
print('HICALD Reproduction - ICALD on Heart-Disease')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('='*70)

# Rubric parameters:
# beta=0.1  -> lambda_reg=0.9  (1 - beta = lambda_reg -> 1-0.1=0.9)
# t0=0.5    -> t=0.5
# K=5       -> hardcoded in ICALD loss
# lr=1e-3, weight_decay=1e-4, batch_size=128
# n_epochs=600, early_stopping_patience=50
# train_split=0.7, val_split=0.1, test_split=0.2 (default)
# n_runs=5

metrics = train_and_evaluate(
    dataset_str='heart-disease',
    model_str='ICALD_Classifier',
    epochs=600,
    batch_size=128,
    lr=1e-3,
    num_runs=5,
    lambda_reg=0.9,    # corresponds to beta=0.1
    t=0.5,             # corresponds to t0=0.5
    lambda_kce=0.1,    # not used for ICALD (only MMD models)
    sigma2_z=None,
    early_stop=True,
    patience=50,
    min_epochs=20,
)

print()
print('='*70)
print('REPRODUCTION RESULTS')
print('='*70)

for k in sorted(metrics.keys()):
    mean, std = metrics[k]
    if k in ['Accuracy', 'AP', 'AUC']:
        print(f'{k:15s}: {mean*100:.3f} +/- {std*100:.3f}')
    else:
        print(f'{k:15s}: {mean:.6f} +/- {std:.6f}')

# Save results
row = build_row('heart-disease', 'ICALD_Classifier', 600, metrics)
df = pd.DataFrame([row])
csv_path = 'reproduction_results.csv'
df.to_csv(csv_path, index=False)
print(f'\nResults saved to {csv_path}')

# Also write a simple summary for the manifest
with open('reproduction_summary.txt', 'w') as f:
    for k in sorted(metrics.keys()):
        mean, std = metrics[k]
        if k in ['Accuracy', 'AP', 'AUC']:
            f.write(f'{k}: {mean*100:.3f} +/- {std*100:.3f}\n')
        else:
            f.write(f'{k}: {mean:.6f} +/- {std:.6f}\n')

print('Done!')

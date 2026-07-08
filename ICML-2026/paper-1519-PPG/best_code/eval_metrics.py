#!/usr/bin/env python3
"""Extract reproduction metrics from PePG experiment output."""
import json
import glob
import sys

def get_latest_output():
    files = sorted(glob.glob('data/outputs_*.json'))
    if not files:
        print('ERROR: No output files found in data/')
        sys.exit(1)
    return files[-1]

def extract_metrics(filepath):
    with open(filepath) as f:
        data = json.load(f)
    result = data[0]
    v_mean = result['v_values_mean'][-1]
    v_std = result['v_values_std'][-1]
    d_mean = result['d_diff_mean'][-1]
    d_std = result['d_diff_std'][-1]
    return {
        'Expected_Average_Return_Vpipi': round(v_mean, 6),
        'Expected_Average_Return_Vpipi_std': round(v_std, 6),
        'Policy_Stability_d_diff': round(d_mean, 8),
        'Policy_Stability_d_diff_std': round(d_std, 8),
        'iterations': len(result['v_values_mean']),
        'config': {k: v for k, v in result.items() if not isinstance(v, list)},
    }

if __name__ == '__main__':
    fp = get_latest_output()
    print(f'Reading: {fp}')
    metrics = extract_metrics(fp)
    print(json.dumps(metrics, indent=2))

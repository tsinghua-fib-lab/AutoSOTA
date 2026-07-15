
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Simulate running from root
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

# Cell 1: Setup
sys.path.append(os.path.abspath('scripts'))

# Cell 3: Load Data
RESULT_FILE = 'data/results/experiment_results_Llama-3.2-1B-Instruct.json'

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['valid'])} valid and {len(data['invalid'])} invalid proofs.")
else:
    print("Results file not found. Please run an experiment first!")
    sys.exit(1)

# Cell 4: Plotting Logic
def extract_metric(data, metric='hfer', layer=12):
    vals = []
    for item in data:
        traj = item['trajectory']
        if layer < len(traj):
             val = traj[layer].get(metric)
             if val is not None: vals.append(val)
    return vals

valid_vals = extract_metric(data['valid'])
invalid_vals = extract_metric(data['invalid'])

plt.figure(figsize=(10,6))
plt.hist(valid_vals, bins=30, alpha=0.5, label='Valid', density=True)
plt.hist(invalid_vals, bins=30, alpha=0.5, label='Invalid', density=True)
plt.title("HFER Distribution (Layer 12)")
plt.legend()
# Replace show() with savefig for verification
plt.savefig('output/notebook_verification_plot.png')
print("Saved notebook_verification_plot.png")
